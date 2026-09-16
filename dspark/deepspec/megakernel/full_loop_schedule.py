from __future__ import annotations

from dataclasses import dataclass

from deepspec.megakernel.contract import DeepSeekV4MegaKernelSpec
from deepspec.megakernel.v4_abi import V4LaunchShape
from deepspec.megakernel.v4_schedule import (
    V4PhaseKind,
    V4WorkerRole,
    build_v4_phase_program,
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class FullLoopSpec:
    """Frozen GB300 experiment boundary for one complete decode engine.

    The persistent launch begins after prompt KV bootstrap, matching the
    decode-only SGLang comparison boundary. It owns every subsequent target
    and grafted-speculator operation until EOS, length, or an explicit abort.
    """

    target_layers: int = 43
    graft_layers: int = 3
    target_tap_layers: tuple[int, ...] = (40, 41, 42)
    verify_rows: int = 6
    draft_tokens: int = 5
    hidden_size: int = 4096
    vocab_size: int = 129280
    attention_heads: int = 64
    head_dim: int = 512
    routed_experts: int = 256
    experts_per_token: int = 6
    expert_intermediate_size: int = 2048
    workers: int = 152
    threads: int = 256
    checkpoint_tensors: int = 72317
    checkpoint_bytes: int = 166878536440
    checkpoint_revision: str = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"

    def require_supported(self) -> None:
        expected = type(self)()
        if self != expected:
            raise ValueError(f"unsupported full-loop experiment spec: {self!r}")


@dataclass(frozen=True)
class FullLoopPhase:
    phase_id: int
    name: str
    kind: V4PhaseKind
    dependencies: tuple[str, ...]
    work_units: int
    roles: tuple[V4WorkerRole, ...]
    domain: str
    layer: int = -1


class _Builder:
    def __init__(self) -> None:
        self.phases: list[FullLoopPhase] = []
        self.names: set[str] = set()

    def add(
        self,
        name: str,
        kind: V4PhaseKind,
        dependencies: tuple[str, ...],
        work_units: int,
        roles: tuple[V4WorkerRole, ...],
        *,
        domain: str,
        layer: int = -1,
    ) -> str:
        if name in self.names:
            raise ValueError(f"duplicate full-loop phase {name}")
        unknown = [dependency for dependency in dependencies if dependency not in self.names]
        if unknown:
            raise ValueError(f"phase {name} has unknown dependencies: {unknown}")
        if work_units <= 0:
            raise ValueError(f"phase {name} has no work")
        self.names.add(name)
        self.phases.append(
            FullLoopPhase(
                phase_id=len(self.phases),
                name=name,
                kind=kind,
                dependencies=dependencies,
                work_units=work_units,
                roles=roles,
                domain=domain,
                layer=layer,
            )
        )
        return name


_GEMM = (V4WorkerRole.TMA, V4WorkerRole.MMA, V4WorkerRole.EPILOGUE)
_VECTOR = (V4WorkerRole.EPILOGUE,)
_ATTENTION = (V4WorkerRole.TMA, V4WorkerRole.MMA, V4WorkerRole.EPILOGUE)
_CONTROL = (V4WorkerRole.CONTROLLER,)


def build_full_loop_program(spec: FullLoopSpec | None = None) -> tuple[FullLoopPhase, ...]:
    """Build one target-verify + grafted-proposal iteration of the decode loop.

    The returned graph is a DAG for one iteration. ``loop.commit_and_continue``
    advances the device epoch and feeds the next iteration inside the kernel;
    the back-edge intentionally lives in the persistent controller rather than
    in this topological phase table.
    """

    spec = spec or FullLoopSpec()
    spec.require_supported()
    dspark = DeepSeekV4MegaKernelSpec()
    dspark.require_supported()
    if dspark.num_target_layers != spec.target_layers:
        raise ValueError("target-layer count drifted from the validated DSpark contract")
    if dspark.num_draft_layers != spec.graft_layers:
        raise ValueError("graft-layer count drifted from the validated DSpark contract")

    builder = _Builder()
    input_ready = builder.add(
        "loop.input_ready",
        V4PhaseKind.SCHEDULER,
        (),
        1,
        _CONTROL,
        domain="control",
    )
    hidden_ready = builder.add(
        "target.embedding_hc_expand",
        V4PhaseKind.VECTOR,
        (input_ready,),
        _ceil_div(spec.verify_rows * 4 * spec.hidden_size, 512),
        (V4WorkerRole.TMA, V4WorkerRole.EPILOGUE),
        domain="target",
    )
    taps: list[str] = []
    compressed_ratios = (0, 0) + tuple(4 if layer % 2 == 0 else 128 for layer in range(2, 43))

    for layer in range(spec.target_layers):
        prefix = f"target.layer_{layer}"
        attn_mix = builder.add(
            f"{prefix}.attn_hc_projection",
            V4PhaseKind.GEMM,
            (hidden_ready,),
            spec.verify_rows * 24,
            _GEMM,
            domain="target",
            layer=layer,
        )
        attn_sinkhorn = builder.add(
            f"{prefix}.attn_hc_sinkhorn_reduce_norm",
            V4PhaseKind.VECTOR,
            (attn_mix,),
            spec.verify_rows,
            _VECTOR,
            domain="target",
            layer=layer,
        )
        q_a = builder.add(
            f"{prefix}.q_a",
            V4PhaseKind.GEMM,
            (attn_sinkhorn,),
            32,
            _GEMM,
            domain="target",
            layer=layer,
        )
        q_norm = builder.add(
            f"{prefix}.q_norm_q_b_rope",
            V4PhaseKind.GEMM,
            (q_a,),
            256,
            _GEMM,
            domain="target",
            layer=layer,
        )
        kv = builder.add(
            f"{prefix}.kv_projection_norm_rope_quant",
            V4PhaseKind.GEMM,
            (attn_sinkhorn,),
            16,
            _GEMM,
            domain="target",
            layer=layer,
        )
        attention_dependencies = [q_norm, kv]
        ratio = compressed_ratios[layer]
        if ratio:
            compressed_kv = builder.add(
                f"{prefix}.compressed_kv_r{ratio}",
                V4PhaseKind.ATTENTION,
                (attn_sinkhorn,),
                spec.verify_rows,
                _ATTENTION,
                domain="target",
                layer=layer,
            )
            attention_dependencies.append(compressed_kv)
            if ratio == 4:
                attention_dependencies.append(
                    builder.add(
                        f"{prefix}.index_top512",
                        V4PhaseKind.ATTENTION,
                        (attn_sinkhorn, q_a),
                        spec.attention_heads,
                        _ATTENTION,
                        domain="target",
                        layer=layer,
                    )
                )
        attention = builder.add(
            f"{prefix}.sparse_attention",
            V4PhaseKind.ATTENTION,
            tuple(attention_dependencies),
            spec.verify_rows * 8,
            _ATTENTION,
            domain="target",
            layer=layer,
        )
        wo_a = builder.add(
            f"{prefix}.inverse_rope_grouped_wo_a",
            V4PhaseKind.GEMM,
            (attention,),
            64,
            _GEMM,
            domain="target",
            layer=layer,
        )
        wo_b = builder.add(
            f"{prefix}.wo_b_attn_hc_post",
            V4PhaseKind.GEMM,
            (wo_a,),
            128,
            _GEMM,
            domain="target",
            layer=layer,
        )
        ffn_mix = builder.add(
            f"{prefix}.ffn_hc_projection_sinkhorn_norm",
            V4PhaseKind.GEMM,
            (wo_b,),
            spec.verify_rows * 24,
            _GEMM,
            domain="target",
            layer=layer,
        )
        router = builder.add(
            f"{prefix}.router_sqrtsoftplus_top6",
            V4PhaseKind.VECTOR,
            (ffn_mix,),
            spec.verify_rows * 16,
            _VECTOR,
            domain="target",
            layer=layer,
        )
        routed_w13 = builder.add(
            f"{prefix}.fp4_routed_w13",
            V4PhaseKind.EXPERT,
            (router,),
            spec.verify_rows * spec.experts_per_token * 8,
            _GEMM,
            domain="target",
            layer=layer,
        )
        routed_act = builder.add(
            f"{prefix}.routed_swiglu",
            V4PhaseKind.VECTOR,
            (routed_w13,),
            spec.verify_rows * spec.experts_per_token,
            _VECTOR,
            domain="target",
            layer=layer,
        )
        routed_w2 = builder.add(
            f"{prefix}.fp4_routed_w2",
            V4PhaseKind.EXPERT,
            (routed_act,),
            spec.verify_rows * spec.experts_per_token * 8,
            _GEMM,
            domain="target",
            layer=layer,
        )
        shared_w13 = builder.add(
            f"{prefix}.fp8_shared_w13",
            V4PhaseKind.EXPERT,
            (ffn_mix,),
            32,
            _GEMM,
            domain="target",
            layer=layer,
        )
        shared_act = builder.add(
            f"{prefix}.shared_swiglu",
            V4PhaseKind.VECTOR,
            (shared_w13,),
            spec.verify_rows * 4,
            _VECTOR,
            domain="target",
            layer=layer,
        )
        shared_w2 = builder.add(
            f"{prefix}.fp8_shared_w2",
            V4PhaseKind.EXPERT,
            (shared_act,),
            32,
            _GEMM,
            domain="target",
            layer=layer,
        )
        hidden_ready = builder.add(
            f"{prefix}.expert_combine_hc_post",
            V4PhaseKind.VECTOR,
            (routed_w2, shared_w2, ffn_mix),
            _ceil_div(spec.verify_rows * 4 * spec.hidden_size, 512),
            _VECTOR,
            domain="target",
            layer=layer,
        )
        if layer in spec.target_tap_layers:
            taps.append(
                builder.add(
                    f"{prefix}.capture_dspark_tap",
                    V4PhaseKind.VECTOR,
                    (hidden_ready,),
                    _ceil_div(spec.verify_rows * spec.hidden_size, 512),
                    _VECTOR,
                    domain="target",
                    layer=layer,
                )
            )

    target_head = builder.add(
        "target.head_hc_reduce_rmsnorm_lm",
        V4PhaseKind.GEMM,
        (hidden_ready,),
        _ceil_div(spec.vocab_size, 128),
        _GEMM,
        domain="target",
    )
    accepted = builder.add(
        "target.verify_accept_sample_commit",
        V4PhaseKind.SCHEDULER,
        (target_head,),
        spec.verify_rows,
        _CONTROL,
        domain="target",
    )
    select_taps = builder.add(
        "graft.select_accepted_target_taps",
        V4PhaseKind.VECTOR,
        (accepted, *taps),
        len(spec.target_tap_layers) * _ceil_div(spec.hidden_size, 512),
        _VECTOR,
        domain="graft",
    )
    accepted_state = builder.add(
        "loop.advance_accepted_state",
        V4PhaseKind.SCHEDULER,
        (select_taps,),
        1,
        _CONTROL,
        domain="control",
    )
    graft_inputs = builder.add(
        "graft.prepare_inputs",
        V4PhaseKind.VECTOR,
        (accepted_state,),
        spec.verify_rows,
        _VECTOR,
        domain="graft",
    )

    graft_program = build_v4_phase_program(
        dspark,
        V4LaunchShape(batch_size=1, main_rows=1),
        relaxed_tail=True,
        greedy_tail=True,
    )
    graft_names = {phase.name for phase in graft_program}
    for phase in graft_program:
        dependencies = tuple(f"graft.{name}" for name in phase.dependencies)
        if not dependencies:
            dependencies = (graft_inputs,)
        if any(name not in graft_names for name in phase.dependencies):
            raise ValueError(f"invalid graft dependency in {phase.name}")
        builder.add(
            f"graft.{phase.name}",
            phase.kind,
            dependencies,
            phase.work_units,
            phase.roles,
            domain="graft",
        )

    builder.add(
        "loop.commit_and_continue",
        V4PhaseKind.SCHEDULER,
        ("graft.proposal.finalize_audit",),
        1,
        _CONTROL,
        domain="control",
    )
    return tuple(builder.phases)


def full_loop_macro_name(phase: FullLoopPhase) -> str:
    """Return the physical band that owns one semantic phase.

    Keeping this map beside the semantic DAG makes fusion a compiler policy:
    numerical kernels never need to know how Python grouped the model graph.
    """

    name = phase.name
    if name in {
        "loop.input_ready",
        "loop.advance_accepted_state",
        "loop.commit_and_continue",
    }:
        return name
    if name == "target.embedding_hc_expand":
        return "target.bootstrap"
    if name.startswith("target.layer_"):
        prefix, _ = name.rsplit(".", 1)
        # One target-layer opcode owns the mandatory attention-to-FFN barriers
        # internally. The fine graph remains available for numerical checking,
        # but exposing its subregions to the outer scheduler only adds events.
        return f"{prefix}.execute"
    if name in {"target.head_hc_reduce_rmsnorm_lm", "target.verify_accept_sample_commit"}:
        return "target.head_verify_accept"
    if name == "graft.select_accepted_target_taps":
        return "graft.target_inject"
    if name == "graft.prepare_inputs":
        return "graft.input_prepare"
    if name.startswith("graft."):
        # The already-validated standalone speculator is itself one compiled
        # device program. Inline that program as one outer opcode instead of
        # recreating its internal scheduler in the target+graft controller.
        return "graft.execute"
    raise ValueError(f"full-loop phase has no macro-band: {name}")


def build_full_loop_macro_program(
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopPhase, ...]:
    """AOT runtime bands; fine nodes execute inside these fixed macro bodies."""

    detailed = build_full_loop_program(spec)
    grouped: dict[str, list[FullLoopPhase]] = {}
    order: list[str] = []
    for phase in detailed:
        macro = full_loop_macro_name(phase)
        if macro not in grouped:
            grouped[macro] = []
            order.append(macro)
        grouped[macro].append(phase)

    priority = {
        V4PhaseKind.SCHEDULER: 0,
        V4PhaseKind.REDUCTION: 1,
        V4PhaseKind.VECTOR: 2,
        V4PhaseKind.GEMM: 3,
        V4PhaseKind.ATTENTION: 4,
        V4PhaseKind.EXPERT: 5,
    }
    macros = []
    for phase_id, name in enumerate(order):
        members = grouped[name]
        roles = tuple(
            role for role in V4WorkerRole if any(role in member.roles for member in members)
        )
        macros.append(
            FullLoopPhase(
                phase_id=phase_id,
                name=name,
                kind=max((member.kind for member in members), key=priority.__getitem__),
                dependencies=() if phase_id == 0 else (order[phase_id - 1],),
                work_units=sum(member.work_units for member in members),
                roles=roles,
                domain=members[0].domain,
                layer=members[0].layer,
            )
        )
    return tuple(macros)
