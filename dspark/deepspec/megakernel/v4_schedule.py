from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, IntEnum

from deepspec.megakernel.contract import DeepSeekV4MegaKernelSpec


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


# R12 router chain geometry. MUST equal the DSPARK_ROUTER_CHAINS /
# DSPARK_ROUTER_ROWS the relaxed kernel is compiled with -- the generated
# header carries both numbers and dspark_v4_kernel.cu static_asserts them, so
# a mismatch is a build error rather than silently dropped chains.
#
# The shipped (8, 1) K-splits each 4,096-step chain across 32 threads, which
# turns the band's 32 work items of 40 live threads into 160 items of 256:
# 218.4 -> 35.8 us on-path over three layers, measured same-session. The
# frozen #101 tiling is (40, 1) and stays exactly selectable as the control.
# Contract programs never read this -- they keep `ceil(256 / 8)`.
_ROUTER_CHAINS_PER_ITEM = int(os.environ.get("DSPARK_ROUTER_CHAINS") or 8)
_ROUTER_ROWS_PER_CHAIN = int(os.environ.get("DSPARK_ROUTER_ROWS") or 1)


def router_chain_geometry() -> tuple[int, int]:
    """(chains per work item, rows per chain) the relaxed program is built for."""

    return _ROUTER_CHAINS_PER_ITEM, _ROUTER_ROWS_PER_CHAIN


def set_router_chain_geometry(chains: int, rows: int) -> None:
    """Sweep hook: retarget the relaxed router item count in this process.

    Used only by `scripts/megakernel/v4_batch_bench.py --router-sweep`, which
    walks the split-factor curve against its own in-run control and must
    rebuild the generated program for each point. Production callers take the
    module defaults.
    """

    global _ROUTER_CHAINS_PER_ITEM, _ROUTER_ROWS_PER_CHAIN
    _ROUTER_CHAINS_PER_ITEM = int(chains)
    _ROUTER_ROWS_PER_CHAIN = int(rows)


class V4WorkerRole(IntEnum):
    CONTROLLER = 0
    TMA = 1
    MMA = 2
    EPILOGUE = 3

    @property
    def warps(self) -> int:
        return {
            V4WorkerRole.CONTROLLER: 1,
            V4WorkerRole.TMA: 1,
            V4WorkerRole.MMA: 4,
            V4WorkerRole.EPILOGUE: 2,
        }[self]


class V4PhaseKind(str, Enum):
    GEMM = "gemm"
    VECTOR = "vector"
    ATTENTION = "attention"
    EXPERT = "expert"
    REDUCTION = "reduction"
    SCHEDULER = "scheduler"


@dataclass(frozen=True)
class V4Phase:
    phase_id: int
    name: str
    kind: V4PhaseKind
    dependencies: tuple[str, ...]
    work_units: int
    roles: tuple[V4WorkerRole, ...]
    synchronization: str = "release_acquire_dependency_counter"


class _ProgramBuilder:
    def __init__(self):
        self.phases: list[V4Phase] = []
        self.names: set[str] = set()

    def add(
        self,
        name: str,
        kind: V4PhaseKind,
        dependencies: tuple[str, ...],
        work_units: int,
        roles: tuple[V4WorkerRole, ...],
    ) -> str:
        if name in self.names:
            raise ValueError(f"duplicate V4 phase {name}")
        unknown = [dependency for dependency in dependencies if dependency not in self.names]
        if unknown:
            raise ValueError(f"phase {name} has unknown or forward dependencies: {unknown}")
        if work_units <= 0:
            raise ValueError(f"phase {name} has no work")
        self.names.add(name)
        self.phases.append(
            V4Phase(
                phase_id=len(self.phases),
                name=name,
                kind=kind,
                dependencies=dependencies,
                work_units=work_units,
                roles=roles,
            )
        )
        return name


_GEMM_ROLES = (V4WorkerRole.TMA, V4WorkerRole.MMA, V4WorkerRole.EPILOGUE)
_VECTOR_ROLES = (V4WorkerRole.EPILOGUE,)
_ATTENTION_ROLES = (V4WorkerRole.TMA, V4WorkerRole.MMA, V4WorkerRole.EPILOGUE)
_SCHEDULER_ROLES = (V4WorkerRole.CONTROLLER,)


def _gemm_tiles(m: int, n: int, k: int, *, split_k: int = 1) -> int:
    del k  # K affects the tile body, while split_k captures scheduler parallelism.
    return max(_ceil_div(m, 64) * _ceil_div(n, 128) * split_k, 1)


def _vector_tiles(elements: int) -> int:
    # A 512-element vector tile leaves at least two waves for the 129,280-wide
    # vocabulary phases on 148 B200 SMs.
    return max(_ceil_div(elements, 512), 1)


def build_v4_phase_program(
    spec: DeepSeekV4MegaKernelSpec,
    shape: object,
    relaxed_tail: bool = False,
    greedy_tail: bool = False,
    static_tp2_tail: bool = False,
    hc_split: bool = False,
    woa_split_k: int = 1,
    w13_fused_swiglu: bool = False,
    routed_group_ready: bool = False,
) -> tuple[V4Phase, ...]:
    """Specialize the fixed V4 graph into a device-resident task program.

    A persistent CTA owns one controller warp plus TMA, TCGen/MMA, and
    epilogue warp roles. Ready phases live in device queues. The last tile of a
    phase release-publishes successors whose atomic dependency count reaches
    zero; an idle controller probes local work and steals remote work. No host
    role, sleep, cooperative grid barrier, or nested launch exists in this
    program.
    """

    spec.require_supported()
    if static_tp2_tail and not relaxed_tail:
        raise ValueError("static_tp2_tail requires relaxed_tail=True")
    if static_tp2_tail and not greedy_tail:
        raise ValueError("static_tp2_tail requires greedy_tail=True")
    if greedy_tail and not relaxed_tail:
        raise ValueError("greedy_tail requires relaxed_tail=True")
    batch = int(getattr(shape, "batch_size"))
    if hc_split and not (relaxed_tail and greedy_tail and batch == 1):
        raise ValueError("HC split requires the batch-1 relaxed greedy program")
    if woa_split_k not in (1, 2) or (woa_split_k != 1 and not hc_split):
        raise ValueError("WO_A split-K2 requires the batch-1 HC split program")
    if w13_fused_swiglu and not hc_split:
        raise ValueError("W13 SwiGLU fusion requires the batch-1 HC split program")
    if routed_group_ready and not (w13_fused_swiglu and woa_split_k == 2):
        raise ValueError("group readiness requires fused W13/SwiGLU with WO_A split2")
    main_rows = int(getattr(shape, "main_rows"))
    block = spec.block_size
    hidden = spec.hidden_size
    builder = _ProgramBuilder()

    main_quant = builder.add(
        "main.activation_quant",
        V4PhaseKind.VECTOR,
        (),
        batch * main_rows * spec.target_feature_width // 128,
        _VECTOR_ROLES,
    )
    main_projection = builder.add(
        "main.projection",
        V4PhaseKind.GEMM,
        (main_quant,),
        _gemm_tiles(batch * main_rows, hidden, spec.target_feature_width, split_k=8),
        _GEMM_ROLES,
    )
    main_reduce = builder.add(
        "main.split_k_reduce",
        V4PhaseKind.REDUCTION,
        (main_projection,),
        _vector_tiles(batch * main_rows * hidden),
        _VECTOR_ROLES,
    )
    main_norm = builder.add(
        "main.rmsnorm",
        V4PhaseKind.VECTOR,
        (main_reduce,),
        _vector_tiles(batch * main_rows * hidden),
        _VECTOR_ROLES,
    )
    embedding = builder.add(
        "draft.embedding_hc_expand",
        V4PhaseKind.VECTOR,
        (),
        _vector_tiles(batch * block * spec.hc_multiplier * hidden),
        (V4WorkerRole.TMA, V4WorkerRole.EPILOGUE),
    )
    main_projected_quant = builder.add(
        "main.projected_activation_quant",
        V4PhaseKind.VECTOR,
        (main_norm,),
        batch * main_rows * hidden // 128,
        _VECTOR_ROLES,
    )

    main_kv: list[str] = []
    for layer in range(spec.num_draft_layers):
        projected = builder.add(
            f"layer_{layer}.main_kv_projection",
            V4PhaseKind.GEMM,
            (main_projected_quant,),
            _gemm_tiles(batch * main_rows, spec.kv_width, hidden, split_k=4),
            _GEMM_ROLES,
        )
        main_kv.append(
            builder.add(
                f"layer_{layer}.main_kv_norm_rope_quant_store",
                V4PhaseKind.VECTOR,
                (projected,),
                _vector_tiles(batch * main_rows * spec.kv_width),
                _VECTOR_ROLES,
            )
        )

    hidden_ready = embedding
    for layer in range(spec.num_draft_layers):
        prefix = f"layer_{layer}"
        attn_mix = builder.add(
            f"{prefix}.attn_hc_projection",
            V4PhaseKind.GEMM,
            (hidden_ready,),
            # One task per (row, mix) output so the 16,384-wide reductions
            # spread across the grid instead of serializing on one CTA.
            batch * block * (2 + spec.hc_multiplier) * spec.hc_multiplier,
            _GEMM_ROLES,
        )
        attn_sinkhorn = builder.add(
            f"{prefix}.attn_hc_sinkhorn",
            V4PhaseKind.VECTOR,
            (attn_mix,),
            batch * block,
            _VECTOR_ROLES,
        )
        if relaxed_tail and not hc_split:
            # Relaxed program stage 2b: reduce_rmsnorm rides the sinkhorn
            # phase (identical 1-row tiling, direct chain) — one fewer
            # critical-path boundary per layer per band.
            attn_input = attn_sinkhorn
        else:
            attn_input = builder.add(
                f"{prefix}.attn_hc_reduce_rmsnorm",
                V4PhaseKind.VECTOR,
                (attn_mix,) if hc_split else (attn_sinkhorn,),
                batch * block,
                _VECTOR_ROLES,
            )
        # Batched serving (R3): the small dense bands tile OVER BATCH -- one
        # item is one (batch element, output tile, split) triple -- because
        # their weights are small (wq_a 4 MiB, wkv 2 MiB, wo_a 4 MiB) and
        # widening their UMMA N-mode would need a per-body atom/layout/TMEM
        # retemplate for little traffic. The weight-bound bands (LM head
        # 1.06 GiB, markov_w2 66 MiB x 5, main projection 50 MiB) keep a
        # BATCH-CONSTANT item count and carry the rows on the tensor-core N
        # mode instead, so their weights stream exactly once per proposal.
        #
        # `batch * _gemm_tiles(block, ...)` equals the retired
        # `_gemm_tiles(batch * block, ...)` at batch 1 (5 rows are one 64-row
        # M tile), so the batch-1 program is unchanged tile for tile.
        q_a = builder.add(
            f"{prefix}.q_a",
            V4PhaseKind.GEMM,
            (attn_input,),
            # R4 band 4: N-widened, batch-INVARIANT item count.
            _gemm_tiles(block, spec.q_lora_rank, hidden, split_k=4)
            * (1 if relaxed_tail else batch),
            _GEMM_ROLES,
        )
        q_a_norm = builder.add(
            f"{prefix}.q_a_rmsnorm",
            V4PhaseKind.VECTOR,
            (q_a,),
            batch * block,
            _VECTOR_ROLES,
        )
        q_b = builder.add(
            f"{prefix}.q_b",
            V4PhaseKind.GEMM,
            (q_a_norm,),
            # R4 band 2: q_b is WEIGHT-BOUND (33.5 MiB of FP8), so the relaxed
            # body carries all 5*batch flat rows on the tensor-core N mode and
            # the item count is batch-INVARIANT at 256 output tiles.
            _gemm_tiles(block, spec.query_width, spec.q_lora_rank) * (1 if relaxed_tail else batch),
            _GEMM_ROLES,
        )
        draft_kv_projection = builder.add(
            f"{prefix}.draft_kv_projection",
            V4PhaseKind.GEMM,
            (attn_input,),
            # R4 band 5: N-widened, batch-INVARIANT item count.
            _gemm_tiles(block, spec.kv_width, hidden, split_k=4) * (1 if relaxed_tail else batch),
            _GEMM_ROLES,
        )
        q_ready = builder.add(
            f"{prefix}.q_norm_rope",
            V4PhaseKind.VECTOR,
            (q_b,),
            # Relaxed stage-3 iter-9: bitwise batched<8> body — one item is
            # 8 heads of one row, so 320 (row,head) tiles become 40.
            _vector_tiles(batch * block * spec.query_width) // (8 if relaxed_tail else 1),
            _VECTOR_ROLES,
        )
        draft_kv_ready = builder.add(
            f"{prefix}.draft_kv_norm_rope_quant",
            V4PhaseKind.VECTOR,
            (draft_kv_projection,),
            _vector_tiles(batch * block * spec.kv_width),
            _VECTOR_ROLES,
        )
        attention = builder.add(
            f"{prefix}.sparse_attention",
            V4PhaseKind.ATTENTION,
            (q_ready, draft_kv_ready, main_kv[layer]),
            # Relaxed stage-3 iter-4: one item = 8 heads of one row (the
            # batched8p body, bitwise-identical to the reference), so 320
            # (row, head) items become 40 (row, head-group) items.
            batch * block * spec.num_attention_heads // (8 if relaxed_tail else 1),
            _ATTENTION_ROLES,
        )
        output_a = builder.add(
            f"{prefix}.inverse_rope_grouped_wo_a",
            V4PhaseKind.GEMM,
            (attention,),
            # #102 retile: (group, rank_tile, row) triples — 320 tiles instead
            # of 64; chains and 128-rank quantization groups are unchanged.
            # Relaxed stage-3 iter-5: the UMMA body carries the 5 rows on the
            # tensor-core N dim, so items are (group, rank_tile) pairs = 64.
            #
            # R4 band 1: wo_a is WEIGHT-BOUND (64 MiB read once per item
            # sweep), so the relaxed body now carries ALL 5*batch flat rows on
            # the tensor-core N mode and the item count is batch-INVARIANT.
            # The contract tiling (relaxed_tail=False) is batch-1 only and
            # keeps its (batch * block) row factor, so both batch-1 programs
            # are unchanged tile for tile.
            spec.output_groups
            * _ceil_div(spec.output_lora_rank, 128)
            * (woa_split_k if relaxed_tail else batch * block),
            _GEMM_ROLES,
        )
        if woa_split_k != 1:
            # WO_B reuses these partials only after every reduction completes.
            output_a = builder.add(
                f"{prefix}.wo_a_reduce_quant",
                V4PhaseKind.VECTOR,
                (output_a,),
                spec.output_groups * _ceil_div(spec.output_lora_rank, 128),
                _VECTOR_ROLES,
            )
        output_b = builder.add(
            f"{prefix}.wo_b",
            V4PhaseKind.GEMM,
            (output_a,),
            # R4 band 3: wo_b is WEIGHT-BOUND (33.5 MiB of FP8 read once per
            # item sweep), so the relaxed body carries all 5*batch flat rows
            # on the tensor-core N mode and the item count is batch-INVARIANT.
            _gemm_tiles(
                block,
                hidden,
                spec.output_groups * spec.output_lora_rank,
                split_k=4,
            )
            * (1 if relaxed_tail else batch),
            _GEMM_ROLES,
        )
        attn_post = builder.add(
            f"{prefix}.attn_hc_post",
            V4PhaseKind.VECTOR,
            (output_b, attn_sinkhorn),
            _vector_tiles(batch * block * hidden),
            _VECTOR_ROLES,
        )
        ffn_mix = builder.add(
            f"{prefix}.ffn_hc_projection",
            V4PhaseKind.GEMM,
            (attn_post,),
            batch * block * (2 + spec.hc_multiplier) * spec.hc_multiplier,
            _GEMM_ROLES,
        )
        ffn_sinkhorn = builder.add(
            f"{prefix}.ffn_hc_sinkhorn",
            V4PhaseKind.VECTOR,
            (ffn_mix,),
            batch * block,
            _VECTOR_ROLES,
        )
        if relaxed_tail and not hc_split:
            ffn_input = ffn_sinkhorn
        else:
            ffn_input = builder.add(
                f"{prefix}.ffn_hc_reduce_rmsnorm",
                V4PhaseKind.VECTOR,
                (ffn_mix,) if hc_split else (ffn_sinkhorn,),
                batch * block,
                _VECTOR_ROLES,
            )
        router_scores = builder.add(
            f"{prefix}.router_sqrtsoftplus_top6",
            V4PhaseKind.GEMM,
            (ffn_input,),
            # #101 retile: 8 experts per tile, every (expert, row) chain on
            # its own thread — 32 claimable tiles instead of the 2-tile
            # 256x4096 GEMV crater that idled 146 CTAs (~0.7 ms x 3 layers).
            #
            # Batched serving (R3): this band MUST carry a batch factor. Its
            # body runs 8 experts x block rows as one dependent per-thread
            # chain each, so without the factor a batch-B launch would need
            # 8 * 5 * B > 256 threads and would silently drop rows at B > 6.
            #
            # R12: the relaxed build K-splits each chain, so its item count is
            # derived from the chain geometry the kernel is compiled with
            # (router_chain_geometry() / DSPARK_ROUTER_CHAINS,
            # DSPARK_ROUTER_ROWS). The default geometry (40, 1) reproduces
            # `batch * ceil(256 / 8)` exactly, so the checked-in headers are
            # unchanged until the geometry is changed on purpose. The contract
            # program never sees the knob.
            (
                batch
                * (spec.num_routed_experts * (block // _ROUTER_ROWS_PER_CHAIN))
                // _ROUTER_CHAINS_PER_ITEM
                if relaxed_tail
                else batch * _ceil_div(spec.num_routed_experts, 8)
            ),
            _GEMM_ROLES,
        )
        router = builder.add(
            f"{prefix}.router_top6",
            V4PhaseKind.REDUCTION,
            (router_scores,),
            batch * block,
            _VECTOR_ROLES,
        )
        routed_w13 = builder.add(
            f"{prefix}.fp4_routed_w13",
            V4PhaseKind.EXPERT,
            (router,),
            batch
            * block
            * spec.num_activated_experts
            * _ceil_div(2 * spec.expert_intermediate_size, 128),
            _GEMM_ROLES,
        )
        if w13_fused_swiglu:
            routed_swiglu = routed_w13
        else:
            routed_swiglu = builder.add(
                f"{prefix}.routed_swiglu",
                V4PhaseKind.VECTOR,
                (routed_w13,),
                _vector_tiles(
                    batch * block * spec.num_activated_experts * spec.expert_intermediate_size
                ),
                _VECTOR_ROLES,
            )
        routed = builder.add(
            f"{prefix}.fp4_routed_w2",
            V4PhaseKind.EXPERT,
            # Group-local GPU counters guard each W2 operand independently.
            (router if routed_group_ready else routed_swiglu,),
            batch * block * spec.num_activated_experts * _ceil_div(spec.hidden_size, 128),
            _GEMM_ROLES,
        )
        shared_w13 = builder.add(
            f"{prefix}.fp8_shared_w13",
            V4PhaseKind.EXPERT,
            (ffn_input,),
            # R4 band 7: N-widened, batch-INVARIANT item count.
            _ceil_div(2 * spec.expert_intermediate_size, 128) * (1 if relaxed_tail else batch),
            _GEMM_ROLES,
        )
        shared_swiglu = builder.add(
            f"{prefix}.shared_swiglu",
            V4PhaseKind.VECTOR,
            (shared_w13,),
            _vector_tiles(batch * block * spec.expert_intermediate_size),
            _VECTOR_ROLES,
        )
        shared = builder.add(
            f"{prefix}.fp8_shared_w2",
            V4PhaseKind.EXPERT,
            (shared_swiglu,),
            # R4 band 6: N-widened, batch-INVARIANT item count.
            _ceil_div(spec.hidden_size, 128) * (1 if relaxed_tail else batch),
            _GEMM_ROLES,
        )
        expert_sum = builder.add(
            f"{prefix}.expert_combine_hc_post",
            V4PhaseKind.VECTOR,
            (routed, shared, ffn_sinkhorn),
            _vector_tiles(batch * block * spec.mhc_width),
            _VECTOR_ROLES,
        )
        hidden_ready = expert_sum

    head_reduce = builder.add(
        "head.hc_reduce_rmsnorm",
        V4PhaseKind.VECTOR,
        (hidden_ready,),
        # The body is per-ROW (one 148-CTA-wide item does the whole 16,384
        # RMS tree, the four serialized per-stream dots, the gate and the
        # second RMS for one draft row) and early-outs on item >= 5, so the
        # 512-element vector tiling claimed 155 no-op tiles per proposal.
        # The relaxed program tiles it at its real item space.
        batch * block if relaxed_tail else _vector_tiles(batch * block * spec.mhc_width),
        _VECTOR_ROLES,
    )
    # Measured static-TP2 ownership: rank 1 has compute headroom while rank 0
    # remains on the proposal critical path. Keep the split tile-aligned and
    # give rank 0 only 40% of LM/Markov-W2 vocabulary work.
    tail_vocab_size = spec.vocab_size * 2 // 5 if static_tp2_tail else spec.vocab_size
    lm_rows = [
        builder.add(
            "head.lm_row_0",
            V4PhaseKind.GEMM,
            (head_reduce,),
            _gemm_tiles(batch, tail_vocab_size, hidden, split_k=1),
            _GEMM_ROLES,
        )
    ]
    if greedy_tail:
        # LM row 0's tensor-core body materializes all five rows together.
        # The four historical one-unit row phases are dependency aliases only.
        lm_rows.extend([lm_rows[0]] * (block - 1))
    else:
        for step in range(1, block):
            lm_rows.append(
                builder.add(
                    f"head.lm_row_{step}",
                    V4PhaseKind.GEMM,
                    (lm_rows[0],),
                    1,
                    _GEMM_ROLES,
                )
            )

    prior_sample: str | None = None
    confidence_ready: list[str] = []
    for step in range(block):
        prefix = f"tail_{step}"
        previous_token_dependency = () if prior_sample is None else (prior_sample,)
        if relaxed_tail:
            # Relaxed-numerics DAG (acceptance-parity program, run-notes
            # relaxed-drafter/PLAN.md stage 2): the tail's cost is phase
            # boundaries, not bytes. Per step, 7 phases collapse to 4:
            # confidence fuses into the gather (both single-tile, chained),
            # correct+argmax fuses into markov_w2's epilogue (order-free
            # packed atomicMax per #102), and sum_reduce's total/tile
            # selection recomputes redundantly inside normalize.
            markov_gather = builder.add(
                f"{prefix}.markov_gather",
                V4PhaseKind.GEMM,
                previous_token_dependency + (head_reduce,),
                batch,
                _GEMM_ROLES,
            )
            confidence_ready.append(markov_gather)
            markov_logits = builder.add(
                f"{prefix}.markov_w2",
                V4PhaseKind.GEMM,
                (markov_gather, lm_rows[step]),
                _gemm_tiles(batch, tail_vocab_size, spec.markov_rank),
                _GEMM_ROLES,
            )
            if greedy_tail:
                # The fused Markov phase already computes a packed global
                # argmax. Its last-completion controller publishes the chosen
                # token before successors, so T=0 serving needs neither the
                # no-op exp phase nor full-vocabulary one-hot probabilities.
                prior_sample = markov_logits
                continue
            exp_sum = builder.add(
                f"{prefix}.softmax_exp_sum",
                V4PhaseKind.VECTOR,
                (markov_logits,),
                _vector_tiles(batch * spec.vocab_size),
                _VECTOR_ROLES,
            )
            prior_sample = builder.add(
                f"{prefix}.normalize_scan_sample",
                V4PhaseKind.VECTOR,
                (exp_sum,),
                _vector_tiles(batch * spec.vocab_size),
                _VECTOR_ROLES,
            )
            continue
        markov_gather = builder.add(
            f"{prefix}.markov_gather",
            V4PhaseKind.VECTOR,
            previous_token_dependency,
            batch,
            (V4WorkerRole.TMA, V4WorkerRole.EPILOGUE),
        )
        confidence_ready.append(
            builder.add(
                f"{prefix}.confidence_sts",
                V4PhaseKind.GEMM,
                (markov_gather, head_reduce),
                _gemm_tiles(batch, 1, hidden + spec.markov_rank),
                _GEMM_ROLES,
            )
        )
        markov_logits = builder.add(
            f"{prefix}.markov_w2",
            V4PhaseKind.GEMM,
            (markov_gather,),
            _gemm_tiles(batch, spec.vocab_size, spec.markov_rank),
            _GEMM_ROLES,
        )
        # #102: the global argmax is fused into correct_logits_partial_max via
        # a packed (value, lowest-index) atomicMax — float max is
        # order-invariant, so the fusion is bit-exact and the five 1-tile
        # max_reduce serialization points disappear from the chain.
        corrected = builder.add(
            f"{prefix}.correct_logits_partial_max",
            V4PhaseKind.VECTOR,
            (markov_logits, lm_rows[step]),
            _vector_tiles(batch * spec.vocab_size),
            _VECTOR_ROLES,
        )
        exp_sum = builder.add(
            f"{prefix}.softmax_exp_sum",
            V4PhaseKind.VECTOR,
            (corrected,),
            _vector_tiles(batch * spec.vocab_size),
            _VECTOR_ROLES,
        )
        total = builder.add(
            f"{prefix}.sum_reduce",
            V4PhaseKind.REDUCTION,
            (exp_sum,),
            batch,
            _VECTOR_ROLES,
        )
        prior_sample = builder.add(
            f"{prefix}.normalize_scan_sample",
            V4PhaseKind.VECTOR,
            (total,),
            _vector_tiles(batch * spec.vocab_size),
            _VECTOR_ROLES,
        )

    assert prior_sample is not None
    if greedy_tail:
        builder.add(
            "proposal.finalize_audit",
            V4PhaseKind.SCHEDULER,
            (prior_sample,),
            1,
            _SCHEDULER_ROLES,
        )
        return tuple(builder.phases)
    scheduler = builder.add(
        "prefix.causal_algorithm_1",
        V4PhaseKind.SCHEDULER,
        tuple(confidence_ready),
        batch * block,
        _SCHEDULER_ROLES,
    )
    builder.add(
        "proposal.finalize_audit",
        V4PhaseKind.SCHEDULER,
        (prior_sample, scheduler),
        1,
        _SCHEDULER_ROLES,
    )
    return tuple(builder.phases)
