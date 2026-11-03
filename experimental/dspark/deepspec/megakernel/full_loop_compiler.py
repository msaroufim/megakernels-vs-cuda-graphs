from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from deepspec.megakernel.full_loop_schedule import (
    FullLoopSpec,
    build_full_loop_macro_program,
    build_full_loop_program,
    full_loop_macro_name,
)
from deepspec.megakernel.full_loop_weights import target_layer_compress_ratio


class FullLoopExecutor(str, Enum):
    """Physical executor selected for one compiler-emitted outer band."""

    INPUT_CONTROLLER = "input_controller"
    TARGET_BOOTSTRAP = "target_bootstrap"
    TARGET_LAYER_GRAPH = "target_layer_graph"
    TARGET_HEAD_GRAPH = "target_head_graph"
    TARGET_INJECT_GRAPH = "target_inject_graph"
    LOOP_STATE_UPDATE = "loop_state_update"
    GRAFT_INPUT_GRAPH = "graft_input_graph"
    GRAFT_GRAPH = "graft_graph"
    TAIL_RELAUNCH = "tail_relaunch"


class GraftExecutor(str, Enum):
    """Interchangeable proposal executors behind the full-loop graft ABI."""

    SGLANG_CUDA_GRAPH = "sglang_cuda_graph"
    PROPOSAL_MEGAKERNEL = "proposal_megakernel"


class TargetLayerVariant(str, Enum):
    """AOT numerical graph variants required by the frozen target model."""

    HASH_DENSE = "hash_dense"
    HASH_COMPRESSED_R4 = "hash_compressed_r4"
    SCORE_COMPRESSED_R4 = "score_compressed_r4"
    SCORE_COMPRESSED_R128 = "score_compressed_r128"


@dataclass(frozen=True)
class GB300TargetLowering:
    """Physical target-layer policy selected by the standalone compiler.

    The semantic schedule says *what* a target block computes.  This record
    says *how* its measured sm_103 implementation is composed.  Numerical
    kernels remain separately compiled leaves, while stream dependencies and
    fixed launch choices live in one inspectable compiler value instead of
    being scattered through the runtime builder.
    """

    variant: TargetLayerVariant
    attention: str = "flashmla"
    qnorm: str = "flashinfer"
    q_lora_norm_quant: str = "sm103_fused"
    mhc: str = "deepgemm_tilelang"
    router: str = "sglang_fused_gate"
    routed_expert: str = "flashinfer_mxfp4"
    shared_expert: str = "deepgemm_fp8"
    wo_a: str = "deepgemm_fp8"
    moe_hc_post: str = "sm103_fused"
    compressor: str = "sglang_c4"
    compressed_cache_store: str = "sglang_fused"
    mhc_pre_splits: int = 64
    fused_mhc_splits: int = 8
    fused_mhc_tile_mix_outputs: int = 2
    moe_max_num_tokens: int = 8
    # Captured leaf kernels may use PDL internally. Parent child-graph edges
    # remain full dependencies because CUDA programmatic edge data connects
    # kernel nodes, not opaque child-graph nodes.
    moe_enable_pdl: bool | None = True
    overlap_query_kv: bool = True
    overlap_target_cache_pack: bool = True
    overlap_routed_shared: bool = True
    keep_wo_a_fp8: bool = True
    fuse_moe_finalize: bool = True


@dataclass(frozen=True)
class GB300TargetHeadLowering:
    """AOT physical choices for target verification and device commit."""

    hc_head: str = "sm103_fused_hc_rmsnorm"
    norm: str = "sm103_fused_hc_rmsnorm"
    lm_head: str = "torch_bf16_mm_kn"
    lm_head_layout: str = "kn_contiguous"
    greedy_argmax: str = "sglang_split_topk1"
    accept_commit: str = "sm103_fused"
    autotune: bool = False


@dataclass(frozen=True)
class GB300GraftLowering:
    """AOT physical choices for the inference-only graft executor."""

    executor: str = "sglang_cuda_graph"
    attention: str = "flashinfer"
    routed_expert: str = "flashinfer_mxfp4"
    target_inject: str = "dspark_swa_kv"
    target_tap_reduce: str = "sm103_mean_concat"
    target_inject_layout: str = "sm103_commit_mask"
    input_prepare: str = "sm103_fill_bonus"
    greedy_argmax: str = "sglang_split_topk1"
    hc_head: str = "sm103_fused_hc_rmsnorm"
    lm_head_layout: str = "kn_contiguous"
    direct_proposal_store: bool = True
    share_embedding: bool = True
    share_lm_head: bool = True
    autotune: bool = False


@dataclass(frozen=True)
class GB300LoopLowering:
    """AOT policy for composing repeated decode rounds into one graph launch."""

    state_update: str = "sm103_accept_to_compressor_plans"
    compressor_plan_abi: str = "sglang_legacy_c4_c128"
    metadata_scope: str = "shared_across_target_layers"
    relaunch: str = "aot_parent_unroll"
    autotune: bool = False


@dataclass(frozen=True)
class CompiledFullLoopBand:
    """A small outer opcode whose numerical implementation stays specialized.

    A graph executor is a compiler boundary, not an invitation to inline every
    implementation into one CUDA symbol.  That distinction lets attention,
    routed MoE, and the graft retain independent launch geometry and register
    allocation while still participating in one device-resident program.
    """

    band_id: int
    name: str
    executor: FullLoopExecutor
    dependency: int | None
    work_units: int
    semantic_phase_ids: tuple[int, ...]
    layer: int = -1
    compress_ratio: int = -1
    router: str = "none"
    target_variant: TargetLayerVariant | None = None
    target_lowering: GB300TargetLowering | None = None
    target_head_lowering: GB300TargetHeadLowering | None = None
    graft_lowering: GB300GraftLowering | None = None
    loop_lowering: GB300LoopLowering | None = None
    captures_tap: bool = False


@dataclass(frozen=True)
class CompiledFullLoopProgram:
    """Immutable compiler output consumed by the GB300 runtime builder."""

    bands: tuple[CompiledFullLoopBand, ...]
    semantic_phases: int
    semantic_work_units: int
    workers: int
    threads: int
    launch_backend: str = "aot_composed_cuda_graph"

    @property
    def target_bands(self) -> tuple[CompiledFullLoopBand, ...]:
        return tuple(
            band for band in self.bands if band.executor is FullLoopExecutor.TARGET_LAYER_GRAPH
        )


def _executor_for(name: str) -> FullLoopExecutor:
    if name == "loop.input_ready":
        return FullLoopExecutor.INPUT_CONTROLLER
    if name == "target.bootstrap":
        return FullLoopExecutor.TARGET_BOOTSTRAP
    if name.startswith("target.layer_") and name.endswith(".execute"):
        return FullLoopExecutor.TARGET_LAYER_GRAPH
    if name == "target.head_verify_accept":
        return FullLoopExecutor.TARGET_HEAD_GRAPH
    if name == "graft.target_inject":
        return FullLoopExecutor.TARGET_INJECT_GRAPH
    if name == "loop.advance_accepted_state":
        return FullLoopExecutor.LOOP_STATE_UPDATE
    if name == "graft.input_prepare":
        return FullLoopExecutor.GRAFT_INPUT_GRAPH
    if name == "graft.execute":
        return FullLoopExecutor.GRAFT_GRAPH
    if name == "loop.commit_and_continue":
        return FullLoopExecutor.TAIL_RELAUNCH
    raise ValueError(f"macro band has no full-loop executor: {name}")


def _target_variant(*, layer: int, compress_ratio: int, router: str) -> TargetLayerVariant:
    key = (compress_ratio, router)
    variants = {
        (0, "hash"): TargetLayerVariant.HASH_DENSE,
        (4, "hash"): TargetLayerVariant.HASH_COMPRESSED_R4,
        (4, "score"): TargetLayerVariant.SCORE_COMPRESSED_R4,
        (128, "score"): TargetLayerVariant.SCORE_COMPRESSED_R128,
    }
    try:
        return variants[key]
    except KeyError as error:
        raise ValueError(
            f"target layer {layer} has no AOT graph variant for "
            f"compression={compress_ratio}, router={router}"
        ) from error


def compile_gb300_target_lowering(
    *, layer: int, variant: TargetLayerVariant
) -> GB300TargetLowering:
    """Select the profiler-retained, no-autotune lowering for one block."""

    if not 0 <= layer < 43:
        raise ValueError("target layer must be in [0, 43)")
    router = "hash_fused" if layer < 3 else "sglang_fused_gate"
    compressor = {
        TargetLayerVariant.HASH_DENSE: "none",
        TargetLayerVariant.HASH_COMPRESSED_R4: "sglang_c4",
        TargetLayerVariant.SCORE_COMPRESSED_R4: "sglang_c4",
        TargetLayerVariant.SCORE_COMPRESSED_R128: "sglang_c128",
    }[variant]
    return GB300TargetLowering(
        variant=variant,
        router=router,
        compressor=compressor,
    )


def compile_gb300_graft_lowering(
    executor: GraftExecutor | str = GraftExecutor.SGLANG_CUDA_GRAPH,
) -> GB300GraftLowering:
    """Select the profiled graft graph without runtime tuning."""

    executor = GraftExecutor(executor)
    if executor is GraftExecutor.PROPOSAL_MEGAKERNEL:
        return GB300GraftLowering(
            executor=executor.value,
            attention="dspark_persistent",
            routed_expert="dspark_persistent_fp4",
            greedy_argmax="dspark_greedy_tail",
        )
    return GB300GraftLowering(executor=executor.value)


def compile_gb300_target_head_lowering() -> GB300TargetHeadLowering:
    """Select the profiled target-head graph without runtime tuning."""

    return GB300TargetHeadLowering()


def compile_gb300_loop_lowering() -> GB300LoopLowering:
    """Select the fixed controller ABI used by every persistent iteration."""

    return GB300LoopLowering()


def compile_full_loop_program(
    spec: FullLoopSpec | None = None,
    *,
    graft_executor: GraftExecutor | str = GraftExecutor.SGLANG_CUDA_GRAPH,
) -> CompiledFullLoopProgram:
    """Lower the semantic DAG into one AOT-composed CUDA Graph program.

    Python remains the source of truth for model variants, dependencies, and
    checkpoint routing. CUDA receives a linear band program and immutable
    per-layer metadata; the numerical child graphs remain separately compiled.
    """

    spec = spec or FullLoopSpec()
    spec.require_supported()
    semantic = build_full_loop_program(spec)
    macros = build_full_loop_macro_program(spec)
    semantic_members: dict[str, list[int]] = {macro.name: [] for macro in macros}
    for phase in semantic:
        semantic_members[full_loop_macro_name(phase)].append(phase.phase_id)
    bands = []
    for index, macro in enumerate(macros):
        executor = _executor_for(macro.name)
        layer = macro.layer if executor is FullLoopExecutor.TARGET_LAYER_GRAPH else -1
        ratio = target_layer_compress_ratio(layer) if layer >= 0 else -1
        router = "hash" if 0 <= layer <= 2 else ("score" if layer >= 3 else "none")
        target_variant = (
            _target_variant(layer=layer, compress_ratio=ratio, router=router)
            if layer >= 0
            else None
        )
        target_lowering = (
            compile_gb300_target_lowering(layer=layer, variant=target_variant)
            if target_variant is not None
            else None
        )
        bands.append(
            CompiledFullLoopBand(
                band_id=index,
                name=macro.name,
                executor=executor,
                dependency=None if index == 0 else index - 1,
                work_units=macro.work_units,
                semantic_phase_ids=tuple(semantic_members[macro.name]),
                layer=layer,
                compress_ratio=ratio,
                router=router,
                target_variant=target_variant,
                target_lowering=target_lowering,
                target_head_lowering=(
                    compile_gb300_target_head_lowering()
                    if executor is FullLoopExecutor.TARGET_HEAD_GRAPH
                    else None
                ),
                graft_lowering=(
                    compile_gb300_graft_lowering(graft_executor)
                    if executor
                    in {
                        FullLoopExecutor.TARGET_INJECT_GRAPH,
                        FullLoopExecutor.GRAFT_INPUT_GRAPH,
                        FullLoopExecutor.GRAFT_GRAPH,
                    }
                    else None
                ),
                loop_lowering=(
                    compile_gb300_loop_lowering()
                    if executor
                    in {
                        FullLoopExecutor.LOOP_STATE_UPDATE,
                        FullLoopExecutor.TAIL_RELAUNCH,
                    }
                    else None
                ),
                captures_tap=layer in spec.target_tap_layers,
            )
        )
    lowered_phase_ids = [phase_id for band in bands for phase_id in band.semantic_phase_ids]
    if lowered_phase_ids != list(range(len(semantic))):
        raise ValueError("full-loop fusion did not preserve semantic phase order")
    return CompiledFullLoopProgram(
        bands=tuple(bands),
        semantic_phases=len(semantic),
        semantic_work_units=sum(phase.work_units for phase in semantic),
        workers=spec.workers,
        threads=spec.threads,
    )
