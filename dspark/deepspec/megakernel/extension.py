from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

TRACE_COLUMNS = 12
FULL_TRACE_COLUMNS = 3
FULL_TRACE_PHASE_CAPACITY = 160

V4_EXECUTE_MAIN = 1 << 0
V4_EXECUTE_EMBEDDING = 1 << 1
V4_EXECUTE_MAIN_KV = 1 << 2
V4_EXECUTE_ATTN_HC = 1 << 3
V4_EXECUTE_ATTN_PROJECTIONS = 1 << 4
V4_EXECUTE_SPARSE_ATTENTION = 1 << 5
V4_EXECUTE_ATTENTION_OUTPUT = 1 << 6
V4_EXECUTE_FFN_ROUTER = 1 << 7
V4_EXECUTE_EXPERTS = 1 << 8
V4_EXECUTE_HEAD = 1 << 9
V4_EXECUTE_TAIL = 1 << 10
V4_EXECUTE_ALL = (1 << 11) - 1


def _cutlass_include_path() -> Path:
    spec = importlib.util.find_spec("cutlass_library")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("nvidia-cutlass==4.2.0.0 is required to build the megakernel")
    include_path = Path(next(iter(spec.submodule_search_locations))) / "source" / "include"
    if not (include_path / "cute" / "tensor.hpp").is_file():
        raise RuntimeError(f"CUTLASS C++ headers not found under {include_path}")
    return include_path


@functools.lru_cache(maxsize=2)
def load_dspark_tail_extension(*, instrumented: bool = False):
    source_dir = Path(__file__).resolve().parent / "csrc"
    name = "deepspec_dspark_tail_trace" if instrumented else "deepspec_dspark_tail"
    cuda_flags = ["-O3", "-lineinfo", "--std=c++17"]
    if instrumented:
        cuda_flags.append("-DDSPARK_ENABLE_DEVICE_TRACE=1")
    return load(
        name=name,
        sources=[
            str(source_dir / "dspark_tail.cpp"),
            str(source_dir / "dspark_tail_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[str(_cutlass_include_path())],
        with_cuda=True,
        verbose=False,
    )


@functools.lru_cache(maxsize=2)
def load_dspark_full_extension(*, instrumented: bool = False):
    source_dir = Path(__file__).resolve().parent / "csrc"
    name = "deepspec_dspark_full_trace" if instrumented else "deepspec_dspark_full"
    cuda_flags = ["-O3", "-lineinfo", "--std=c++17"]
    if instrumented:
        cuda_flags.append("-DDSPARK_ENABLE_DEVICE_TRACE=1")
    return load(
        name=name,
        sources=[
            str(source_dir / "dspark_full.cpp"),
            str(source_dir / "dspark_full_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=cuda_flags,
        extra_include_paths=[str(_cutlass_include_path())],
        with_cuda=True,
        verbose=False,
    )


def _confidence_head_prefix_enabled(**kwargs) -> bool:
    """Default only for the certified full proposal; 0 is the diagnostic opt-out."""
    requested = os.environ.get("DSPARK_CONFIDENCE_HEAD_PREFIX")
    if requested not in (None, "0", "1"):
        raise ValueError("DSPARK_CONFIDENCE_HEAD_PREFIX must be 0 or 1")
    required = {
        "DSPARK_V4_STATIC_QUEUES": "1",
        "DSPARK_V4_ROUTED_DYNAMIC_CLAIMS": "1",
        "DSPARK_V4_SHARED_STAGE_CONTEXT": "1",
        "DSPARK_SHARED_CONTINUE": "1",
        "DSPARK_ROUTED_W2_WORKERS": "128",
        "DSPARK_LM_BULK_STAGING": "1",
        "DSPARK_LM_STREAMED_K": "256",
    }
    # Keep the CU's rejected configurations and diagnostic-only source variants
    # out of the default. Direct compiler-flag injection must explicitly opt out.
    excluded = (
        "DSPARK_V4_STATIC_TP2_TAIL",
        "DSPARK_V4_TP2_ABLATE",
        "DSPARK_V4_DIRECT_DEPENDENCIES",
        "DSPARK_V4_COMPLETION_RELEASE",
        "DSPARK_V4_QUEUE_LOOKAHEAD",
        "DSPARK_ROUTED_GROUP_READY",
        "DSPARK_MARKOV_GATHER_CONTINUE",
        "DSPARK_V4_FINE_GRAINED_OVERLAP",
        "DSPARK_V4_STATIC_GAPS",
        "DSPARK_HEAD_SPLIT",
        "DSPARK_HEAD_PREFETCH",
        "DSPARK_V4_PERSISTENT_TMEM",
        "DSPARK_PHASE_OUTLINE",
        "DSPARK_SCHEDULER_HANDOFF",
        "DSPARK_FRONT_EMBED_FIRST",
        "DSPARK_ROUTED_INTERLEAVE",
        "DSPARK_SHARED_SPLIT_K",
        "DSPARK_V4_ENABLE_DEVICE_TRACE",
        "DSPARK_ENABLE_DEVICE_TRACE",
        "DSPARK_V4_PHASE_TIMESTAMPS",
        "DSPARK_V4_HOST_PROGRESS",
        "DSPARK_V4_BOUNDARY_TIMESTAMPS",
        "DSPARK_HEAD_CLOCK",
        "DSPARK_ATTN_CLOCK",
        "DSPARK_HEAD_CLOCK_PROBE",
        "DSPARK_ATTN_CLOCK_PROBE",
    )
    eligible = (
        kwargs.get("relaxed_dag", False)
        and kwargs.get("greedy_tail", False)
        and kwargs.get("batch_size", 1) == 1
        and kwargs.get("full_loop_device_epoch", False)
        and kwargs.get("compiled_execution_mask") == V4_EXECUTE_ALL
        and kwargs.get("compiled_draft_layer_mask") == 7
        and not kwargs.get("instrumented", False)
        and not kwargs.get("static_tp2_tail", False)
        and not kwargs.get("tp2_ablate", False)
        and all(os.environ.get(name) == value for name, value in required.items())
        and not any(os.environ.get(name) not in (None, "", "0") for name in excluded)
    )
    if requested == "1" and not eligible:
        raise ValueError(
            "confidence prefix requires the certified full-mask batch-one greedy configuration"
        )
    return eligible and requested != "0"


def _cache_v4_scheduler_build(function):
    # Unlike the older environment-only study switches, this guarded default
    # and its diagnostic opt-out must select different in-process cache entries.
    @functools.lru_cache(maxsize=32)
    def cached(prefix_enabled, **kwargs):
        return function(**kwargs)

    @functools.wraps(function)
    def wrapped(**kwargs):
        return cached(_confidence_head_prefix_enabled(**kwargs), **kwargs)

    setattr(wrapped, "cache_clear", cached.cache_clear)
    setattr(wrapped, "cache_info", cached.cache_info)
    return wrapped


@_cache_v4_scheduler_build
def load_dspark_v4_scheduler_extension(
    *,
    instrumented: bool = False,
    relaxed_dag: bool = False,
    greedy_tail: bool = False,
    static_tp2_tail: bool = False,
    batch_size: int = 1,
    tp2_ablate: bool = False,
    full_loop_device_epoch: bool = False,
    compiled_execution_mask: int | None = None,
    compiled_draft_layer_mask: int | None = None,
):
    """Build the GB300 V4 persistent scheduler surface.

    This is intentionally not named `proposal`: numerical phase bodies must
    pass the official oracle before the production entry point is exposed.
    ``relaxed_dag`` selects the fused-tail generated header (relaxed-numerics
    program; acceptance-parity gates, never the frozen oracle contract).
    """

    if static_tp2_tail:
        raise ValueError("the legacy B200 static TP2 tail is not supported by the GB300 path")
    if greedy_tail and not relaxed_dag:
        raise ValueError("greedy_tail requires relaxed_dag=True")
    # greedy_tail + instrumented is legal: the device trace records only
    # (phase_id, worker, role, timestamps), and build_v4_phase_program already
    # takes greedy_tail/batch, so the host-side phase map is derivable. The
    # caller MUST pass the matching program to V4DeviceTrace.from_tensors --
    # a mismatched map raises "unknown V4 phase id" rather than mislabelling.
    if batch_size != 1 and not (relaxed_dag and greedy_tail):
        raise ValueError("batch_size > 1 requires relaxed_dag=True and greedy_tail=True")
    static_queues = os.environ.get("DSPARK_V4_STATIC_QUEUES") == "1"
    phase_outline = os.environ.get("DSPARK_PHASE_OUTLINE")
    if phase_outline is not None and (
        phase_outline not in tuple(str(mask) for mask in range(1, 8))
        or not (relaxed_dag and greedy_tail and batch_size == 1 and static_queues)
    ):
        raise ValueError(
            "phase outlining requires the batch-1 static relaxed greedy program and mask 1..7"
        )
    descriptor_arithmetic = os.environ.get("DSPARK_DESCRIPTOR_ARITHMETIC")
    if descriptor_arithmetic is not None and (
        descriptor_arithmetic not in tuple(str(mask) for mask in range(1, 16))
        or not (relaxed_dag and greedy_tail and batch_size == 1)
    ):
        raise ValueError(
            "descriptor arithmetic requires the batch-1 relaxed greedy program and mask 1..15"
        )
    scheduler_handoff = os.environ.get("DSPARK_SCHEDULER_HANDOFF")
    if scheduler_handoff is not None and (
        scheduler_handoff not in ("1", "2", "3") or not static_queues
    ):
        raise ValueError("scheduler handoff experiments require static queues and mask 1, 2, or 3")
    routed_dynamic_claims = os.environ.get("DSPARK_V4_ROUTED_DYNAMIC_CLAIMS") == "1"
    if routed_dynamic_claims and (
        not static_queues or os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP") == "1"
    ):
        raise ValueError("routed dynamic claims require static queues without fine overlap")
    routed_packed_sfa = os.environ.get("DSPARK_ROUTED_PACKED_SFA")
    if routed_packed_sfa is not None:
        if routed_packed_sfa not in ("1", "2"):
            raise ValueError("routed packed SFA mode must be 1 or 2")
        if not (relaxed_dag and greedy_tail and batch_size == 1):
            raise ValueError("packed routed SFA requires the batch-1 relaxed greedy program")
    routed_bulk_weights = os.environ.get("DSPARK_ROUTED_BULK_WEIGHTS") == "1"
    if routed_bulk_weights and (
        routed_packed_sfa != "2" or os.environ.get("DSPARK_W13_DG_CHUNKS") != "16"
    ):
        raise ValueError("routed bulk weights require packed SFA mode 2 and K512 stages")
    markov_bulk_k = os.environ.get("DSPARK_MARKOV_BULK_K")
    if markov_bulk_k is not None:
        if markov_bulk_k not in ("128", "256"):
            raise ValueError("Markov bulk K must be 128 or 256")
        if not (relaxed_dag and greedy_tail and batch_size == 1):
            raise ValueError("Markov bulk staging requires the batch-1 relaxed greedy program")
        if (
            markov_bulk_k == "256"
            and int(os.environ.get("DSPARK_V4_DYNAMIC_SMEM_BYTES", "139520")) < 153600
        ):
            raise ValueError("Markov K256 bulk staging requires 153600 shared-memory bytes")
    # The bulk128 batch-one program only consumes column zero of the N8
    # accumulator. Keep its direct drain separate in the extension cache.
    markov_direct_drain = markov_bulk_k == "128"
    woa_bulk_staging = os.environ.get("DSPARK_WOA_BULK_STAGING") == "1"
    woa_streamed_k = os.environ.get("DSPARK_WOA_STREAMED_K")
    if woa_streamed_k is not None and not (
        woa_streamed_k in ("128", "256")
        and woa_bulk_staging
        and relaxed_dag
        and greedy_tail
        and batch_size == 1
        and not tp2_ablate
    ):
        raise ValueError("WO_A streamed rings require packed batch-1 relaxed greedy weights")
    qb_block_scaled = os.environ.get("DSPARK_QB_BLOCK_SCALED")
    dense_block_scaled = os.environ.get("DSPARK_DENSE_BLOCK_SCALED")
    if os.environ.get("DSPARK_FP8_PRESERVE_K128") == "1":
        raise ValueError("The retained bulk FP8 path requires packed scale4 arithmetic")
    shared_bulk_staging = os.environ.get("DSPARK_SHARED_BULK_STAGING") == "1"
    if shared_bulk_staging and not (
        relaxed_dag and greedy_tail and batch_size == 1 and not tp2_ablate
    ):
        raise ValueError("shared bulk staging requires the batch-1 relaxed greedy program")
    fp8_scale4 = os.environ.get("DSPARK_FP8_SCALE4") == "1"
    if (qb_block_scaled or dense_block_scaled) and not fp8_scale4:
        raise ValueError("The retained bulk FP8 path requires DSPARK_FP8_SCALE4=1")
    if fp8_scale4 and not (relaxed_dag and greedy_tail and batch_size == 1 and not tp2_ablate):
        raise ValueError("packed FP8 scales require the batch-1 relaxed greedy program")
    shared_native_scale4 = os.environ.get("DSPARK_SHARED_NATIVE_SCALE4") == "1"
    if shared_native_scale4 and not (fp8_scale4 and shared_bulk_staging):
        raise ValueError("native shared FP8 requires packed scales and shared bulk weights")
    hc_prefetch = os.environ.get("DSPARK_HC_PREFETCH") == "1"
    attn_mma8 = os.environ.get("DSPARK_ATTN_MMA8") == "1"
    if attn_mma8 and not (relaxed_dag and greedy_tail and batch_size == 1 and not tp2_ablate):
        raise ValueError("attention MMA8 requires the batch-1 relaxed greedy program")
    router_warp_top6 = os.environ.get("DSPARK_ROUTER_WARP_TOP6")
    if router_warp_top6 is not None and router_warp_top6 not in ("1", "2"):
        raise ValueError("router warp top6 mode must be 1 or 2")
    routed_reuse_tmem = os.environ.get("DSPARK_ROUTED_REUSE_TMEM") == "1"
    if routed_reuse_tmem and not (
        relaxed_dag
        and greedy_tail
        and batch_size == 1
        and static_queues
        and routed_dynamic_claims
        and os.environ.get("DSPARK_V4_PERSISTENT_TMEM") != "1"
        and os.environ.get("DSPARK_ROUTED_PACKED_SFA") == "2"
    ):
        raise ValueError(
            "routed TMEM reuse requires batch-1 routed claims without global TMEM reuse"
        )
    routed_group_masks = os.environ.get("DSPARK_ROUTED_GROUP_MASKS") == "1"
    if routed_group_masks and not (
        relaxed_dag
        and greedy_tail
        and batch_size == 1
        and not tp2_ablate
        and os.environ.get("DSPARK_ROUTED_PACKED_SFA") == "2"
    ):
        raise ValueError("routed group masks require batch-1 relaxed greedy packed SFA2")
    routed_sf_ring = os.environ.get("DSPARK_ROUTED_SF_RING")
    if routed_sf_ring is not None and (
        routed_sf_ring not in ("1", "2")
        or not (relaxed_dag and greedy_tail and batch_size == 1)
        or os.environ.get("DSPARK_ROUTED_PACKED_SFA") != "2"
        or os.environ.get("DSPARK_W13_DG_CHUNKS") != "16"
        or os.environ.get("DSPARK_W13_DG_TILE_PAIRS") != "1"
        or os.environ.get("DSPARK_W13_DG_STAGES") != "2"
    ):
        raise ValueError("routed SF ring requires batch-1 relaxed greedy with packed SFA2 and K512")
    hc_split = os.environ.get("DSPARK_HC_SPLIT") == "1"
    if hc_split and not (
        relaxed_dag
        and greedy_tail
        and batch_size == 1
        and static_queues
        and os.environ.get("DSPARK_V4_SHARED_FIRST") == "1"
        and os.environ.get("DSPARK_HC_PREFETCH") == "1"
        and os.environ.get("DSPARK_HC_ELEMENT_LANES") == "1"
        and os.environ.get("DSPARK_HC_COMPUTE_OVERLAP") != "1"
    ):
        raise ValueError(
            "HC split requires batch-1 shared-first relaxed greedy "
            "with HC prefetch and element lanes"
        )
    woa_split_k = os.environ.get("DSPARK_WOA_SPLIT_K")
    if woa_split_k is not None and not (
        woa_split_k == "2"
        and woa_streamed_k == "256"
        and hc_split
        and not os.environ.get("DSPARK_V4_STATIC_GAPS")
        and not os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP")
    ):
        raise ValueError("WO_A split-K2 requires K256 rings and the HC split schedule")
    swiglu_warp = os.environ.get("DSPARK_SWIGLU_WARP_QUANT") == "1"
    if swiglu_warp and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("warp SwiGLU quantization requires batch-1 relaxed greedy execution")
    compact_tiles = os.environ.get("DSPARK_ROUTED_COMPACT_TILES") == "1"
    compact_workers = os.environ.get("DSPARK_ROUTED_COMPACT_WORKERS")
    if compact_workers is not None and (
        not compact_tiles or compact_workers not in {str(n) for n in range(1, 153)}
    ):
        raise ValueError("compact worker count requires compact tiles and must be 1..152")
    # Compact streams are indexed by ticket, independent of completion credits.
    # 960 / 8 gives exactly 120 tickets, covering every compact stream without
    # the 120 additional empty tickets issued by the historical chunk of four.
    compact_claim_chunk = os.environ.get("DSPARK_ROUTED_CLAIM_CHUNK")
    compact_claims_valid = compact_claim_chunk in (None, "4") or (
        compact_claim_chunk == "8" and compact_workers == "120"
    )
    if compact_tiles and not (
        relaxed_dag
        and greedy_tail
        and batch_size == 1
        and routed_dynamic_claims
        and os.environ.get("DSPARK_ROUTED_GROUP_MASKS") == "1"
        and os.environ.get("DSPARK_W13_DG_TILE_PAIRS") == "1"
        and compact_claims_valid
        and not tp2_ablate
    ):
        raise ValueError(
            "compact routed claims need four-item tickets, or eight for exactly 120 streams"
        )
    w13_fused_swiglu = os.environ.get("DSPARK_ROUTED_FUSED_SWIGLU") == "1"
    if w13_fused_swiglu and not (
        compact_tiles
        and hc_split
        and swiglu_warp
        and os.environ.get("DSPARK_ROUTED_COMPACT_WORKERS") == "120"
        and not os.environ.get("DSPARK_V4_STATIC_GAPS")
        and not os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP")
    ):
        raise ValueError("W13 SwiGLU fusion requires compact120 and HC split without overlap")
    routed_interleave = os.environ.get("DSPARK_ROUTED_INTERLEAVE") == "1"
    if routed_interleave and not (
        compact_tiles
        and compact_workers == "128"
        and hc_split
        and not w13_fused_swiglu
        and woa_split_k == "2"
        and os.environ.get("DSPARK_ROUTED_PACKED_SFA") == "2"
        and os.environ.get("DSPARK_ROUTED_BULK_WEIGHTS") != "1"
        and os.environ.get("DSPARK_W13_DG_CHUNKS") == "16"
        and not os.environ.get("DSPARK_V4_STATIC_GAPS")
        and not os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP")
    ):
        raise ValueError("interleaved routed W13 requires compact128, SFA2 and HC split2")
    front_embed_first = os.environ.get("DSPARK_FRONT_EMBED_FIRST") == "1"
    if front_embed_first and not (
        w13_fused_swiglu
        and woa_split_k == "2"
        and not routed_interleave
        and os.environ.get("DSPARK_ROUTED_GROUP_READY") != "1"
    ):
        raise ValueError("front embedding priority requires the fused W13/split2 program")
    direct_value = os.environ.get("DSPARK_V4_DIRECT_DEPENDENCIES")
    if direct_value not in (None, "1", "2"):
        raise ValueError("direct dependency mode must be 1 or 2")
    direct_dependencies = int(direct_value or 0)
    lookahead_value = os.environ.get("DSPARK_V4_QUEUE_LOOKAHEAD")
    if lookahead_value is not None and lookahead_value not in ("8", "32"):
        raise ValueError("queue lookahead must be 8 or 32")
    queue_lookahead = int(lookahead_value or 0)
    if queue_lookahead and not (
        direct_dependencies == 2
        and compiled_execution_mask == V4_EXECUTE_ALL
        and compiled_draft_layer_mask == 7
        and os.environ.get("DSPARK_ROUTED_CLAIM_CHUNK") in (None, "4")
        and os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP") != "1"
    ):
        raise ValueError("queue lookahead requires direct counters and all compiled phases")
    completion_release = os.environ.get("DSPARK_V4_COMPLETION_RELEASE") == "1"
    if direct_dependencies and not (hc_split and routed_dynamic_claims and not tp2_ablate):
        raise ValueError(
            "direct dependencies require single-GPU HC split with routed dynamic claims"
        )
    if completion_release and not (relaxed_dag and greedy_tail and static_queues):
        raise ValueError("ordered completion requires relaxed greedy static queues")
    gap_value = os.environ.get("DSPARK_V4_STATIC_GAPS")
    if gap_value not in (None, "1", "2"):
        raise ValueError("static gap scheduling mode must be 1 or 2")
    gap_schedule = int(gap_value or 0)
    if gap_schedule and not hc_split:
        raise ValueError("static gap scheduling requires HC split")
    routed_group_ready = os.environ.get("DSPARK_ROUTED_GROUP_READY") == "1"
    if routed_group_ready and not (
        w13_fused_swiglu
        and woa_split_k == "2"
        and compact_claim_chunk in (None, "4")
        and not direct_dependencies
        and compiled_execution_mask == 2047
        and compiled_draft_layer_mask == 7
        and os.environ.get("DSPARK_V4_SHARED_STAGE_CONTEXT") == "1"
    ):
        raise ValueError("group readiness requires the full compiled fused W13/split2 program")
    shared_continue = os.environ.get("DSPARK_SHARED_CONTINUE") == "1"
    if shared_continue and not (
        w13_fused_swiglu
        and woa_split_k == "2"
        and compact_claim_chunk == "8"
        and not routed_interleave
        and not routed_group_ready
        and not front_embed_first
        and not os.environ.get("DSPARK_SHARED_SPLIT_K")
    ):
        raise ValueError("shared continuation requires unsplit shared W13 and compact120 claim8")
    routed_weight_warp = os.environ.get("DSPARK_ROUTED_WEIGHT_WARP")
    if routed_weight_warp is not None and not (
        routed_weight_warp == "1"
        and w13_fused_swiglu
        and shared_continue
        and static_queues
        and routed_dynamic_claims
        and compact_tiles
        and compact_workers == "120"
        and compact_claim_chunk == "8"
        and os.environ.get("DSPARK_ROUTED_PACKED_SFA") == "2"
        and os.environ.get("DSPARK_W13_DG_CHUNKS") == "16"
        and os.environ.get("DSPARK_W13_DG_STAGES") == "2"
        and os.environ.get("DSPARK_W13_DG_TILE_PAIRS") == "1"
        and os.environ.get("DSPARK_V4_DYNAMIC_SMEM_BYTES") == "153600"
        and os.environ.get("DSPARK_W13_DG_CREW", "0") == "0"
        and os.environ.get("DSPARK_ROUTED_SF_RING", "0") == "0"
        and os.environ.get("DSPARK_ROUTED_BULK_WEIGHTS", "0") == "0"
        and os.environ.get("DSPARK_ROUTED_W2_WORKERS") in (None, "128")
        and os.environ.get("DSPARK_SHARED_SPLIT_K") is None
        and os.environ.get("DSPARK_HEAD_SPLIT") != "1"
        and relaxed_dag
        and greedy_tail
        and batch_size == 1
        and not tp2_ablate
        and not direct_dependencies
        and not queue_lookahead
    ):
        raise ValueError(
            "dedicated W13 weight warp requires fused120 K512/stages2 shared continuation"
        )
    launch_smem = os.environ.get("DSPARK_V4_LAUNCH_SMEM_BYTES")
    w13_stage3 = os.environ.get("DSPARK_ROUTED_W13_STAGE3")
    if launch_smem is not None and not (
        launch_smem == "227328"
        and routed_weight_warp == "1"
        and os.environ.get("DSPARK_V4_DYNAMIC_SMEM_BYTES") == "153600"
    ):
        raise ValueError(
            "launch-only shared reservation requires current weight-warp body and 227328"
        )
    if w13_stage3 is not None and not (w13_stage3 == "1" and launch_smem == "227328"):
        raise ValueError("W13-only third stage requires launch-only 227328 reservation")
    routed_w2_workers = os.environ.get("DSPARK_ROUTED_W2_WORKERS")
    if routed_w2_workers is not None and not (
        routed_w2_workers == "128"
        and routed_weight_warp == "1"
        and w13_stage3 == "1"
        and launch_smem == "227328"
        and compiled_execution_mask == 2047
        and compiled_draft_layer_mask == 7
        and not gap_schedule
        and not os.environ.get("DSPARK_ROUTED_W2_CREDITS")
    ):
        raise ValueError(
            "W2 width128 requires current full stage3/weight-warp single-GPU shared continuation"
        )
    hc_compute_overlap = os.environ.get("DSPARK_HC_COMPUTE_OVERLAP") == "1"
    if hc_compute_overlap and not (
        relaxed_dag
        and greedy_tail
        and batch_size == 1
        and os.environ.get("DSPARK_HC_PREFETCH") == "1"
        and os.environ.get("DSPARK_HC_ELEMENT_LANES") == "1"
    ):
        raise ValueError(
            "HC compute overlap requires batch-1 relaxed greedy HC prefetch and element lanes"
        )
    head_prefetch = os.environ.get("DSPARK_HEAD_PREFETCH") == "1"
    if head_prefetch and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("head prefetch requires the batch-1 relaxed greedy program")

    bf16_quant_bits = os.environ.get("DSPARK_BF16_QUANT_BITS") == "1"
    main_kv_warp_quant = os.environ.get("DSPARK_MAIN_KV_WARP_QUANT") == "1"
    if main_kv_warp_quant and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("main KV warp quantization requires the batch-1 relaxed greedy program")
    if bf16_quant_bits and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("BF16 scale-bit quantization requires the batch-1 relaxed greedy program")
    if hc_prefetch and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("HC prefetch requires the batch-1 relaxed greedy program")
    if dense_block_scaled is not None:
        if dense_block_scaled not in ("2", "4"):
            raise ValueError("dense block scaling requires a two- or four-stage ring")
        if not (relaxed_dag and greedy_tail and batch_size == 1):
            raise ValueError("dense block scaling requires the batch-1 relaxed greedy program")
    if qb_block_scaled is not None:
        if qb_block_scaled not in ("2", "4"):
            raise ValueError("QB block scaling requires a two- or four-stage ring")
        if not (relaxed_dag and greedy_tail and batch_size == 1):
            raise ValueError("QB block scaling requires the batch-1 relaxed greedy program")
    if woa_bulk_staging and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("WO_A bulk staging requires the batch-1 relaxed greedy program")
    warp_tree_reductions = os.environ.get("DSPARK_WARP_TREE_REDUCTIONS") == "1"
    shared_k128_ring = os.environ.get("DSPARK_SHARED_K128_RING") == "1"
    if shared_k128_ring and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("shared K128 staging requires the batch-1 relaxed greedy program")
    shared_first = os.environ.get("DSPARK_V4_SHARED_FIRST") == "1"
    if shared_first and (
        not static_queues
        or batch_size != 1
        or os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP") == "1"
    ):
        raise ValueError(
            "shared-first scheduling requires batch-1 static queues without fine overlap"
        )
    persistent_tmem = os.environ.get("DSPARK_V4_PERSISTENT_TMEM") == "1"
    if persistent_tmem and not static_queues:
        raise ValueError("DSPARK_V4_PERSISTENT_TMEM requires static queues")
    shared_stage_context = os.environ.get("DSPARK_V4_SHARED_STAGE_CONTEXT") == "1"
    if shared_stage_context and not static_queues:
        raise ValueError("DSPARK_V4_SHARED_STAGE_CONTEXT requires static queues")
    hc_element_lanes = os.environ.get("DSPARK_HC_ELEMENT_LANES") == "1"
    if hc_element_lanes and not relaxed_dag:
        raise ValueError("DSPARK_HC_ELEMENT_LANES requires relaxed_dag=True")
    lm_bulk_staging = os.environ.get("DSPARK_LM_BULK_STAGING") == "1"
    lm_streamed_k = os.environ.get("DSPARK_LM_STREAMED_K")
    if lm_streamed_k is not None:
        if lm_streamed_k not in ("128", "256") or not lm_bulk_staging:
            raise ValueError("streamed LM requires bulk staging and K128 or K256")
        if int(os.environ.get("DSPARK_V4_DYNAMIC_SMEM_BYTES", "139520")) < 153600:
            raise ValueError("streamed LM requires 153600 shared-memory bytes")
    if lm_bulk_staging:
        if not (relaxed_dag and greedy_tail and batch_size == 1):
            raise ValueError("DSPARK_LM_BULK_STAGING requires the batch-1 relaxed greedy build")
        if any(
            os.environ.get(flag) == "1" for flag in ("DSPARK_LM_K128_RING", "DSPARK_LM_STAGE5_RING")
        ):
            raise ValueError("bulk LM staging cannot be combined with other LM ring experiments")
        if int(os.environ.get("DSPARK_V4_DYNAMIC_SMEM_BYTES", "139520")) < 139520:
            raise ValueError("bulk LM staging requires at least 139520 shared-memory bytes")
    if static_queues and not (relaxed_dag and greedy_tail and batch_size == 1):
        raise ValueError("DSPARK_V4_STATIC_QUEUES requires the batch-1 relaxed greedy GB300 build")
    fine_overlap = os.environ.get("DSPARK_V4_FINE_GRAINED_OVERLAP") == "1"
    if fine_overlap and not static_queues:
        raise ValueError("DSPARK_V4_FINE_GRAINED_OVERLAP requires DSPARK_V4_STATIC_QUEUES=1")
    routed_claim_chunk = os.environ.get("DSPARK_ROUTED_CLAIM_CHUNK")
    if routed_claim_chunk and static_queues:
        if not routed_dynamic_claims:
            raise ValueError(
                "DSPARK_ROUTED_CLAIM_CHUNK requires dynamic queues or routed dynamic claims"
            )
        chunk = int(routed_claim_chunk)
        tile_pairs = int(os.environ.get("DSPARK_W13_DG_TILE_PAIRS", "2"))
        if chunk < 1 or chunk > 32 or 32 % chunk or chunk % tile_pairs:
            raise ValueError("static routed chunks must divide 32 and contain whole fused tiles")
    if routed_claim_chunk and not relaxed_dag:
        raise ValueError("DSPARK_ROUTED_CLAIM_CHUNK requires relaxed_dag=True")
    lm_k128_ring = os.environ.get("DSPARK_LM_K128_RING") == "1"
    if lm_k128_ring and not relaxed_dag:
        raise ValueError("DSPARK_LM_K128_RING requires relaxed_dag=True")
    lm_stage5_ring = os.environ.get("DSPARK_LM_STAGE5_RING") == "1"
    if lm_stage5_ring and not relaxed_dag:
        raise ValueError("DSPARK_LM_STAGE5_RING requires relaxed_dag=True")
    proj_k128_ring = os.environ.get("DSPARK_PROJ_K128_RING") == "1"
    if proj_k128_ring and not relaxed_dag:
        raise ValueError("DSPARK_PROJ_K128_RING requires relaxed_dag=True")
    source_dir = Path(__file__).resolve().parent / "csrc"
    if full_loop_device_epoch:
        if not (relaxed_dag and batch_size == 1):
            raise ValueError("full_loop_device_epoch requires the batch-1 relaxed scheduler")
    if (compiled_execution_mask is None) != (compiled_draft_layer_mask is None):
        raise ValueError("compiled execution and draft-layer masks must be provided together")
    if compiled_execution_mask is not None:
        assert compiled_draft_layer_mask is not None
        if not static_queues:
            raise ValueError("compiled phase masks require DSPARK_V4_STATIC_QUEUES=1")
        if not 0 <= compiled_execution_mask <= V4_EXECUTE_ALL:
            raise ValueError("compiled execution mask contains unsupported phase bits")
        if not 1 <= int(compiled_draft_layer_mask) <= 0x7:
            raise ValueError("compiled draft-layer mask must select one to three layers")
    confidence_head_prefix = _confidence_head_prefix_enabled(
        instrumented=instrumented,
        relaxed_dag=relaxed_dag,
        greedy_tail=greedy_tail,
        static_tp2_tail=static_tp2_tail,
        batch_size=batch_size,
        tp2_ablate=tp2_ablate,
        full_loop_device_epoch=full_loop_device_epoch,
        compiled_execution_mask=compiled_execution_mask,
        compiled_draft_layer_mask=compiled_draft_layer_mask,
    )
    name = (
        "deepspec_dspark_v4_scheduler"
        + ("_trace" if instrumented else "")
        + ("_relaxed" if relaxed_dag else "")
        + ("_greedy" if greedy_tail else "")
        + (f"_b{batch_size}" if batch_size != 1 else "")
        + ("_tp2ablate" if tp2_ablate else "")
        + ("_staticqueues" if static_queues else "")
        + ("_routeclaim" if routed_dynamic_claims else "")
        + ("_shfirst" if shared_first else "")
        + ("_sharedbulk" if shared_bulk_staging else "")
        + ("_sf4" if fp8_scale4 else "")
        + ("_shsf4" if shared_native_scale4 else "")
        + ("_shcontinue" if shared_continue else "")
        + ("_w2width128" if routed_w2_workers else "")
        + ("_warptree" if warp_tree_reductions else "")
        + ("_woabulk" if woa_bulk_staging else "")
        + (f"_woastream{woa_streamed_k}" if woa_streamed_k else "")
        + (f"_qbmxf8s{qb_block_scaled}" if qb_block_scaled else "")
        + (f"_densemxf8s{dense_block_scaled}" if dense_block_scaled else "")
        + ("_hcprefetch" if hc_prefetch else "")
        + ("_attnmma8" if attn_mma8 else "")
        + ("_headprefetch" if head_prefetch else "")
        + ("_hcoverlap" if hc_compute_overlap else "")
        + ("_hcsplit" if hc_split else "")
        + ("_w13swiglu" if w13_fused_swiglu else "")
        + ("_groupready" if routed_group_ready else "")
        + (f"_woasplit{woa_split_k}" if woa_split_k else "")
        + (f"_staticgaps{gap_schedule}" if gap_schedule else "")
        + (f"_directdeps{direct_dependencies}" if direct_dependencies else "")
        + (f"_lookahead{queue_lookahead}" if queue_lookahead else "")
        + ("_swigluwarp" if swiglu_warp else "")
        + ("_frontembed" if front_embed_first else "")
        + ("_interleave128" if routed_interleave else "")
        + ("_compacttiles" if compact_tiles else "")
        + (f"w{compact_workers}" if compact_workers else "")
        + ("_w13weightwarp1" if routed_weight_warp else "")
        + ("_completionrelease" if completion_release else "")
        + (f"_sfring{routed_sf_ring}" if routed_sf_ring is not None else "")
        + (f"_routerwarptop6{router_warp_top6}" if router_warp_top6 else "")
        + ("_routetmem" if routed_reuse_tmem else "")
        + ("_groupmasks" if routed_group_masks else "")
        + (f"_outline{phase_outline}" if phase_outline is not None else "")
        + (f"_descarith{descriptor_arithmetic}" if descriptor_arithmetic is not None else "")
        + ("_qbits" if bf16_quant_bits else "")
        + ("_routedbulk" if routed_bulk_weights else "")
        + (f"_handoff{scheduler_handoff}" if scheduler_handoff is not None else "")
        + ("_kvwarp" if main_kv_warp_quant else "")
        + (f"_mkbulk{markov_bulk_k}" if markov_bulk_k is not None else "")
        + ("_mkdrain1" if markov_direct_drain else "")
        + (f"_packedsfa{routed_packed_sfa}" if routed_packed_sfa is not None else "")
        + ("_shk128" if shared_k128_ring else "")
        + ("_ptmem" if persistent_tmem else "")
        + ("_shctx" if shared_stage_context else "")
        + ("_hc16" if hc_element_lanes else "")
        + ("_lmbulk64" if lm_bulk_staging else "")
        + (f"_lmstream{lm_streamed_k}" if lm_streamed_k else "")
        + ("_fineoverlap" if fine_overlap else "")
        + ("_lmk128ring" if lm_k128_ring else "")
        + ("_lmstage5ring" if lm_stage5_ring else "")
        + ("_projk128ring" if proj_k128_ring else "")
        + ("_fullloopepoch" if full_loop_device_epoch else "")
        + ("_confprefix" if confidence_head_prefix else "")
        + (
            f"_xm{compiled_execution_mask:x}_lm{compiled_draft_layer_mask:x}"
            if compiled_execution_mask is not None
            else ""
        )
    )
    cxx_flags = ["-O3", "-std=c++17"]
    cuda_flags = ["-O3", "-lineinfo", "--std=c++17"]
    ldflags: list[str] = []
    if instrumented:
        # The host TU needs the same define: it registers the attribution
        # probe dump only in trace builds.
        cuda_flags.append("-DDSPARK_V4_ENABLE_DEVICE_TRACE=1")
        cxx_flags.append("-DDSPARK_V4_ENABLE_DEVICE_TRACE=1")
    if relaxed_dag:
        cuda_flags.append("-DDSPARK_V4_RELAXED_DAG=1")
        # B2: relaxed builds compile the bitwise-proven packed-subbyte
        # tensor-map weight staging into the routed FP4 W13/W2 bodies. The
        # host-side cuTensorMapEncodeTiled call links against the CUDA
        # driver library (stubs at build time, real libcuda at run time).
        # The contract module never receives this define and stays
        # byte-identical.
        cuda_flags.append("-DDSPARK_W13_TMA_CANDIDATE=1")
        ldflags = ["-lcuda"]
        if CUDA_HOME is not None:
            cuda_stub_dir = Path(CUDA_HOME) / "lib64" / "stubs"
            if (cuda_stub_dir / "libcuda.so").is_file():
                ldflags.insert(0, f"-L{cuda_stub_dir}")
    if greedy_tail:
        cuda_flags.append("-DDSPARK_V4_GREEDY_TAIL=1")
    if full_loop_device_epoch:
        cuda_flags.append("-DDSPARK_V4_FULL_LOOP_DEVICE_EPOCH=1")
    if static_queues:
        cuda_flags.append("-DDSPARK_V4_STATIC_QUEUES=1")
    if routed_dynamic_claims:
        cuda_flags.append("-DDSPARK_V4_ROUTED_DYNAMIC_CLAIMS=1")
    if shared_k128_ring:
        cuda_flags.append("-DDSPARK_SHARED_K128_RING=1")
    if routed_packed_sfa is not None:
        cuda_flags.append(f"-DDSPARK_ROUTED_PACKED_SFA={routed_packed_sfa}")
    if routed_bulk_weights:
        cuda_flags.append("-DDSPARK_ROUTED_BULK_WEIGHTS=1")
    if scheduler_handoff is not None:
        cuda_flags.append(f"-DDSPARK_SCHEDULER_HANDOFF={scheduler_handoff}")
    if markov_bulk_k is not None:
        cuda_flags.append(f"-DDSPARK_MARKOV_BULK_K={markov_bulk_k}")
    if markov_direct_drain:
        cuda_flags.append("-DDSPARK_MARKOV_DIRECT_DRAIN=1")
    if woa_bulk_staging:
        cuda_flags.append("-DDSPARK_WOA_BULK_STAGING=1")
    if woa_streamed_k:
        cuda_flags.append(f"-DDSPARK_WOA_STREAMED_K={woa_streamed_k}")
    if qb_block_scaled:
        cuda_flags.append(f"-DDSPARK_QB_BLOCK_SCALED={qb_block_scaled}")
    if dense_block_scaled:
        cuda_flags.append(f"-DDSPARK_DENSE_BLOCK_SCALED={dense_block_scaled}")
    if fp8_scale4:
        cuda_flags.append("-DDSPARK_FP8_SCALE4=1")
    if shared_native_scale4:
        cuda_flags.append("-DDSPARK_SHARED_NATIVE_SCALE4=1")
    if routed_w2_workers:
        cuda_flags.append("-DDSPARK_ROUTED_W2_WORKERS=128")
    if routed_weight_warp:
        cuda_flags.append("-DDSPARK_ROUTED_WEIGHT_WARP=1")
    if shared_continue:
        cuda_flags.append("-DDSPARK_SHARED_CONTINUE=1")
    if hc_prefetch:
        cuda_flags.append("-DDSPARK_HC_PREFETCH=1")
    if attn_mma8:
        cuda_flags.append("-DDSPARK_ATTN_MMA8=1")
    if router_warp_top6:
        cuda_flags.append(f"-DDSPARK_ROUTER_WARP_TOP6={router_warp_top6}")
    if routed_reuse_tmem:
        cuda_flags.append("-DDSPARK_ROUTED_REUSE_TMEM=1")
    if routed_group_masks:
        cuda_flags.append("-DDSPARK_ROUTED_GROUP_MASKS=1")
    if routed_sf_ring is not None:
        cuda_flags.append(f"-DDSPARK_ROUTED_SF_RING={routed_sf_ring}")
    if swiglu_warp:
        cuda_flags.append("-DDSPARK_SWIGLU_WARP_QUANT=1")
    if compact_tiles:
        cuda_flags.append("-DDSPARK_ROUTED_COMPACT_TILES=1")
        if compact_workers is not None:
            cuda_flags.append(f"-DDSPARK_ROUTED_COMPACT_WORKERS={compact_workers}")
    if front_embed_first:
        cuda_flags.append("-DDSPARK_FRONT_EMBED_FIRST=1")
    if routed_interleave:
        cuda_flags.append("-DDSPARK_ROUTED_INTERLEAVE=1")
    if direct_dependencies:
        cuda_flags.append(f"-DDSPARK_V4_DIRECT_DEPENDENCIES={direct_dependencies}")
    if queue_lookahead:
        cuda_flags.append(f"-DDSPARK_V4_QUEUE_LOOKAHEAD={queue_lookahead}")
    if completion_release or direct_dependencies:
        cuda_flags.append("-DDSPARK_V4_COMPLETION_RELEASE=1")
    if gap_schedule:
        cuda_flags.append(f"-DDSPARK_V4_STATIC_GAPS={gap_schedule}")
    if routed_group_ready:
        cuda_flags.append("-DDSPARK_ROUTED_GROUP_READY=1")
    if w13_fused_swiglu:
        cuda_flags.append("-DDSPARK_ROUTED_FUSED_SWIGLU=1")
    if woa_split_k:
        cuda_flags.append(f"-DDSPARK_WOA_SPLIT_K={woa_split_k}")
    if hc_split:
        cuda_flags.append("-DDSPARK_HC_SPLIT")
    if hc_compute_overlap:
        cuda_flags.append("-DDSPARK_HC_COMPUTE_OVERLAP")
    if head_prefetch:
        cuda_flags.append("-DDSPARK_HEAD_PREFETCH=1")
    if phase_outline is not None:
        cuda_flags.append(f"-DDSPARK_PHASE_OUTLINE={phase_outline}")
    if descriptor_arithmetic is not None:
        cuda_flags.append(f"-DDSPARK_DESCRIPTOR_ARITHMETIC={descriptor_arithmetic}")
    if bf16_quant_bits:
        cuda_flags.append("-DDSPARK_BF16_QUANT_BITS=1")
    if main_kv_warp_quant:
        cuda_flags.append("-DDSPARK_MAIN_KV_WARP_QUANT=1")
    if warp_tree_reductions:
        cuda_flags.append("-DDSPARK_WARP_TREE_REDUCTIONS=1")
    if shared_bulk_staging:
        cuda_flags.append("-DDSPARK_SHARED_BULK_STAGING=1")
    if shared_first:
        cuda_flags.append("-DDSPARK_V4_SHARED_FIRST=1")
    if persistent_tmem:
        cuda_flags.append("-DDSPARK_V4_PERSISTENT_TMEM=1")
    if shared_stage_context:
        cuda_flags.append("-DDSPARK_V4_SHARED_STAGE_CONTEXT=1")
    if hc_element_lanes:
        cuda_flags.append("-DDSPARK_HC_ELEMENT_LANES=1")
    if lm_bulk_staging:
        cuda_flags.append("-DDSPARK_LM_BULK_STAGING=1")
    if lm_streamed_k:
        cuda_flags.append(f"-DDSPARK_LM_STREAMED_K={lm_streamed_k}")
    if compiled_execution_mask is not None:
        cuda_flags.extend(
            [
                f"-DDSPARK_V4_COMPILED_EXECUTION_MASK={compiled_execution_mask}",
                f"-DDSPARK_V4_COMPILED_DRAFT_LAYER_MASK={compiled_draft_layer_mask}",
            ]
        )
    if fine_overlap:
        cuda_flags.append("-DDSPARK_V4_FINE_GRAINED_OVERLAP=1")
    if lm_k128_ring:
        cuda_flags.append("-DDSPARK_LM_K128_RING=1")
    if lm_stage5_ring:
        cuda_flags.append("-DDSPARK_LM_STAGE5_RING=1")
    if proj_k128_ring:
        cuda_flags.append("-DDSPARK_PROJ_K128_RING=1")
    if batch_size != 1:
        # Batched serving specialization (R3). The define selects the batched
        # generated header and widens the tensor-core N mode of the
        # weight-bound bands; batch_size == 1 never sees it, so the batch-1
        # binary is exactly the pre-batch one.
        batch_define = f"-DDSPARK_V4_BATCH={batch_size}"
        cxx_flags.append(batch_define)
        cuda_flags.append(batch_define)
    if tp2_ablate:
        # TP2 band-ownership ablation (timing only, partial numerics). The
        # shipped builds never carry this define.
        ablate_define = "-DDSPARK_V4_TP2_ABLATE=1"
        cxx_flags.append(ablate_define)
        cuda_flags.append(ablate_define)
    # Shared-memory budget / DeepGEMM ring-geometry study (R7/R22). These
    # defines default inside the headers to the shipped literals, so an unset
    # environment reproduces the byte-identical build; setting any one
    # also renames the module so the two variants cannot share a JIT cache
    # entry.
    #
    # R11 adds a RELAXED-BUILD DEFAULT for the routed DeepGEMM bodies: the
    # fused 256-column routed tile (DSPARK_W13_DG_TILE_PAIRS = 2) with the
    # ring depth it leaves room for (DSPARK_W13_DG_STAGES = 3). Measured
    # same-session on one B200, 41 interleaved samples per point, bitwise
    # identical to the shipped body on both bands: routed W13 129.5 -> 102.4
    # us and routed W2 71.6 -> 58.1 us at the production claim chunk, a
    # 1.253x on the pair. The pair fits in the SHIPPED 139,520 B budget
    # (sizeof(DgSharedStorage) 122,880, 16,640 B spare), so nothing else in
    # the kernel pays an L1 carveout for it, and the CONTRACT build never
    # receives either define and stays byte-identical.
    study_defines = {}
    if relaxed_dag:
        study_defines["DSPARK_W13_DG_STAGES"] = 3
        study_defines["DSPARK_W13_DG_TILE_PAIRS"] = 2
        # R13 DELIBERATELY DOES NOT SET DSPARK_W13_DG_CREW. The wide staging
        # crew (mask 1) is 1.0208x on the PHASE BENCH and 1.022x SLOWER ON THE
        # PATH at batch 1 -- see the falsification recorded in
        # dspark_w13_phase.cuh next to kDgCrew. Anything measured only in the
        # phase-bench lane has to be re-measured with
        # `v4_batch_bench.py --crew-configs`, whose route table is the real
        # one, before it can become a default here.
    # R12 router chain geometry. Relaxed-build only, and the environment
    # overrides are what `v4_batch_bench.py --router-sweep` walks; the
    # generated relaxed headers carry the matching chain counts, and the
    # kernel static_asserts the two against each other.
    # R13 adds DSPARK_W13_DG_CREW to the relaxed-only set: it re-assigns which
    # warp of the routed DeepGEMM body stages, multiplies and drains, and the
    # contract build must never see it.
    _ROUTER_DEFINES = (
        "DSPARK_ROUTER_CHAINS",
        "DSPARK_ROUTER_SPLIT",
        "DSPARK_ROUTER_ROWS",
        "DSPARK_ROUTER_UNROLL",
        "DSPARK_W13_DG_CREW",
        "DSPARK_ROUTED_CLAIM_CHUNK",
    )
    for variable in (
        "DSPARK_V4_DYNAMIC_SMEM_BYTES",
        "DSPARK_V4_LAUNCH_SMEM_BYTES",
        "DSPARK_ROUTED_W13_STAGE3",
        "DSPARK_W13_DG_STAGES",
        "DSPARK_W13_DG_TILE_PAIRS",
        "DSPARK_W13_DG_CHUNKS",
        "DSPARK_W13_DG_STAGE_UNROLL",
    ) + _ROUTER_DEFINES:
        if (variable in _ROUTER_DEFINES or variable == "DSPARK_W13_DG_CHUNKS") and not relaxed_dag:
            continue
        value = os.environ.get(variable)
        if value:
            study_defines[variable] = int(value)
    for define in (
        "DSPARK_V4_DYNAMIC_SMEM_BYTES",
        "DSPARK_V4_LAUNCH_SMEM_BYTES",
        "DSPARK_ROUTED_W13_STAGE3",
        "DSPARK_W13_DG_STAGES",
        "DSPARK_W13_DG_TILE_PAIRS",
        "DSPARK_W13_DG_CHUNKS",
        "DSPARK_W13_DG_STAGE_UNROLL",
    ) + _ROUTER_DEFINES:
        if define not in study_defines:
            continue
        number = study_defines[define]
        cuda_flags.append(f"-D{define}={number}")
        # Keep the stage-unroll experiment's tag compact. The static-queue
        # K512/q_b module name is already near CPython's 200-character
        # module-name lookup limit. Existing shorter names remain stable.
        name += (
            f"_dgu{number}"
            if define == "DSPARK_W13_DG_STAGE_UNROLL"
            else f"_{define.lower()}{number}"
        )
    if confidence_head_prefix:
        cuda_flags.append("-DDSPARK_CONFIDENCE_HEAD_PREFIX=1")
    # CPython's shared-library loader truncates the module part of PyInit to
    # 200 characters. Preserve all build choices in a digest and leave room
    # for PyTorch's process-local version suffix when experiments get long.
    if len(name) > 190:
        name = f"{name[:150]}_{hashlib.sha256(name.encode()).hexdigest()[:16]}"
    sources = [
        str(source_dir / "dspark_v4.cpp"),
        str(source_dir / "dspark_v4_kernel.cu"),
    ]
    return load(
        name=name,
        sources=sources,
        extra_cflags=cxx_flags,
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=ldflags,
        extra_include_paths=[str(_cutlass_include_path())],
        with_cuda=True,
        verbose=False,
    )


@dataclass
class TailWorkspace:
    barrier_state: torch.Tensor
    block_max: torch.Tensor
    block_index: torch.Tensor
    block_sum: torch.Tensor
    global_values: torch.Tensor
    token_ids: torch.Tensor
    corrected_logits: torch.Tensor
    draft_probs: torch.Tensor
    confidence_logits: torch.Tensor
    trace: torch.Tensor


@dataclass(frozen=True)
class FullWeights:
    top: tuple[torch.Tensor, ...]
    layers: tuple[torch.Tensor, ...]


@dataclass
class FullWorkspace:
    tensors: tuple[torch.Tensor, ...]

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.tensors[1]

    @property
    def base_logits(self) -> torch.Tensor:
        return self.tensors[12]

    @property
    def corrected_logits(self) -> torch.Tensor:
        return self.tensors[13]

    @property
    def draft_probs(self) -> torch.Tensor:
        return self.tensors[14]

    @property
    def token_ids(self) -> torch.Tensor:
        return self.tensors[15]

    @property
    def confidence_logits(self) -> torch.Tensor:
        return self.tensors[16]

    @property
    def trace(self) -> torch.Tensor:
        return self.tensors[24]

    @property
    def verify_input_ids(self) -> torch.Tensor:
        return self.tensors[26]


@dataclass
class V4SchedulerWorkspace:
    scratch: torch.Tensor
    trace_records: torch.Tensor
    trace_counts: torch.Tensor
    launch_audit: torch.Tensor


@dataclass
class V4ProposalOutputs:
    output_ids: torch.Tensor
    corrected_logits: torch.Tensor
    draft_probabilities: torch.Tensor
    confidence_logits: torch.Tensor
    calibrated_confidences: torch.Tensor
    scheduled_prefix_lengths: torch.Tensor
    scheduler_read_mask: torch.Tensor
    scheduler_summary: torch.Tensor


_V4_WORKSPACE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}


def view_v4_workspace_region(
    workspace: V4SchedulerWorkspace,
    name: str,
    batch_size: int = 1,
) -> torch.Tensor:
    """Return a typed alias of one semantic V4 scratch region for debugging."""

    from deepspec.megakernel.v4_abi import V4LaunchShape, build_v4_launch_abi

    region = build_v4_launch_abi(None, V4LaunchShape(batch_size=batch_size)).workspace.region(name)
    raw = workspace.scratch.narrow(0, region.offset_bytes, region.nbytes)
    return raw.view(_V4_WORKSPACE_DTYPES[region.dtype]).view(region.shape)


def allocate_v4_scheduler_workspace(
    *,
    device: torch.device | str,
    extra_scratch_bytes: int = 0,
    batch_size: int = 1,
) -> V4SchedulerWorkspace:
    """Allocate reusable scheduler state; its zeroing is setup, not proposal work.

    ``extra_scratch_bytes`` appends caller-owned storage after the frozen V4
    workspace ABI. Semantic workspace-region views never include this tail.
    """

    from deepspec.megakernel.v4_abi import V4LaunchShape, build_v4_launch_abi

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("V4 scheduler workspace requires a CUDA device")
    if extra_scratch_bytes < 0:
        raise ValueError("extra_scratch_bytes must be non-negative")
    abi = build_v4_launch_abi(None, V4LaunchShape(batch_size=batch_size))
    return V4SchedulerWorkspace(
        scratch=torch.zeros(
            abi.workspace.total_bytes + extra_scratch_bytes,
            dtype=torch.uint8,
            device=device,
        ),
        trace_records=torch.empty(
            abi.tensor("trace_records").shape,
            dtype=torch.int64,
            device=device,
        ),
        trace_counts=torch.empty(
            abi.tensor("trace_counts").shape,
            dtype=torch.int32,
            device=device,
        ),
        launch_audit=torch.empty(
            abi.tensor("launch_audit").shape,
            dtype=torch.uint64,
            device=device,
        ),
    )


def allocate_v4_proposal_outputs(
    *,
    device: torch.device | str,
    batch_size: int = 1,
) -> V4ProposalOutputs:
    """Allocate reusable numerical proposal outputs without launching CUDA work."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("V4 proposal outputs require a CUDA device")
    b = int(batch_size)
    return V4ProposalOutputs(
        output_ids=torch.empty(b, 6, dtype=torch.int32, device=device),
        corrected_logits=torch.empty(b, 5, 129280, dtype=torch.float32, device=device),
        draft_probabilities=torch.empty(b, 5, 129280, dtype=torch.float32, device=device),
        confidence_logits=torch.empty(b, 5, dtype=torch.float32, device=device),
        calibrated_confidences=torch.empty(b, 5, dtype=torch.float32, device=device),
        scheduled_prefix_lengths=torch.empty(b, dtype=torch.int32, device=device),
        scheduler_read_mask=torch.empty(b, 5, dtype=torch.uint8, device=device),
        scheduler_summary=torch.empty(4, dtype=torch.float32, device=device),
    )


def run_v4_scheduler_smoke(
    weight_arena: torch.Tensor,
    weight_offsets: torch.Tensor,
    *,
    workspace: V4SchedulerWorkspace,
    proposal_epoch: int,
) -> V4SchedulerWorkspace:
    extension = load_dspark_v4_scheduler_extension(instrumented=True)
    extension.scheduler_smoke(
        weight_arena,
        weight_offsets,
        workspace.scratch,
        workspace.trace_records,
        workspace.trace_counts,
        workspace.launch_audit,
        int(proposal_epoch),
    )
    return workspace


def run_v4_main_stage(
    main_hidden: torch.Tensor,
    weight_arena: torch.Tensor,
    weight_offsets: torch.Tensor,
    *,
    workspace: V4SchedulerWorkspace,
    proposal_epoch: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the debug main-projection phase inside the persistent DAG.

    This is an intermediate-parity surface, not a production proposal API.
    """

    if output is None:
        output = torch.empty(
            1,
            1,
            4096,
            dtype=torch.bfloat16,
            device=main_hidden.device,
        )
    extension = load_dspark_v4_scheduler_extension()
    extension.main_stage(
        main_hidden,
        weight_arena,
        weight_offsets,
        workspace.scratch,
        workspace.trace_records,
        workspace.trace_counts,
        workspace.launch_audit,
        output,
        int(proposal_epoch),
    )
    return output


def run_v4_front_stage(
    anchor: torch.Tensor,
    main_hidden: torch.Tensor,
    weight_arena: torch.Tensor,
    weight_offsets: torch.Tensor,
    *,
    workspace: V4SchedulerWorkspace,
    proposal_epoch: int,
    main_output: torch.Tensor | None = None,
    embedding_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both independent numerical front roots inside one persistent launch."""

    if main_output is None:
        main_output = torch.empty(
            1,
            1,
            4096,
            dtype=torch.bfloat16,
            device=main_hidden.device,
        )
    if embedding_output is None:
        embedding_output = torch.empty(
            1,
            5,
            4,
            4096,
            dtype=torch.bfloat16,
            device=main_hidden.device,
        )
    extension = load_dspark_v4_scheduler_extension()
    extension.front_stage(
        anchor,
        main_hidden,
        weight_arena,
        weight_offsets,
        workspace.scratch,
        workspace.trace_records,
        workspace.trace_counts,
        workspace.launch_audit,
        main_output,
        embedding_output,
        int(proposal_epoch),
    )
    return main_output, embedding_output


def run_v4_front_kv_stage(
    anchor: torch.Tensor,
    main_hidden: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    weight_arena: torch.Tensor,
    weight_offsets: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    start_pos: int,
    draft_layer_count: int = 3,
    workspace: V4SchedulerWorkspace,
    proposal_epoch: int,
    main_output: torch.Tensor | None = None,
    embedding_output: torch.Tensor | None = None,
    proposal_outputs: V4ProposalOutputs | None = None,
    uniforms: torch.Tensor | None = None,
    sampling_temperature: float = 0.0,
    sts_temperatures: torch.Tensor | None = None,
    steps_per_second: torch.Tensor | None = None,
    instrumented: bool = False,
    relaxed_dag: bool = False,
    greedy_tail: bool = False,
    static_tp2_tail: bool = False,
    batch_size: int = 1,
    tp2_ablate: bool = False,
    full_loop_device_epoch: bool = False,
    execution_mask: int | None = None,
    draft_layer_mask: int | None = None,
    compiled_execution_mask: int | None = None,
    compiled_draft_layer_mask: int | None = None,
    extension_module: ModuleType | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run both front roots and all three incremental target-KV updates.

    ``greedy_tail`` is a zero-temperature serving mode. It materializes only
    output IDs and confidence outputs from the historical tail surface; draft
    probabilities and prefix-scheduler outputs are intentionally unwritten.
    ``static_tp2_tail`` selects the fixed 40/60 rank-local vocabulary program
    and its separate TP2 lifecycle surface.

    ``execution_mask`` and ``draft_layer_mask`` are an experimental graph-graft
    surface.  Omitting both preserves the production one-launch path exactly.
    ``extension_module`` binds a previously built module for isolated A/B
    harnesses; callers must supply the module matching all launch options.
    """

    b = int(batch_size)
    if main_output is None:
        main_output = torch.empty(
            b,
            1,
            4096,
            dtype=torch.bfloat16,
            device=main_hidden.device,
        )
    if embedding_output is None:
        embedding_output = torch.empty(
            b,
            5,
            4,
            4096,
            dtype=torch.bfloat16,
            device=main_hidden.device,
        )
    if proposal_outputs is None:
        proposal_outputs = allocate_v4_proposal_outputs(device=main_hidden.device, batch_size=b)
    if sampling_temperature < 0.0:
        raise ValueError("sampling_temperature must be non-negative")
    if sampling_temperature >= 1.0e-5 and uniforms is None:
        raise ValueError("positive sampling_temperature requires controlled uniforms")
    if (execution_mask is None) != (draft_layer_mask is None):
        raise ValueError("execution_mask and draft_layer_mask must be provided together")
    if execution_mask is not None and not 0 <= int(execution_mask) <= V4_EXECUTE_ALL:
        raise ValueError("execution_mask contains unsupported phase bits")
    if draft_layer_mask is not None and not 1 <= int(draft_layer_mask) <= 0x7:
        raise ValueError("draft_layer_mask must select at least one of three draft layers")
    calibration_enabled = sts_temperatures is not None
    prefix_enabled = steps_per_second is not None
    if uniforms is None:
        uniforms = torch.empty(b, 5, dtype=torch.float32, device=main_hidden.device)
    if sts_temperatures is None:
        sts_temperatures = torch.empty(5, dtype=torch.float32, device=main_hidden.device)
    if steps_per_second is None:
        steps_per_second = torch.empty(b * 6 + 1, dtype=torch.float32, device=main_hidden.device)
    extension = extension_module
    if extension is None:
        extension = load_dspark_v4_scheduler_extension(
            instrumented=instrumented,
            relaxed_dag=relaxed_dag,
            greedy_tail=greedy_tail,
            static_tp2_tail=static_tp2_tail,
            batch_size=b,
            tp2_ablate=tp2_ablate,
            full_loop_device_epoch=full_loop_device_epoch,
            compiled_execution_mask=compiled_execution_mask,
            compiled_draft_layer_mask=compiled_draft_layer_mask,
        )
    launch = extension.front_kv_stage if execution_mask is None else extension.front_kv_stage_masked
    arguments = (
        anchor,
        main_hidden,
        rope_cos_sin,
        weight_arena,
        weight_offsets,
        workspace.scratch,
        workspace.trace_records,
        workspace.trace_counts,
        workspace.launch_audit,
        main_output,
        embedding_output,
        kv_cache,
        proposal_outputs.output_ids,
        proposal_outputs.corrected_logits,
        proposal_outputs.draft_probabilities,
        proposal_outputs.confidence_logits,
        proposal_outputs.calibrated_confidences,
        proposal_outputs.scheduled_prefix_lengths,
        proposal_outputs.scheduler_read_mask,
        proposal_outputs.scheduler_summary,
        uniforms,
        sts_temperatures,
        steps_per_second,
        float(sampling_temperature),
        calibration_enabled,
        prefix_enabled,
        int(start_pos),
        int(draft_layer_count),
        int(proposal_epoch),
    )
    if execution_mask is None:
        launch(*arguments)
    else:
        assert draft_layer_mask is not None
        launch(*arguments, int(execution_mask), int(draft_layer_mask))
    return main_output, embedding_output, kv_cache


def allocate_tail_workspace(
    *,
    steps: int,
    vocab_size: int,
    device: torch.device | str,
) -> TailWorkspace:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("tail workspace requires a CUDA device")
    properties = torch.cuda.get_device_properties(device)
    workers = int(properties.multi_processor_count)
    return TailWorkspace(
        barrier_state=torch.zeros(2, dtype=torch.int32, device=device),
        block_max=torch.empty(workers, dtype=torch.float32, device=device),
        block_index=torch.empty(workers, dtype=torch.int32, device=device),
        block_sum=torch.empty(workers, dtype=torch.float32, device=device),
        global_values=torch.empty(2, dtype=torch.float32, device=device),
        token_ids=torch.empty(steps, dtype=torch.int64, device=device),
        corrected_logits=torch.empty(steps, vocab_size, dtype=torch.float32, device=device),
        draft_probs=torch.empty(steps, vocab_size, dtype=torch.float32, device=device),
        confidence_logits=torch.empty(steps, dtype=torch.float32, device=device),
        trace=torch.empty(workers, steps, TRACE_COLUMNS, dtype=torch.int64, device=device),
    )


def pack_full_weights(model) -> FullWeights:
    if model.markov_head is None or model.confidence_head is None:
        raise ValueError("full megakernel requires Markov and confidence heads")
    top = (
        model.embed_tokens.weight,
        model.fc.weight,
        model.hidden_norm.weight,
        model.norm.weight,
        model.lm_head.weight,
        model.markov_head.markov_w1.weight,
        model.markov_head.markov_w2.weight,
        model.confidence_head.proj.weight,
        model.confidence_head.proj.bias,
    )
    layers = []
    for layer in model.layers:
        layers.extend(
            (
                layer.input_layernorm.weight,
                layer.self_attn.q_proj.weight,
                layer.self_attn.k_proj.weight,
                layer.self_attn.v_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.self_attn.q_norm.weight,
                layer.self_attn.k_norm.weight,
                layer.post_attention_layernorm.weight,
                layer.mlp.gate_proj.weight,
                layer.mlp.up_proj.weight,
                layer.mlp.down_proj.weight,
            )
        )
    return FullWeights(
        top=tuple(tensor.detach().contiguous() for tensor in top),
        layers=tuple(tensor.detach().contiguous() for tensor in layers),
    )


def allocate_full_workspace(
    *,
    block_size: int,
    context_length: int,
    max_cache_length: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> FullWorkspace:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("full workspace requires a CUDA device")
    workers = int(torch.cuda.get_device_properties(device).multi_processor_count)
    q_width = num_attention_heads * head_dim
    kv_width = num_key_value_heads * head_dim
    block_rows = max(16, ((block_size + 15) // 16) * 16)
    context_rows = max(16, ((context_length + 15) // 16) * 16)
    tensors = (
        torch.empty(context_rows, hidden_size, dtype=dtype, device=device),
        torch.empty(max(block_rows, 64), hidden_size, dtype=dtype, device=device),
        torch.empty(block_rows, hidden_size, dtype=dtype, device=device),
        torch.empty(block_rows, q_width, dtype=dtype, device=device),
        torch.empty(context_rows, kv_width, dtype=dtype, device=device),
        torch.empty(context_rows, kv_width, dtype=dtype, device=device),
        torch.empty(block_rows, kv_width, dtype=dtype, device=device),
        torch.empty(block_rows, kv_width, dtype=dtype, device=device),
        torch.empty(block_rows, q_width, dtype=dtype, device=device),
        torch.empty(block_rows, hidden_size, dtype=dtype, device=device),
        torch.empty(block_rows, intermediate_size, dtype=dtype, device=device),
        torch.empty(block_rows, intermediate_size, dtype=dtype, device=device),
        torch.empty(block_rows, vocab_size, dtype=dtype, device=device),
        torch.empty(block_size, vocab_size, dtype=torch.float32, device=device),
        torch.empty(block_size, vocab_size, dtype=torch.float32, device=device),
        torch.empty(block_size, dtype=torch.int64, device=device),
        torch.empty(block_size, dtype=torch.float32, device=device),
        torch.zeros(
            num_layers,
            max_cache_length,
            num_key_value_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        torch.zeros(
            num_layers,
            max_cache_length,
            num_key_value_heads,
            head_dim,
            dtype=dtype,
            device=device,
        ),
        torch.empty(workers, dtype=torch.float32, device=device),
        torch.empty(workers, dtype=torch.int32, device=device),
        torch.empty(workers, dtype=torch.float32, device=device),
        torch.empty(4, dtype=torch.float32, device=device),
        torch.empty(
            context_rows,
            num_layers * hidden_size,
            dtype=dtype,
            device=device,
        ),
        torch.empty(
            workers,
            FULL_TRACE_PHASE_CAPACITY,
            FULL_TRACE_COLUMNS,
            dtype=torch.int64,
            device=device,
        ),
        torch.empty(
            max(
                8 * max(block_rows, context_rows) * hidden_size,
                16 * block_rows * intermediate_size,
            ),
            dtype=torch.float32,
            device=device,
        ),
        torch.empty(block_size + 1, dtype=torch.int64, device=device),
    )
    return FullWorkspace(tensors=tensors)


def run_tail_megakernel(
    base_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    first_prev_token: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    confidence_weight: torch.Tensor,
    confidence_bias: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    temperature: float,
    workspace: TailWorkspace,
    instrumented: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    extension = load_dspark_tail_extension(instrumented=instrumented)
    return tuple(
        extension.tail(
            base_logits,
            hidden_states,
            first_prev_token,
            markov_w1,
            markov_w2,
            confidence_weight,
            confidence_bias,
            uniforms,
            float(temperature),
            workspace.barrier_state,
            workspace.block_max,
            workspace.block_index,
            workspace.block_sum,
            workspace.global_values,
            workspace.token_ids,
            workspace.corrected_logits,
            workspace.draft_probs,
            workspace.confidence_logits,
            workspace.trace,
        )
    )


def run_full_megakernel(
    draft_input_ids: torch.Tensor,
    target_hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    uniforms: torch.Tensor,
    *,
    temperature: float,
    past_length: int,
    rms_epsilon: float,
    rope_theta: float,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    weights: FullWeights,
    workspace: FullWorkspace,
    instrumented: bool = False,
    first_prev_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    extension = load_dspark_full_extension(instrumented=instrumented)
    if first_prev_token is None:
        first_prev_token = draft_input_ids[:1]
    outputs = tuple(
        extension.full(
            draft_input_ids,
            first_prev_token,
            target_hidden_states,
            position_ids,
            uniforms,
            float(temperature),
            int(past_length),
            float(rms_epsilon),
            float(rope_theta),
            int(num_attention_heads),
            int(num_key_value_heads),
            int(head_dim),
            list(weights.top),
            list(weights.layers),
            list(workspace.tensors),
        )
    )
    block_size = int(draft_input_ids.numel())
    return (
        outputs[0][:block_size],
        outputs[1][:block_size],
        *outputs[2:],
    )
