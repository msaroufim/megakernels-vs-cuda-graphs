from __future__ import annotations

from dataclasses import dataclass
from math import prod

from deepspec.megakernel.contract import DeepSeekV4MegaKernelSpec

_DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint32": 4,
    "uint64": 8,
}


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


# Build-time batch specializations with a generated header in the tree.
_SUPPORTED_BATCH_SIZES = frozenset({1, 2, 4, 8})


@dataclass(frozen=True)
class V4LaunchShape:
    """Runtime dimensions supported by the GB300 specialization."""

    batch_size: int = 1
    main_rows: int = 1
    workers: int = 152
    trace_events_per_worker: int = 2048
    local_queue_depth: int = 16

    def require_supported(self) -> None:
        # Batched serving (R3): batch is a BUILD-time specialization. The
        # device bodies carry the batch's rows on the tensor-core N mode for
        # the weight-bound bands and tile over batch elsewhere; the routed
        # route-table encoding caps the useful range at 8 (kMemberBits = 8
        # packs 8 members of a <=256-row table into one uint64).
        if self.batch_size not in _SUPPORTED_BATCH_SIZES:
            raise ValueError(
                "the DeepSeek-V4-Flash specialization supports batch_size in "
                f"{sorted(_SUPPORTED_BATCH_SIZES)}, not {self.batch_size}"
            )
        if self.main_rows <= 0:
            raise ValueError("main_rows must be positive")
        if self.workers != 152:
            raise ValueError(
                "GB300 execution requires exactly one persistent CTA on each of 152 SMs"
            )
        if self.trace_events_per_worker <= 0:
            raise ValueError("trace_events_per_worker must be positive")
        if self.local_queue_depth <= 0:
            raise ValueError("local_queue_depth must be positive")


@dataclass(frozen=True)
class TensorABI:
    name: str
    dtype: str
    shape: tuple[int, ...]
    direction: str
    purpose: str

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class WorkspaceRegion:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    nbytes: int
    lifetime: str

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.nbytes


@dataclass(frozen=True)
class V4WeightArenaABI:
    """One device-side descriptor replaces thousands of CUDA parameters.

    Setup may pack and upload weights once. A proposal launch receives only a
    pointer to this stable descriptor; it may not launch conversion kernels.
    """

    descriptor_name: str = "weights"
    arena_pointer_field: str = "arena"
    offset_table_pointer_field: str = "tensor_offsets"
    metadata_pointer_field: str = "tensor_metadata"
    source_tensor_count: int = 4707
    runtime_tensor_count: int = 4704
    source_stored_bytes: int = 12_980_961_820
    setup_outside_proposal: bool = True
    conversions_during_proposal: int = 0


@dataclass(frozen=True)
class V4WorkspaceLayout:
    regions: tuple[WorkspaceRegion, ...]
    total_bytes: int
    alignment: int

    def region(self, name: str) -> WorkspaceRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)


@dataclass(frozen=True)
class V4LaunchABI:
    spec: DeepSeekV4MegaKernelSpec
    shape: V4LaunchShape
    inputs: tuple[TensorABI, ...]
    inout: tuple[TensorABI, ...]
    outputs: tuple[TensorABI, ...]
    workspace: V4WorkspaceLayout
    weight_arena: V4WeightArenaABI
    top_level_kernel_launches: int = 1
    nested_kernel_launches: int = 0

    def tensor(self, name: str) -> TensorABI:
        for tensor in (*self.inputs, *self.inout, *self.outputs):
            if tensor.name == name:
                return tensor
        raise KeyError(name)


class _WorkspaceBuilder:
    def __init__(self, alignment: int = 256):
        self.alignment = alignment
        self.offset = 0
        self.regions: list[WorkspaceRegion] = []

    def add(
        self,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        lifetime: str,
    ) -> None:
        self.offset = _align_up(self.offset, self.alignment)
        nbytes = prod(shape) * _DTYPE_BYTES[dtype]
        self.regions.append(WorkspaceRegion(name, dtype, shape, self.offset, nbytes, lifetime))
        self.offset += nbytes

    def finish(self) -> V4WorkspaceLayout:
        return V4WorkspaceLayout(
            regions=tuple(self.regions),
            total_bytes=_align_up(self.offset, self.alignment),
            alignment=self.alignment,
        )


def _build_workspace(
    spec: DeepSeekV4MegaKernelSpec,
    shape: V4LaunchShape,
    *,
    phase_count: int,
) -> V4WorkspaceLayout:
    b = shape.batch_size
    s = spec.block_size
    h = spec.hidden_size
    hc = spec.hc_multiplier
    v = spec.vocab_size
    heads = spec.num_attention_heads
    d = spec.head_dim
    topk = spec.num_activated_experts
    expert = spec.expert_intermediate_size
    builder = _WorkspaceBuilder()

    # Correctness-first allocation keeps semantic boundaries distinct. Once
    # parity is locked, lifetime-disjoint regions can be aliased deliberately.
    builder.add(
        "main_hidden_quantized",
        "uint8",
        (b, shape.main_rows, spec.target_feature_width),
        "main_projection",
    )
    builder.add(
        "main_hidden_scales",
        "uint8",
        (b, shape.main_rows, spec.target_feature_width // 128),
        "main_projection",
    )
    builder.add(
        "main_projection_partials",
        "float32",
        (b, shape.main_rows, h, 8),
        "main_projection",
    )
    builder.add("main_projection_fp32", "float32", (b, shape.main_rows, h), "main_norm")
    builder.add("main_rms_partial", "float32", (b, shape.main_rows, 8), "main_norm")
    builder.add("main_projected", "bfloat16", (b, shape.main_rows, h), "proposal")
    builder.add(
        "main_projected_quantized",
        "uint8",
        (b, shape.main_rows, h),
        "main_kv",
    )
    builder.add(
        "main_projected_scales",
        "uint8",
        (b, shape.main_rows, h // 128),
        "main_kv",
    )
    builder.add(
        "main_kv_partials",
        "float32",
        (spec.num_draft_layers, b, shape.main_rows, spec.kv_width, 4),
        "main_kv",
    )
    builder.add("hidden_streams", "bfloat16", (b, s, hc, h), "proposal")
    builder.add(
        "all_layer_main_kv",
        "bfloat16",
        (spec.num_draft_layers, b, shape.main_rows, d),
        "until_attention",
    )
    builder.add("hc_mixes", "float32", (b, s, (2 + hc) * hc), "subblock")
    builder.add("hc_pre", "float32", (b, s, hc), "subblock")
    builder.add("hc_post", "float32", (b, s, hc), "subblock")
    builder.add("hc_comb", "float32", (b, s, hc, hc), "subblock")
    builder.add("normalized_hidden", "bfloat16", (b, s, h), "subblock")
    builder.add("attn_input_quantized", "uint8", (b, s, h), "attention")
    builder.add("attn_input_scales", "uint8", (b, s, h // 128), "attention")
    builder.add(
        "q_a_partials",
        "float32",
        (b, s, spec.q_lora_rank, 4),
        "attention",
    )
    builder.add("q_lora", "bfloat16", (b, s, spec.q_lora_rank), "attention")
    builder.add(
        "q_lora_quantized",
        "uint8",
        (b, s, spec.q_lora_rank),
        "attention",
    )
    builder.add(
        "q_lora_scales",
        "uint8",
        (b, s, spec.q_lora_rank // 128),
        "attention",
    )
    builder.add("query_projection", "bfloat16", (b, s, heads, d), "attention")
    builder.add("query_inverse_rms", "bfloat16", (b, s, heads), "attention")
    builder.add("queries", "bfloat16", (b, s, heads, d), "attention")
    builder.add(
        "draft_kv_partials",
        "float32",
        (b, s, spec.kv_width, 4),
        "attention",
    )
    builder.add("draft_kv", "bfloat16", (b, s, d), "attention")
    builder.add("attention_accumulator", "float32", (b, s, heads, d), "attention")
    builder.add("attention_raw", "bfloat16", (b, s, heads, d), "attention")
    builder.add("attention_values", "bfloat16", (b, s, heads, d), "attention")
    builder.add(
        "output_lora",
        "bfloat16",
        (b, s, spec.output_groups, spec.output_lora_rank),
        "attention",
    )
    builder.add(
        "output_lora_quantized",
        "uint8",
        (b, s, spec.output_groups, spec.output_lora_rank),
        "attention",
    )
    builder.add(
        "output_lora_scales",
        "uint8",
        (b, s, spec.output_groups * spec.output_lora_rank // 128),
        "attention",
    )
    builder.add(
        "attention_output_partials",
        "float32",
        (b, s, h, 4),
        "attention",
    )
    builder.add("attention_output", "bfloat16", (b, s, h), "attention")
    builder.add("attention_hidden_streams", "bfloat16", (b, s, hc, h), "attention")
    builder.add("ffn_hc_mixes", "float32", (b, s, (2 + hc) * hc), "ffn")
    builder.add("ffn_hc_pre", "float32", (b, s, hc), "ffn")
    builder.add("ffn_hc_post", "float32", (b, s, hc), "ffn")
    builder.add("ffn_hc_comb", "float32", (b, s, hc, hc), "ffn")
    builder.add("ffn_normalized_hidden", "bfloat16", (b, s, h), "ffn")
    builder.add("ffn_input_quantized", "uint8", (b, s, h), "ffn")
    builder.add("ffn_input_scales", "uint8", (b, s, h // 128), "ffn")
    builder.add("router_scores", "float32", (b, s, spec.num_routed_experts), "moe")
    builder.add("router_indices", "int32", (b, s, topk), "moe")
    builder.add("router_weights", "float32", (b, s, topk), "moe")
    builder.add("routed_w13", "bfloat16", (b, s, topk, 2, expert), "moe")
    builder.add("shared_w13", "bfloat16", (b, s, 2, expert), "moe")
    builder.add("routed_swiglu", "bfloat16", (b, s, topk, expert), "moe")
    builder.add("routed_swiglu_quantized", "uint8", (b, s, topk, expert), "moe")
    builder.add(
        "routed_swiglu_scales",
        "uint8",
        (b, s, topk, expert // 128),
        "moe",
    )
    builder.add("shared_swiglu", "bfloat16", (b, s, expert), "moe")
    builder.add("shared_swiglu_quantized", "uint8", (b, s, expert), "moe")
    builder.add("shared_swiglu_scales", "uint8", (b, s, expert // 128), "moe")
    builder.add("routed_output_partials", "bfloat16", (b, s, topk, h), "moe")
    builder.add("routed_output", "float32", (b, s, h), "moe")
    builder.add("shared_output", "bfloat16", (b, s, h), "moe")
    builder.add("head_hidden", "bfloat16", (b, s, h), "tail")
    builder.add("head_normalized", "bfloat16", (b, s, h), "tail")
    builder.add("base_logits", "float32", (b, s, v), "tail")
    builder.add("markov_embeddings", "bfloat16", (b, s, spec.markov_rank), "tail")
    builder.add("markov_logits_row", "float32", (b, v), "tail_step")
    builder.add("softmax_partial_max", "float32", (b, s, 512), "tail_step")
    builder.add("softmax_partial_sum", "float32", (b, s, 512), "tail_step")
    builder.add("sample_scan", "float32", (b, s, 512), "tail_step")
    builder.add("prefix_survival", "float32", (b, s), "scheduler")
    builder.add("prefix_heap", "uint64", (b * s,), "scheduler")

    # Hazy-style persistent execution state: ready phase IDs are local-first,
    # phase tile claims are atomic, and idle CTAs steal before recording wait.
    builder.add("phase_dependency_arrivals", "uint64", (phase_count,), "kernel")
    builder.add("phase_next_tile", "uint64", (phase_count,), "kernel")
    builder.add("phase_completed_tiles", "uint64", (phase_count,), "kernel")
    builder.add("phase_publish_epoch", "uint64", (phase_count,), "kernel")
    builder.add(
        "local_ready_queues",
        "uint32",
        (shape.workers, shape.local_queue_depth),
        "kernel",
    )
    builder.add("local_queue_state", "uint64", (shape.workers, 2), "kernel")
    builder.add("worker_state", "uint64", (shape.workers, 8), "kernel")
    return builder.finish()


def build_v4_launch_abi(
    spec: DeepSeekV4MegaKernelSpec | None = None,
    shape: V4LaunchShape | None = None,
) -> V4LaunchABI:
    """Build the concrete one-launch ABI for a V4-Flash proposal."""

    spec = spec or DeepSeekV4MegaKernelSpec()
    shape = shape or V4LaunchShape()
    spec.require_supported()
    shape.require_supported()

    # Local import prevents the phase schema from depending on the ABI builder.
    from deepspec.megakernel.v4_schedule import build_v4_phase_program

    phase_count = len(build_v4_phase_program(spec, shape))
    b = shape.batch_size
    s = spec.block_size
    v = spec.vocab_size
    scheduler_curve_size = b * (s + 1) + 1
    inputs = (
        TensorABI("anchor_token_ids", "int32", (b,), "input", "proposal anchor"),
        TensorABI(
            "main_hidden",
            "bfloat16",
            (b, shape.main_rows, spec.target_feature_width),
            "input",
            "concatenated target taps",
        ),
        TensorABI("start_pos", "int64", (1,), "input", "target decode position"),
        TensorABI(
            "rope_cos_sin",
            "float32",
            (shape.main_rows + s, spec.rope_head_dim // 2, 2),
            "input",
            "precomputed proposal-position RoPE",
        ),
        TensorABI("uniforms", "float32", (b, s), "input", "controlled sampling draws"),
        TensorABI(
            "sampling_temperature",
            "float32",
            (1,),
            "input",
            "explicit proposal sampling temperature",
        ),
        TensorABI(
            "sts_temperatures",
            "float32",
            (s,),
            "input",
            "per-position sequential temperature scaling",
        ),
        TensorABI(
            "steps_per_second",
            "float32",
            (scheduler_curve_size,),
            "input",
            "profiled SPS(B), indexed by verification token batch",
        ),
        TensorABI(
            "runtime_flags",
            "uint32",
            (3,),
            "input",
            "scheduler enable, trace enable, launch-audit enable",
        ),
    )
    inout = (
        TensorABI(
            "kv_cache",
            "bfloat16",
            (
                spec.num_draft_layers,
                b,
                spec.window_size,
                spec.kv_width,
            ),
            "inout",
            "three circular target-KV windows",
        ),
    )
    outputs = (
        TensorABI("output_ids", "int32", (b, s + 1), "output", "anchor plus drafts"),
        TensorABI(
            "corrected_logits",
            "float32",
            (b, s, v),
            "output",
            "base plus Markov logits",
        ),
        TensorABI(
            "draft_probabilities",
            "float32",
            (b, s, v),
            "output",
            "locally normalized proposal distributions",
        ),
        TensorABI(
            "confidence_logits",
            "float32",
            (b, s),
            "output",
            "raw learned confidence projection",
        ),
        TensorABI(
            "calibrated_confidences",
            "float32",
            (b, s),
            "output",
            "sigmoid(logit / STS temperature)",
        ),
        TensorABI(
            "scheduled_prefix_lengths",
            "int32",
            (b,),
            "output",
            "Algorithm 1 causal verification lengths",
        ),
        TensorABI(
            "scheduler_read_mask",
            "uint8",
            (b, s),
            "output",
            "proof surface for non-anticipating confidence reads",
        ),
        TensorABI(
            "scheduler_summary",
            "float32",
            (4,),
            "output",
            "expected accepts, target batch, throughput, candidates read",
        ),
        TensorABI(
            "trace_records",
            "int64",
            (shape.workers, shape.trace_events_per_worker, 8),
            "output",
            "same-launch GPU event ring",
        ),
        TensorABI(
            "trace_counts",
            "int32",
            (shape.workers,),
            "output",
            "per-worker event counts",
        ),
        TensorABI(
            "launch_audit",
            "uint64",
            (8,),
            "output",
            "kernel magic, entry/exit clocks, scheduler counters, overflow",
        ),
    )
    return V4LaunchABI(
        spec=spec,
        shape=shape,
        inputs=inputs,
        inout=inout,
        outputs=outputs,
        workspace=_build_workspace(spec, shape, phase_count=phase_count),
        weight_arena=V4WeightArenaABI(),
    )
