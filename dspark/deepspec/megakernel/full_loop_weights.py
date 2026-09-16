from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping

from deepspec.megakernel.full_loop_schedule import FullLoopSpec

_DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I64": 8,
    # The target checkpoint stores two FP4 values in each signed byte.  The
    # safetensors metadata therefore reports I8 and the packed physical shape.
    "I8": 1,
}


@dataclass(frozen=True)
class FullLoopCheckpointTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class FullLoopArenaTensor:
    tensor: FullLoopCheckpointTensor
    offset: int

    @property
    def end(self) -> int:
        return self.offset + self.tensor.nbytes


@dataclass(frozen=True)
class TargetLayerArenaMetadata:
    """Compact offset table consumed by a persistent target-layer scheduler.

    ``fixed_offsets`` follows ``fixed_fields`` and uses -1 for fields that are
    absent in a layer's attention/router mode.  ``expert_offsets`` is always a
    routed-expert-major 256x6 table in w1/w2/w3 weight/scale order.  Keeping
    offsets instead of materialized pointers makes the table relocatable when
    the 145 GiB raw-weight arena is allocated at a different address.
    """

    layer_id: int
    compress_ratio: int
    fixed_fields: tuple[str, ...]
    fixed_offsets: tuple[int, ...]
    expert_offsets: tuple[tuple[int, ...], ...]

    def offset(self, field: str) -> int:
        try:
            index = self.fixed_fields.index(field)
        except ValueError as error:
            raise KeyError(field) from error
        return self.fixed_offsets[index]

    def flatten(self) -> tuple[int, ...]:
        return self.fixed_offsets + tuple(
            offset for expert in self.expert_offsets for offset in expert
        )


@dataclass(frozen=True)
class TargetFlashInferLayerArenaMetadata:
    """Relocatable bindings for one compiler-packed FlashInfer expert block."""

    layer_id: int
    compress_ratio: int
    fixed_fields: tuple[str, ...]
    fixed_offsets: tuple[int, ...]
    gemm1_weight_offset: int
    gemm1_scale_offset: int
    gemm2_weight_offset: int
    gemm2_scale_offset: int

    def offset(self, field: str) -> int:
        try:
            index = self.fixed_fields.index(field)
        except ValueError as error:
            raise KeyError(field) from error
        return self.fixed_offsets[index]

    @property
    def expert_offsets(self) -> tuple[int, int, int, int]:
        return (
            self.gemm1_weight_offset,
            self.gemm1_scale_offset,
            self.gemm2_weight_offset,
            self.gemm2_scale_offset,
        )

    def flatten(self) -> tuple[int, ...]:
        return self.fixed_offsets + self.expert_offsets


def _tensor(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> FullLoopCheckpointTensor:
    return FullLoopCheckpointTensor(name, dtype, shape)


def expected_target_globals(
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopCheckpointTensor, ...]:
    """Raw target tensors needed on both sides of the persistent loop."""

    spec = spec or FullLoopSpec()
    spec.require_supported()
    hc_width = 4 * spec.hidden_size
    return (
        _tensor("embed.weight", "BF16", (spec.vocab_size, spec.hidden_size)),
        _tensor("hc_head_base", "F32", (4,)),
        _tensor("hc_head_fn", "F32", (4, hc_width)),
        _tensor("hc_head_scale", "F32", (1,)),
        _tensor("head.weight", "BF16", (spec.vocab_size, spec.hidden_size)),
        _tensor("norm.weight", "BF16", (spec.hidden_size,)),
    )


def target_layer_compress_ratio(layer_id: int) -> int:
    """Return the frozen V4-Flash target attention mode for one block."""

    if not 0 <= layer_id < 43:
        raise ValueError(f"target layer_id must be in [0, 43), got {layer_id}")
    if layer_id < 2:
        return 0
    return 4 if layer_id % 2 == 0 else 128


def expected_target_layer_weights(
    layer_id: int,
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopCheckpointTensor, ...]:
    """Frozen raw checkpoint ABI for one target block.

    The target alternates ratio-4 and ratio-128 compressed attention after two
    ratio-0 blocks.  Layers 0-2 use token-id hash routing and carry ``tid2eid``
    instead of the learned-gate bias used by layers 3-42.  Keeping this manifest
    in raw safetensors form lets the standalone loader place weights directly in
    their final GPU arena without constructing a second 167 GB model object.
    """

    spec = spec or FullLoopSpec()
    spec.require_supported()
    ratio = target_layer_compress_ratio(layer_id)
    prefix = f"layers.{layer_id}"
    h = spec.hidden_size
    q_rank = 1024
    heads = spec.attention_heads
    head_dim = spec.head_dim
    q_width = heads * head_dim
    o_rank = 128
    o_width = heads * o_rank
    expert_width = spec.expert_intermediate_size
    entries = [
        _tensor(f"{prefix}.hc_attn_base", "F32", (24,)),
        _tensor(f"{prefix}.hc_attn_fn", "F32", (24, 4 * h)),
        _tensor(f"{prefix}.hc_attn_scale", "F32", (3,)),
        _tensor(f"{prefix}.hc_ffn_base", "F32", (24,)),
        _tensor(f"{prefix}.hc_ffn_fn", "F32", (24, 4 * h)),
        _tensor(f"{prefix}.hc_ffn_scale", "F32", (3,)),
        _tensor(f"{prefix}.attn_norm.weight", "BF16", (h,)),
        _tensor(f"{prefix}.ffn_norm.weight", "BF16", (h,)),
        _tensor(f"{prefix}.attn.attn_sink", "F32", (heads,)),
        _tensor(f"{prefix}.attn.wq_a.weight", "F8_E4M3", (q_rank, h)),
        _tensor(f"{prefix}.attn.wq_a.scale", "F8_E8M0", (q_rank // 128, h // 128)),
        _tensor(f"{prefix}.attn.q_norm.weight", "BF16", (q_rank,)),
        _tensor(f"{prefix}.attn.wq_b.weight", "F8_E4M3", (q_width, q_rank)),
        _tensor(
            f"{prefix}.attn.wq_b.scale",
            "F8_E8M0",
            (q_width // 128, q_rank // 128),
        ),
        _tensor(f"{prefix}.attn.wkv.weight", "F8_E4M3", (head_dim, h)),
        _tensor(f"{prefix}.attn.wkv.scale", "F8_E8M0", (head_dim // 128, h // 128)),
        _tensor(f"{prefix}.attn.kv_norm.weight", "BF16", (head_dim,)),
        _tensor(f"{prefix}.attn.wo_a.weight", "F8_E4M3", (o_width, h)),
        _tensor(f"{prefix}.attn.wo_a.scale", "F8_E8M0", (o_width // 128, h // 128)),
        _tensor(f"{prefix}.attn.wo_b.weight", "F8_E4M3", (h, o_width)),
        _tensor(f"{prefix}.attn.wo_b.scale", "F8_E8M0", (h // 128, o_width // 128)),
    ]
    if ratio:
        compressor_width = (2 if ratio == 4 else 1) * head_dim
        entries.extend(
            (
                _tensor(f"{prefix}.attn.compressor.ape", "F32", (ratio, compressor_width)),
                _tensor(f"{prefix}.attn.compressor.norm.weight", "BF16", (head_dim,)),
                _tensor(f"{prefix}.attn.compressor.wgate.weight", "BF16", (compressor_width, h)),
                _tensor(f"{prefix}.attn.compressor.wkv.weight", "BF16", (compressor_width, h)),
            )
        )
    if ratio == 4:
        # Ratio-4 learned indexer and its overlapping 128-wide compressor.
        entries.extend(
            (
                _tensor(f"{prefix}.attn.indexer.wq_b.weight", "F8_E4M3", (heads * 128, q_rank)),
                _tensor(
                    f"{prefix}.attn.indexer.wq_b.scale",
                    "F8_E8M0",
                    (heads, q_rank // 128),
                ),
                _tensor(f"{prefix}.attn.indexer.weights_proj.weight", "BF16", (heads, h)),
                _tensor(f"{prefix}.attn.indexer.compressor.ape", "F32", (4, 2 * 128)),
                _tensor(f"{prefix}.attn.indexer.compressor.norm.weight", "BF16", (128,)),
                _tensor(f"{prefix}.attn.indexer.compressor.wgate.weight", "BF16", (2 * 128, h)),
                _tensor(f"{prefix}.attn.indexer.compressor.wkv.weight", "BF16", (2 * 128, h)),
            )
        )
    entries.append(_tensor(f"{prefix}.ffn.gate.weight", "BF16", (spec.routed_experts, h)))
    if layer_id < 3:
        entries.append(_tensor(f"{prefix}.ffn.gate.tid2eid", "I64", (spec.vocab_size, 6)))
    else:
        entries.append(_tensor(f"{prefix}.ffn.gate.bias", "F32", (spec.routed_experts,)))
    for expert in range(spec.routed_experts):
        expert_prefix = f"{prefix}.ffn.experts.{expert}"
        entries.extend(
            (
                _tensor(f"{expert_prefix}.w1.weight", "I8", (expert_width, h // 2)),
                _tensor(
                    f"{expert_prefix}.w1.scale",
                    "F8_E8M0",
                    (expert_width, h // 32),
                ),
                _tensor(f"{expert_prefix}.w2.weight", "I8", (h, expert_width // 2)),
                _tensor(
                    f"{expert_prefix}.w2.scale",
                    "F8_E8M0",
                    (h, expert_width // 32),
                ),
                _tensor(f"{expert_prefix}.w3.weight", "I8", (expert_width, h // 2)),
                _tensor(
                    f"{expert_prefix}.w3.scale",
                    "F8_E8M0",
                    (expert_width, h // 32),
                ),
            )
        )
    shared = f"{prefix}.ffn.shared_experts"
    entries.extend(
        (
            _tensor(f"{shared}.w1.weight", "F8_E4M3", (expert_width, h)),
            _tensor(f"{shared}.w1.scale", "F8_E8M0", (expert_width // 128, h // 128)),
            _tensor(f"{shared}.w2.weight", "F8_E4M3", (h, expert_width)),
            _tensor(f"{shared}.w2.scale", "F8_E8M0", (h // 128, expert_width // 128)),
            _tensor(f"{shared}.w3.weight", "F8_E4M3", (expert_width, h)),
            _tensor(f"{shared}.w3.scale", "F8_E8M0", (expert_width // 128, h // 128)),
        )
    )
    return tuple(entries)


def expected_target_layer42_weights(
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopCheckpointTensor, ...]:
    """Compatibility wrapper for the final ratio-4 target bridge."""

    return expected_target_layer_weights(42, spec)


def expected_target_layers_weights(
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopCheckpointTensor, ...]:
    """Return all target-block tensors in deterministic layer-major order."""

    spec = spec or FullLoopSpec()
    spec.require_supported()
    return tuple(
        tensor
        for layer_id in range(spec.target_layers)
        for tensor in expected_target_layer_weights(layer_id, spec)
    )


_FLASHINFER_EXPERT_FIELDS = (
    "ffn.compiled_experts.gemm1.weight",
    "ffn.compiled_experts.gemm1.scale",
    "ffn.compiled_experts.gemm2.weight",
    "ffn.compiled_experts.gemm2.scale",
)


def expected_target_flashinfer_layer_weights(
    layer_id: int,
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopCheckpointTensor, ...]:
    """Return one layer's final single-copy FlashInfer execution ABI.

    The four compiled expert tensors occupy exactly the bytes of the 256 raw
    W1/W2/W3 tensor sextuples they replace.  They are virtual arena fields,
    not checkpoint names; the streaming compiler fills them expert by expert.
    """

    spec = spec or FullLoopSpec()
    spec.require_supported()
    raw = expected_target_layer_weights(layer_id, spec)
    prefix = f"layers.{layer_id}."
    compiled = (
        _tensor(
            prefix + _FLASHINFER_EXPERT_FIELDS[0],
            "I8",
            (
                spec.routed_experts,
                2 * spec.expert_intermediate_size,
                spec.hidden_size // 2,
            ),
        ),
        _tensor(
            prefix + _FLASHINFER_EXPERT_FIELDS[1],
            "F8_E8M0",
            (
                spec.routed_experts,
                2 * spec.expert_intermediate_size,
                spec.hidden_size // 32,
            ),
        ),
        _tensor(
            prefix + _FLASHINFER_EXPERT_FIELDS[2],
            "I8",
            (
                spec.routed_experts,
                spec.hidden_size,
                spec.expert_intermediate_size // 2,
            ),
        ),
        _tensor(
            prefix + _FLASHINFER_EXPERT_FIELDS[3],
            "F8_E8M0",
            (
                spec.routed_experts,
                spec.hidden_size,
                spec.expert_intermediate_size // 32,
            ),
        ),
    )
    result: list[FullLoopCheckpointTensor] = []
    emitted = False
    for tensor in raw:
        if tensor.name.startswith(prefix + "ffn.experts."):
            if not emitted:
                result.extend(compiled)
                emitted = True
            continue
        result.append(tensor)
    if not emitted:
        raise ValueError(f"target layer {layer_id} contains no routed experts")
    return tuple(result)


def expected_target_flashinfer_layers_weights(
    spec: FullLoopSpec | None = None,
) -> tuple[FullLoopCheckpointTensor, ...]:
    spec = spec or FullLoopSpec()
    spec.require_supported()
    return tuple(
        tensor
        for layer_id in range(spec.target_layers)
        for tensor in expected_target_flashinfer_layer_weights(layer_id, spec)
    )


def target_weight_arena_layout(
    spec: FullLoopSpec | None = None,
    *,
    alignment: int = 256,
) -> tuple[FullLoopArenaTensor, ...]:
    """Lay out raw target tensors for direct checkpoint-to-GPU streaming."""

    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("target arena alignment must be a positive power of two")
    entries = expected_target_globals(spec) + expected_target_layers_weights(spec)
    offset = 0
    layout = []
    for tensor in entries:
        offset = (offset + alignment - 1) & -alignment
        layout.append(FullLoopArenaTensor(tensor, offset))
        offset += tensor.nbytes
    return tuple(layout)


def target_execution_arena_layout(
    spec: FullLoopSpec | None = None,
    *,
    alignment: int = 256,
    expert_backend: str = "custom",
    fp8_woa: bool = False,
) -> tuple[FullLoopArenaTensor, ...]:
    """Lay out the target weights in the ABI consumed by retained kernels.

    Most checkpoint tensors are already in their execution representation.
    Grouped ``wo_a`` is expanded from scaled FP8 to BF16, and target-block norm
    weights are promoted from BF16 to the FP32 ABI consumed by the retained
    kernels.  The final model norm remains BF16 for FlashInfer's head lowering.
    Routed FP4 scales keep their physical size and are permuted to stage-major
    order by the streaming loader.
    """

    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("target arena alignment must be a positive power of two")
    spec = spec or FullLoopSpec()
    spec.require_supported()
    if expert_backend == "custom":
        layer_entries = expected_target_layers_weights(spec)
    elif expert_backend == "flashinfer":
        layer_entries = expected_target_flashinfer_layers_weights(spec)
    else:
        raise ValueError(f"unsupported target expert backend {expert_backend!r}")
    entries = expected_target_globals(spec) + layer_entries
    offset = 0
    layout = []
    for checkpoint_tensor in entries:
        tensor = checkpoint_tensor
        if tensor.name.endswith(".attn.wo_a.weight") and not fp8_woa:
            tensor = FullLoopCheckpointTensor(tensor.name, "BF16", tensor.shape)
        elif tensor.name.startswith("layers.") and tensor.name.endswith("norm.weight"):
            tensor = FullLoopCheckpointTensor(tensor.name, "F32", tensor.shape)
        offset = (offset + alignment - 1) & -alignment
        layout.append(FullLoopArenaTensor(tensor, offset))
        offset += tensor.nbytes
    return tuple(layout)


def target_flashinfer_execution_arena_layout(
    spec: FullLoopSpec | None = None,
    *,
    alignment: int = 256,
    fp8_woa: bool = False,
) -> tuple[FullLoopArenaTensor, ...]:
    """Convenience wrapper for the retained parallel FlashInfer candidate."""

    return target_execution_arena_layout(
        spec,
        alignment=alignment,
        expert_backend="flashinfer",
        fp8_woa=fp8_woa,
    )


_EXPERT_ARENA_FIELDS = (
    "w1.weight",
    "w1.scale",
    "w2.weight",
    "w2.scale",
    "w3.weight",
    "w3.scale",
)


def target_layer_arena_fields(
    spec: FullLoopSpec | None = None,
) -> tuple[str, ...]:
    """Return the stable union of non-routed-expert fields for one block.

    The first four layers cover the union of every target ABI field through
    ratio-0/hash, ratio-4/hash, and ratio-128/score manifests.  Building the
    field list from those manifests avoids a second hand-maintained copy of the
    checkpoint ABI while retaining deterministic CUDA descriptor offsets.
    """

    spec = spec or FullLoopSpec()
    spec.require_supported()
    fields: list[str] = []
    seen: set[str] = set()
    for layer_id in range(4):
        prefix = f"layers.{layer_id}."
        for tensor in expected_target_layer_weights(layer_id, spec):
            field = tensor.name.removeprefix(prefix)
            if field.startswith("ffn.experts.") or field in seen:
                continue
            fields.append(field)
            seen.add(field)
    return tuple(fields)


def target_layer_arena_metadata(
    spec: FullLoopSpec | None = None,
    *,
    alignment: int = 256,
) -> tuple[TargetLayerArenaMetadata, ...]:
    """Bind all 43 target blocks to relocatable raw-arena byte offsets.

    This is the host-side bridge between the checkpoint manifest and a real
    persistent kernel.  It deliberately contains no tensors or Python model
    objects, so the same descriptor can be copied once to the GPU and indexed
    by a resident CTA scheduler without a per-layer host launch or host sync.
    """

    spec = spec or FullLoopSpec()
    spec.require_supported()
    layout = target_execution_arena_layout(spec, alignment=alignment)
    offsets = {entry.tensor.name: entry.offset for entry in layout}
    fixed_fields = target_layer_arena_fields(spec)
    metadata = []
    for layer_id in range(spec.target_layers):
        prefix = f"layers.{layer_id}."
        expected_names = {tensor.name for tensor in expected_target_layer_weights(layer_id, spec)}
        fixed_offsets = tuple(offsets.get(prefix + field, -1) for field in fixed_fields)
        expert_offsets = tuple(
            tuple(
                offsets[f"{prefix}ffn.experts.{expert}.{field}"] for field in _EXPERT_ARENA_FIELDS
            )
            for expert in range(spec.routed_experts)
        )
        bound_names = {
            prefix + field for field, offset in zip(fixed_fields, fixed_offsets) if offset >= 0
        }
        bound_names.update(
            f"{prefix}ffn.experts.{expert}.{field}"
            for expert in range(spec.routed_experts)
            for field in _EXPERT_ARENA_FIELDS
        )
        missing = expected_names - bound_names
        unexpected = bound_names - expected_names
        if missing or unexpected:
            detail = sorted(missing or unexpected)[0]
            raise ValueError(f"target layer {layer_id} arena binding mismatch at {detail}")
        metadata.append(
            TargetLayerArenaMetadata(
                layer_id=layer_id,
                compress_ratio=target_layer_compress_ratio(layer_id),
                fixed_fields=fixed_fields,
                fixed_offsets=fixed_offsets,
                expert_offsets=expert_offsets,
            )
        )
    return tuple(metadata)


def target_flashinfer_layer_arena_metadata(
    spec: FullLoopSpec | None = None,
    *,
    alignment: int = 256,
) -> tuple[TargetFlashInferLayerArenaMetadata, ...]:
    """Bind all target blocks to their final TRT-LLM expert tensor offsets."""

    spec = spec or FullLoopSpec()
    spec.require_supported()
    layout = target_flashinfer_execution_arena_layout(spec, alignment=alignment)
    offsets = {entry.tensor.name: entry.offset for entry in layout}
    fixed_fields = target_layer_arena_fields(spec)
    metadata = []
    for layer_id in range(spec.target_layers):
        prefix = f"layers.{layer_id}."
        fixed_offsets = tuple(offsets.get(prefix + field, -1) for field in fixed_fields)
        expert_offsets = tuple(offsets[prefix + field] for field in _FLASHINFER_EXPERT_FIELDS)
        metadata.append(
            TargetFlashInferLayerArenaMetadata(
                layer_id=layer_id,
                compress_ratio=target_layer_compress_ratio(layer_id),
                fixed_fields=fixed_fields,
                fixed_offsets=fixed_offsets,
                gemm1_weight_offset=expert_offsets[0],
                gemm1_scale_offset=expert_offsets[1],
                gemm2_weight_offset=expert_offsets[2],
                gemm2_scale_offset=expert_offsets[3],
            )
        )
    return tuple(metadata)


def target_layers_weight_map(index: Mapping[str, Any]) -> dict[str, str]:
    """Select and validate every tensor required by the 43 target blocks."""

    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, Mapping):
        raise ValueError("model.safetensors.index.json has no weight_map object")
    selected: dict[str, str] = {}
    for tensor in expected_target_globals() + expected_target_layers_weights():
        shard = raw_weight_map.get(tensor.name)
        if not isinstance(shard, str):
            raise ValueError(f"checkpoint is missing target tensor {tensor.name}")
        selected[tensor.name] = shard
    return selected


def layer42_bridge_weight_map(index: Mapping[str, Any]) -> dict[str, str]:
    """Select and validate the exact layer-42-to-head checkpoint bridge."""

    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, Mapping):
        raise ValueError("model.safetensors.index.json has no weight_map object")
    expected = expected_target_globals() + expected_target_layer42_weights()
    selected: dict[str, str] = {}
    for tensor in expected:
        shard = raw_weight_map.get(tensor.name)
        if not isinstance(shard, str):
            raise ValueError(f"checkpoint is missing bridge tensor {tensor.name}")
        selected[tensor.name] = shard
    return selected


def required_bridge_shards(index: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(set(layer42_bridge_weight_map(index).values())))
