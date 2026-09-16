from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import prod

import torch

from deepspec.megakernel.contract import DeepSeekV4MegaKernelSpec


class V4GlobalWeightSlot(IntEnum):
    EMBEDDING = 0
    LM_HEAD = 1


class V4LayerWeightSlot(IntEnum):
    HC_ATTN_FN = 0
    HC_ATTN_BASE = 1
    HC_ATTN_SCALE = 2
    HC_FFN_FN = 3
    HC_FFN_BASE = 4
    HC_FFN_SCALE = 5
    ATTN_NORM = 6
    FFN_NORM = 7
    ATTN_SINK = 8
    WQ_A = 9
    WQ_A_SCALE = 10
    Q_NORM = 11
    WQ_B = 12
    WQ_B_SCALE = 13
    WKV = 14
    WKV_SCALE = 15
    KV_NORM = 16
    WO_A = 17
    WO_B = 18
    WO_B_SCALE = 19
    GATE = 20
    GATE_BIAS = 21
    MAIN_PROJ = 22
    MAIN_PROJ_SCALE = 23
    MAIN_NORM = 24
    FINAL_NORM = 25
    HC_HEAD_FN = 26
    HC_HEAD_BASE = 27
    HC_HEAD_SCALE = 28
    MARKOV_W1 = 29
    MARKOV_W2 = 30
    CONFIDENCE = 31


class V4ExpertWeightSlot(IntEnum):
    W1 = 0
    W1_SCALE = 1
    W2 = 2
    W2_SCALE = 3
    W3 = 4
    W3_SCALE = 5


_DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "float8_e8m0fnu": 1,
    "float4_e2m1fn_x2": 1,
}


@dataclass(frozen=True)
class V4ExpectedWeight:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_index: int

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class V4WeightEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_index: int
    arena_offset: int
    nbytes: int


@dataclass(frozen=True)
class V4WeightTableLayout:
    global_base: int
    layer_base: int
    routed_expert_base: int
    shared_expert_base: int
    total_slots: int

    @classmethod
    def flash(cls, spec: DeepSeekV4MegaKernelSpec) -> V4WeightTableLayout:
        global_base = 0
        layer_base = global_base + len(V4GlobalWeightSlot)
        routed_expert_base = layer_base + spec.num_draft_layers * len(V4LayerWeightSlot)
        shared_expert_base = (
            routed_expert_base
            + spec.num_draft_layers * spec.num_routed_experts * len(V4ExpertWeightSlot)
        )
        total_slots = shared_expert_base + spec.num_draft_layers * len(V4ExpertWeightSlot)
        return cls(
            global_base,
            layer_base,
            routed_expert_base,
            shared_expert_base,
            total_slots,
        )

    def global_slot(self, slot: V4GlobalWeightSlot) -> int:
        return self.global_base + int(slot)

    def layer_slot(self, layer: int, slot: V4LayerWeightSlot) -> int:
        return self.layer_base + layer * len(V4LayerWeightSlot) + int(slot)

    def routed_expert_slot(
        self,
        layer: int,
        expert: int,
        slot: V4ExpertWeightSlot,
        *,
        experts: int,
    ) -> int:
        return (
            self.routed_expert_base
            + (layer * experts + expert) * len(V4ExpertWeightSlot)
            + int(slot)
        )

    def shared_expert_slot(self, layer: int, slot: V4ExpertWeightSlot) -> int:
        return self.shared_expert_base + layer * len(V4ExpertWeightSlot) + int(slot)


@dataclass(frozen=True)
class V4WeightArenaPlan:
    entries: tuple[V4WeightEntry, ...]
    offset_table: tuple[int, ...]
    arena_bytes: int
    alignment: int


@dataclass(frozen=True)
class V4PackedWeights:
    arena: torch.Tensor
    offsets: torch.Tensor
    entries: tuple[V4WeightEntry, ...]
    alignment: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def expected_v4_weights(
    spec: DeepSeekV4MegaKernelSpec | None = None,
) -> tuple[V4ExpectedWeight, ...]:
    """Return the 4,704 post-conversion runtime parameters in semantic order."""

    spec = spec or DeepSeekV4MegaKernelSpec()
    spec.require_supported()
    h = spec.hidden_size
    v = spec.vocab_size
    hc = spec.hc_multiplier
    mix = (2 + hc) * hc
    mhc = spec.mhc_width
    q_rank = spec.q_lora_rank
    q_width = spec.query_width
    d = spec.head_dim
    groups = spec.output_groups
    o_rank = spec.output_lora_rank
    expert_width = spec.expert_intermediate_size
    layout = V4WeightTableLayout.flash(spec)
    expected: list[V4ExpectedWeight] = []

    def add(name: str, dtype: str, shape: tuple[int, ...], offset_index: int) -> None:
        expected.append(V4ExpectedWeight(name, dtype, shape, offset_index))

    add(
        "embed.weight",
        "bfloat16",
        (v, h),
        layout.global_slot(V4GlobalWeightSlot.EMBEDDING),
    )
    add(
        "head.weight",
        "float32",
        (v, h),
        layout.global_slot(V4GlobalWeightSlot.LM_HEAD),
    )
    for layer in range(spec.num_draft_layers):
        prefix = f"mtp.{layer}"
        layer_specs = (
            ("hc_attn_fn", "float32", (mix, mhc), V4LayerWeightSlot.HC_ATTN_FN),
            ("hc_attn_base", "float32", (mix,), V4LayerWeightSlot.HC_ATTN_BASE),
            ("hc_attn_scale", "float32", (3,), V4LayerWeightSlot.HC_ATTN_SCALE),
            ("hc_ffn_fn", "float32", (mix, mhc), V4LayerWeightSlot.HC_FFN_FN),
            ("hc_ffn_base", "float32", (mix,), V4LayerWeightSlot.HC_FFN_BASE),
            ("hc_ffn_scale", "float32", (3,), V4LayerWeightSlot.HC_FFN_SCALE),
            ("attn_norm.weight", "float32", (h,), V4LayerWeightSlot.ATTN_NORM),
            ("ffn_norm.weight", "float32", (h,), V4LayerWeightSlot.FFN_NORM),
            (
                "attn.attn_sink",
                "float32",
                (spec.num_attention_heads,),
                V4LayerWeightSlot.ATTN_SINK,
            ),
            ("attn.wq_a.weight", "float8_e4m3fn", (q_rank, h), V4LayerWeightSlot.WQ_A),
            (
                "attn.wq_a.scale",
                "float8_e8m0fnu",
                (q_rank // 128, h // 128),
                V4LayerWeightSlot.WQ_A_SCALE,
            ),
            ("attn.q_norm.weight", "float32", (q_rank,), V4LayerWeightSlot.Q_NORM),
            (
                "attn.wq_b.weight",
                "float8_e4m3fn",
                (q_width, q_rank),
                V4LayerWeightSlot.WQ_B,
            ),
            (
                "attn.wq_b.scale",
                "float8_e8m0fnu",
                (q_width // 128, q_rank // 128),
                V4LayerWeightSlot.WQ_B_SCALE,
            ),
            ("attn.wkv.weight", "float8_e4m3fn", (d, h), V4LayerWeightSlot.WKV),
            (
                "attn.wkv.scale",
                "float8_e8m0fnu",
                (d // 128, h // 128),
                V4LayerWeightSlot.WKV_SCALE,
            ),
            ("attn.kv_norm.weight", "float32", (d,), V4LayerWeightSlot.KV_NORM),
            (
                "attn.wo_a.weight",
                "bfloat16",
                (groups * o_rank, q_width // groups),
                V4LayerWeightSlot.WO_A,
            ),
            (
                "attn.wo_b.weight",
                "float8_e4m3fn",
                (h, groups * o_rank),
                V4LayerWeightSlot.WO_B,
            ),
            (
                "attn.wo_b.scale",
                "float8_e8m0fnu",
                (h // 128, groups * o_rank // 128),
                V4LayerWeightSlot.WO_B_SCALE,
            ),
            ("ffn.gate.weight", "bfloat16", (spec.num_routed_experts, h), V4LayerWeightSlot.GATE),
            (
                "ffn.gate.bias",
                "float32",
                (spec.num_routed_experts,),
                V4LayerWeightSlot.GATE_BIAS,
            ),
        )
        for suffix, dtype, tensor_shape, slot in layer_specs:
            add(
                f"{prefix}.{suffix}",
                dtype,
                tensor_shape,
                layout.layer_slot(layer, slot),
            )

        routed_shapes = {
            V4ExpertWeightSlot.W1: (expert_width, h // 2),
            V4ExpertWeightSlot.W1_SCALE: (expert_width, h // 32),
            V4ExpertWeightSlot.W2: (h, expert_width // 2),
            V4ExpertWeightSlot.W2_SCALE: (h, expert_width // 32),
            V4ExpertWeightSlot.W3: (expert_width, h // 2),
            V4ExpertWeightSlot.W3_SCALE: (expert_width, h // 32),
        }
        for expert in range(spec.num_routed_experts):
            for slot in V4ExpertWeightSlot:
                suffix = slot.name.lower().replace("_scale", ".scale")
                if ".scale" not in suffix:
                    suffix += ".weight"
                dtype = "float8_e8m0fnu" if slot.name.endswith("SCALE") else "float4_e2m1fn_x2"
                add(
                    f"{prefix}.ffn.experts.{expert}.{suffix}",
                    dtype,
                    routed_shapes[slot],
                    layout.routed_expert_slot(
                        layer,
                        expert,
                        slot,
                        experts=spec.num_routed_experts,
                    ),
                )

        shared_shapes = {
            V4ExpertWeightSlot.W1: (expert_width, h),
            V4ExpertWeightSlot.W1_SCALE: (expert_width // 128, h // 128),
            V4ExpertWeightSlot.W2: (h, expert_width),
            V4ExpertWeightSlot.W2_SCALE: (h // 128, expert_width // 128),
            V4ExpertWeightSlot.W3: (expert_width, h),
            V4ExpertWeightSlot.W3_SCALE: (expert_width // 128, h // 128),
        }
        for slot in V4ExpertWeightSlot:
            suffix = slot.name.lower().replace("_scale", ".scale")
            if ".scale" not in suffix:
                suffix += ".weight"
            dtype = "float8_e8m0fnu" if slot.name.endswith("SCALE") else "float8_e4m3fn"
            add(
                f"{prefix}.ffn.shared_experts.{suffix}",
                dtype,
                shared_shapes[slot],
                layout.shared_expert_slot(layer, slot),
            )

        if layer == 0:
            add(
                f"{prefix}.main_proj.weight",
                "float8_e4m3fn",
                (h, spec.target_feature_width),
                layout.layer_slot(layer, V4LayerWeightSlot.MAIN_PROJ),
            )
            add(
                f"{prefix}.main_proj.scale",
                "float8_e8m0fnu",
                (h // 128, spec.target_feature_width // 128),
                layout.layer_slot(layer, V4LayerWeightSlot.MAIN_PROJ_SCALE),
            )
            add(
                f"{prefix}.main_norm.weight",
                "float32",
                (h,),
                layout.layer_slot(layer, V4LayerWeightSlot.MAIN_NORM),
            )
        if layer == spec.num_draft_layers - 1:
            tail_specs = (
                ("norm.weight", "float32", (h,), V4LayerWeightSlot.FINAL_NORM),
                ("hc_head_fn", "float32", (hc, mhc), V4LayerWeightSlot.HC_HEAD_FN),
                ("hc_head_base", "float32", (hc,), V4LayerWeightSlot.HC_HEAD_BASE),
                ("hc_head_scale", "float32", (1,), V4LayerWeightSlot.HC_HEAD_SCALE),
                (
                    "markov_head.markov_w1.weight",
                    "bfloat16",
                    (v, spec.markov_rank),
                    V4LayerWeightSlot.MARKOV_W1,
                ),
                (
                    "markov_head.markov_w2.weight",
                    "float32",
                    (v, spec.markov_rank),
                    V4LayerWeightSlot.MARKOV_W2,
                ),
                (
                    "confidence_head.proj.weight",
                    "float32",
                    (1, h + spec.markov_rank),
                    V4LayerWeightSlot.CONFIDENCE,
                ),
            )
            for suffix, dtype, tensor_shape, slot in tail_specs:
                add(
                    f"{prefix}.{suffix}",
                    dtype,
                    tensor_shape,
                    layout.layer_slot(layer, slot),
                )
    return tuple(expected)


def plan_v4_weight_arena(
    spec: DeepSeekV4MegaKernelSpec | None = None,
    *,
    alignment: int = 256,
) -> V4WeightArenaPlan:
    spec = spec or DeepSeekV4MegaKernelSpec()
    expected = expected_v4_weights(spec)
    table = [-1] * V4WeightTableLayout.flash(spec).total_slots
    entries = []
    offset = 0
    for item in expected:
        offset = _align_up(offset, alignment)
        if table[item.offset_index] != -1:
            raise AssertionError(f"duplicate V4 weight offset slot {item.offset_index}")
        table[item.offset_index] = offset
        entries.append(
            V4WeightEntry(
                item.name,
                item.dtype,
                item.shape,
                item.offset_index,
                offset,
                item.nbytes,
            )
        )
        offset += item.nbytes
    return V4WeightArenaPlan(
        entries=tuple(entries),
        offset_table=tuple(table),
        arena_bytes=_align_up(offset, alignment),
        alignment=alignment,
    )


@torch.inference_mode()
def pack_v4_weights(
    proposal: torch.nn.Module,
    spec: DeepSeekV4MegaKernelSpec | None = None,
    *,
    alignment: int = 256,
) -> V4PackedWeights:
    """Pack official post-conversion parameters once into the device arena."""

    spec = spec or DeepSeekV4MegaKernelSpec()
    plan = plan_v4_weight_arena(spec, alignment=alignment)
    parameters = dict(proposal.named_parameters())
    expected_names = {entry.name for entry in plan.entries}
    missing = sorted(expected_names - set(parameters))
    unexpected = sorted(set(parameters) - expected_names)
    if missing or unexpected:
        raise ValueError(
            f"V4 parameter mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    devices = {parameter.device for parameter in parameters.values()}
    if len(devices) != 1:
        raise ValueError(f"V4 parameters must share one device, got {devices}")
    device = next(iter(devices))
    arena = torch.empty(plan.arena_bytes, dtype=torch.uint8, device=device)
    for entry in plan.entries:
        parameter = parameters[entry.name]
        dtype = str(parameter.dtype).removeprefix("torch.")
        if dtype != entry.dtype or tuple(parameter.shape) != entry.shape:
            raise ValueError(
                f"{entry.name} is {dtype}{tuple(parameter.shape)}, "
                f"expected {entry.dtype}{entry.shape}"
            )
        raw = parameter.detach().contiguous().view(torch.uint8).flatten()
        if raw.numel() != entry.nbytes:
            raise ValueError(f"{entry.name} exposes {raw.numel()} bytes, expected {entry.nbytes}")
        arena.narrow(0, entry.arena_offset, entry.nbytes).copy_(raw)
    offsets = torch.tensor(plan.offset_table, dtype=torch.int64, device=device)
    return V4PackedWeights(arena, offsets, plan.entries, alignment)
