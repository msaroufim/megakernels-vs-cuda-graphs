#!/usr/bin/env python3
"""Module-free V4 drafter weight packer for foreign images (SGLang bolt-in).

Replicates ``load_official_v4_flash_weights`` + ``pack_v4_weights``
byte-for-byte WITHOUT instantiating the official reference model: the pinned
snapshot's ``inference/kernel.py`` imports TileLang, which does not install on
the SGLang image's python 3.12, so ``build_official_v4_flash_proposal`` is
unavailable there. Instead this packer walks the megakernel weight manifest
(``expected_v4_weights``/``plan_v4_weight_arena``) and reads each tensor by
name straight from the five drafter checkpoint shards, applying the exact
transforms the official loader applies:

  - ``*.wo_a.weight``: FP8 checkpoint + ``*.wo_a.scale`` -> ``_dequantize_wo_a``
    (the loader's own function, imported — not reimplemented) -> BF16.
  - routed-expert FP4 weights: packed int8 storage, bytes verbatim.
  - manifest float32 entries stored BF16 in the checkpoint: widened via
    ``.to(float32)`` (exact), mirroring the loader's ``.to(target.dtype)``.
  - everything else: ``.to(manifest dtype)`` (identity for fp8/e8m0/bf16).

The RoPE table is rebuilt by lifting ``precompute_freqs_cis`` out of the
snapshot's ``inference/model.py`` source via ast (never importing the module),
with the DSpark-block parameterization: mtp layers have compress_ratio 0, so
YaRN is disabled (original_seq_len=0) and base rope_theta applies.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from deepspec.megakernel.v4_weights import (
    V4WeightArenaPlan,
    plan_v4_weight_arena,
)
from deepspec.modeling.dspark.deepseek_v4.official import (
    OFFICIAL_V4_FLASH_SHARDS,
    _dequantize_wo_a,
    required_checkpoint_shards,
    selected_weight_map,
)

_ALIGN = 256

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e8m0fnu": torch.float8_e8m0fnu,
    "float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
}


def _aligned(value: int) -> int:
    return (value + _ALIGN - 1) // _ALIGN * _ALIGN


def lift_precompute_freqs_cis(model_py: Path):
    """Extract precompute_freqs_cis from the snapshot source without import."""

    tree = ast.parse(model_py.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "precompute_freqs_cis":
            namespace: dict = {"torch": torch, "math": math, "lru_cache": lru_cache}
            code = ast.Module(body=[node], type_ignores=[])
            exec(compile(code, str(model_py), "exec"), namespace)  # noqa: S102
            return namespace["precompute_freqs_cis"]
    raise RuntimeError(f"precompute_freqs_cis not found in {model_py}")


def drafter_freqs_cis(snapshot: Path, *, max_seq_len: int = 65536) -> torch.Tensor:
    """Rebuild proposal.mtp[*].attn.freqs_cis for the DSpark draft layers."""

    with (snapshot / "inference" / "config.json").open() as stream:
        config = json.load(stream)
    n_layers = int(config["n_layers"])
    compress_ratios = config.get("compress_ratios")
    if compress_ratios is not None and any(
        int(ratio) != 0 for ratio in compress_ratios[n_layers : n_layers + 3]
    ):
        raise RuntimeError(
            f"mtp layers expected compress_ratio 0, got {compress_ratios[n_layers:]}"
        )
    # compress_ratio == 0 branch of Attention.__init__: YaRN disabled,
    # base rope_theta (see the pinned inference/model.py).
    precompute = lift_precompute_freqs_cis(snapshot / "inference" / "model.py")
    return precompute(
        int(config.get("rope_head_dim", 64)),
        max_seq_len,
        0,
        float(config.get("rope_theta", 10000.0)),
        float(config.get("rope_factor", 40)),
        int(config.get("beta_fast", 32)),
        int(config.get("beta_slow", 1)),
    )


@dataclass(frozen=True)
class BoltinPackedWeights:
    arena: torch.Tensor  # uint8, relaxed layout (bf16 lm/markov appended)
    offsets: torch.Tensor  # int64, 4724 contract slots + [lm_off, mk_off]
    freqs_cis: torch.Tensor  # complex64 (max_seq_len, rope_head_dim // 2)
    plan: V4WeightArenaPlan
    lm_bf16_offset: int
    markov_w2_bf16_offset: int
    lm_bulk_staging: bool = False
    woa_bulk_staging: bool = False
    qb_block_scaled: bool = False
    dense_block_scaled: bool = False
    markov_bulk_staging: bool = False
    routed_packed_sfa: bool = False
    routed_bulk_weights: bool = False
    routed_interleave: bool = False
    shared_bulk_staging: bool = False


@dataclass(frozen=True)
class BoltinTailWeights:
    """Exact full-vocabulary BF16 tensors shared with the TP2 helper."""

    lm_head_bf16: torch.Tensor
    markov_w1_bf16: torch.Tensor
    markov_w2_bf16: torch.Tensor


def boltin_tail_weight_views(packed: BoltinPackedWeights) -> BoltinTailWeights:
    """Return zero-copy rank-0 views used as one-time rank-1 broadcast sources."""

    if packed.lm_bulk_staging:
        raise ValueError("bulk-staged LM weights do not expose a row-major tail view")
    if packed.markov_bulk_staging:
        raise ValueError("bulk-staged Markov weights do not expose a row-major tail view")
    entries = {entry.name: entry for entry in packed.plan.entries}
    lm = entries["head.weight"]
    w1_name = next(name for name in entries if name.endswith("mtp.2.markov_head.markov_w1.weight"))
    w2_name = next(name for name in entries if name.endswith("mtp.2.markov_head.markov_w2.weight"))
    w1 = entries[w1_name]
    w2 = entries[w2_name]
    expected = {
        "lm_head_bf16": (lm.shape, packed.lm_bf16_offset),
        "markov_w1_bf16": (w1.shape, w1.arena_offset),
        "markov_w2_bf16": (w2.shape, packed.markov_w2_bf16_offset),
    }

    def view(shape: tuple[int, ...], offset: int) -> torch.Tensor:
        elements = math.prod(shape)
        return packed.arena.narrow(0, offset, elements * 2).view(torch.bfloat16).view(shape)

    views = {name: view(shape, offset) for name, (shape, offset) in expected.items()}
    for name, tensor in views.items():
        if not tensor.is_contiguous() or tensor.data_ptr() % 128:
            raise RuntimeError(f"{name} is not a contiguous 128-byte-aligned arena view")
    return BoltinTailWeights(**views)


def pack_lm_bulk_staging(weight: torch.Tensor) -> torch.Tensor:
    """Pack BF16 M128/K16 operands in the scheduler's shared-memory byte order."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("LM staging requires a BF16 matrix")
    rows, columns = weight.shape
    if rows % 128 or columns % 16:
        raise ValueError("LM staging requires whole M128/K16 tiles")
    return weight.reshape(rows // 128, 128, columns // 16, 2, 8).permute(0, 2, 3, 1, 4).contiguous()


def pack_fp8_bulk_staging(weight: torch.Tensor) -> torch.Tensor:
    """Pack FP8 M128/K128 tiles as four K32 interleaved shared-memory tiles."""
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("FP8 staging requires an E4M3 matrix")
    rows, columns = weight.shape
    if rows % 128 or columns % 128:
        raise ValueError("FP8 staging requires whole M128/K128 tiles")
    return (
        weight.view(torch.uint8)
        .reshape(rows // 128, 128, columns // 128, 4, 2, 16)
        .permute(0, 2, 3, 4, 1, 5)
        .contiguous()
    )


def pack_routed_bulk_weights(weight: torch.Tensor) -> torch.Tensor:
    """Pack FP4 bytes into contiguous M128/K512 pipeline stages."""
    raw = weight.view(torch.uint8)
    if raw.ndim != 2:
        raise ValueError("routed bulk packing requires a packed FP4 matrix")
    rows, row_bytes = raw.shape
    if rows % 128 or row_bytes % 256:
        raise ValueError("routed bulk packing requires whole M128/K512 tiles")
    return (
        raw.reshape(rows // 128, 128, row_bytes // 256, 4, 64).permute(0, 2, 3, 1, 4).contiguous()
    )


def pack_routed_sfa(weight_scales: torch.Tensor) -> torch.Tensor:
    """Pack raw E8M0 bytes as M128/K4 UTCCP scale blocks, without conversion."""
    if weight_scales.dtype != torch.uint8 or weight_scales.ndim != 2:
        raise ValueError("routed SFA packing requires a uint8 matrix")
    rows, columns = weight_scales.shape
    if rows % 128 or columns % 4:
        raise ValueError("routed SFA packing requires whole M128/K4 blocks")
    return (
        weight_scales.reshape(rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
    )


@torch.inference_mode()
def pack_boltin_arena(
    snapshot: str | Path,
    *,
    device: torch.device | str = "cuda",
    max_seq_len: int = 65536,
    lm_bulk_staging: bool = False,
    woa_bulk_staging: bool = False,
    qb_block_scaled: bool = False,
    dense_block_scaled: bool = False,
    markov_bulk_staging: bool = False,
    routed_packed_sfa: bool = False,
    routed_bulk_weights: bool = False,
    routed_interleave: bool = False,
    shared_bulk_staging: bool = False,
    log=print,
) -> BoltinPackedWeights:
    snapshot = Path(snapshot)
    device = torch.device(device)
    from safetensors import safe_open

    with (snapshot / "model.safetensors.index.json").open() as stream:
        index = json.load(stream)
    weight_map = selected_weight_map(index)
    shards = required_checkpoint_shards(weight_map)
    if shards != OFFICIAL_V4_FLASH_SHARDS:
        raise ValueError(f"unexpected V4-Flash proposal shards: {shards}")

    plan = plan_v4_weight_arena()
    entries_by_name = {entry.name: entry for entry in plan.entries}
    missing = sorted(set(entries_by_name) - set(weight_map))
    if missing:
        raise ValueError(f"checkpoint lacks manifest tensors: {missing[:8]}")

    # Relaxed appendix: BF16 lm_head + markov_w2 past the frozen arena.
    lm_name = "head.weight"
    markov_name = next(
        name for name in entries_by_name if name.endswith("markov_head.markov_w2.weight")
    )
    lm_entry = entries_by_name[lm_name]
    mk_entry = entries_by_name[markov_name]
    lm_offset = _aligned(plan.arena_bytes)
    lm_bytes = lm_entry.shape[0] * lm_entry.shape[1] * 2  # bf16
    mk_offset = _aligned(lm_offset + lm_bytes)
    mk_bytes = mk_entry.shape[0] * mk_entry.shape[1] * 2  # bf16
    total_bytes = mk_offset + mk_bytes

    arena = torch.zeros(total_bytes, dtype=torch.uint8, device=device)
    packed = 0
    packed_bytes = 0
    for shard in shards:
        names = sorted(
            name
            for name, mapped in weight_map.items()
            if mapped == shard and name in entries_by_name
        )
        with safe_open(snapshot / shard, framework="pt", device="cpu") as reader:
            for name in names:
                entry = entries_by_name[name]
                target_dtype = _DTYPES[entry.dtype]
                source = reader.get_tensor(name)
                if name.endswith(".wo_a.weight"):
                    scale = reader.get_tensor(name.replace("weight", "scale"))
                    source = _dequantize_wo_a(source, scale)
                elif ".experts." in name and name.endswith(".weight"):
                    if source.dtype != torch.int8:
                        raise ValueError(
                            f"expected packed int8 storage for {name}, got {source.dtype}"
                        )
                    source = source.view(torch.float4_e2m1fn_x2)
                converted = source.to(target_dtype)
                if tuple(converted.shape) != entry.shape:
                    raise ValueError(f"{name} is {tuple(converted.shape)}, expected {entry.shape}")
                if woa_bulk_staging and name.endswith(".wo_a.weight"):
                    converted = pack_lm_bulk_staging(converted)
                if qb_block_scaled and name.endswith(".wq_b.weight"):
                    converted = pack_fp8_bulk_staging(converted)
                if dense_block_scaled and (
                    name.endswith(
                        (".wq_a.weight", ".wkv.weight", ".wo_b.weight", ".main_proj.weight")
                    )
                    or name.endswith(".shared_experts.w2.weight")
                ):
                    converted = pack_fp8_bulk_staging(converted)
                if shared_bulk_staging and name.endswith(
                    (".shared_experts.w1.weight", ".shared_experts.w3.weight")
                ):
                    converted = pack_fp8_bulk_staging(converted)
                if routed_bulk_weights and ".experts." in name and name.endswith(".weight"):
                    converted = pack_routed_bulk_weights(converted)
                if routed_packed_sfa and ".experts." in name and name.endswith(".scale"):
                    converted = pack_routed_sfa(converted.view(torch.uint8))
                raw = converted.contiguous().view(torch.uint8).flatten()
                if raw.numel() != entry.nbytes:
                    raise ValueError(f"{name} exposes {raw.numel()} bytes, expected {entry.nbytes}")
                arena.narrow(0, entry.arena_offset, entry.nbytes).copy_(raw, non_blocking=False)
                packed += 1
                packed_bytes += entry.nbytes
                if name == lm_name:
                    lm_weight = source.to(torch.bfloat16)
                    if lm_bulk_staging:
                        lm_weight = pack_lm_bulk_staging(lm_weight)
                    lm_raw = lm_weight.contiguous().view(torch.uint8)
                    arena.narrow(0, lm_offset, lm_bytes).copy_(lm_raw.flatten())
                elif name == markov_name:
                    mk_weight = source.to(torch.bfloat16)
                    if markov_bulk_staging:
                        mk_weight = pack_lm_bulk_staging(mk_weight)
                    mk_raw = mk_weight.contiguous().view(torch.uint8)
                    arena.narrow(0, mk_offset, mk_bytes).copy_(mk_raw.flatten())
        log(f"boltin_pack_shard {shard} packed={packed}/{len(plan.entries)}")

    if packed != len(plan.entries):
        raise RuntimeError(f"packed {packed} of {len(plan.entries)} manifest entries")

    offset_values = list(plan.offset_table)
    if routed_interleave:
        if not routed_packed_sfa or routed_bulk_weights:
            raise ValueError("interleaved W13 requires packed SFA and ordinary FP4 row storage")
        for entry in plan.entries:
            if ".experts." not in entry.name or not entry.name.endswith(".w1.weight"):
                continue
            prefix = entry.name.removesuffix("w1.weight")
            matrices = [
                entries_by_name[prefix + suffix]
                for suffix in (
                    "w1.weight",
                    "w1.scale",
                    "w2.weight",
                    "w2.scale",
                    "w3.weight",
                    "w3.scale",
                )
            ]
            ordered = sorted(matrices, key=lambda item: item.arena_offset)
            base = ordered[0].arena_offset
            span = sum(item.nbytes for item in matrices)
            if ordered[-1].arena_offset + ordered[-1].nbytes != base + span:
                raise ValueError(
                    "interleaved expert matrices must occupy one contiguous arena span"
                )
            raw = [arena.narrow(0, item.arena_offset, item.nbytes) for item in matrices]
            gate = raw[0].reshape(32, 64, 2048)
            up = raw[4].reshape(32, 64, 2048)
            weights = torch.stack((gate, up), dim=1).flatten()

            def unpack_sfa(value):
                return value.reshape(16, 32, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(32, 64, 128)

            scales = pack_routed_sfa(
                torch.stack((unpack_sfa(raw[1]), unpack_sfa(raw[5])), dim=1).reshape(4096, 128)
            ).flatten()
            # Repack all three matrices together so no persistent allocation grows.
            packed_expert = torch.cat((weights, scales, raw[2], raw[3]))
            arena.narrow(0, base, span).copy_(packed_expert)
            w2_offset = base + weights.numel() + scales.numel()
            offset_values[matrices[0].offset_index] = base
            offset_values[matrices[1].offset_index] = base + weights.numel()
            offset_values[matrices[2].offset_index] = w2_offset
            offset_values[matrices[3].offset_index] = w2_offset + matrices[2].nbytes
            # W3 is consumed through W1's combined map in this explicitly tagged layout.
            offset_values[matrices[4].offset_index] = base
            offset_values[matrices[5].offset_index] = base + weights.numel()
    offsets = torch.tensor(
        offset_values + [lm_offset, mk_offset],
        dtype=torch.int64,
        device=device,
    )
    freqs = drafter_freqs_cis(snapshot, max_seq_len=max_seq_len).to(device)
    log(
        f"boltin_pack_done entries={packed} contract_bytes={plan.arena_bytes} "
        f"total_bytes={total_bytes} lm_off={lm_offset} mk_off={mk_offset} "
        f"freqs={tuple(freqs.shape)} {freqs.dtype}"
    )
    return BoltinPackedWeights(
        arena=arena,
        offsets=offsets,
        freqs_cis=freqs,
        plan=plan,
        lm_bf16_offset=lm_offset,
        markov_w2_bf16_offset=mk_offset,
        lm_bulk_staging=lm_bulk_staging,
        woa_bulk_staging=woa_bulk_staging,
        qb_block_scaled=qb_block_scaled,
        dense_block_scaled=dense_block_scaled,
        markov_bulk_staging=markov_bulk_staging,
        routed_packed_sfa=routed_packed_sfa,
        routed_bulk_weights=routed_bulk_weights,
        routed_interleave=routed_interleave,
        shared_bulk_staging=shared_bulk_staging,
    )
