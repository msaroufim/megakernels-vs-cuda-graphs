from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class RoutedMoeLowering:
    """Frozen physical program selected for the routed MoE kernel.

    CUDA consumes only these constants.  The compiler owns the correspondence
    between logical route/output tiles and persistent CTA claims, which keeps
    scheduling policy out of the numerical kernel.
    """

    w13_claim: int = 8
    warp_reduce: bool = True
    route_rows: int = 6 * 6
    gated_width: int = 2 * 2048
    output_tile: int = 128

    def __post_init__(self) -> None:
        if self.w13_claim not in (2, 8):
            raise ValueError(
                "W13 claim must be 2 or 8 program items; claim 4 has an "
                "invalid routed-pipeline barrier lifecycle"
            )
        if self.route_rows <= 0:
            raise ValueError("route rows must be positive")
        if self.gated_width <= 0 or self.gated_width % self.output_tile:
            raise ValueError("gated width must be tiled exactly")
        if self.program_items % self.w13_claim:
            raise ValueError("W13 claim must divide the routed program")

    @property
    def program_items(self) -> int:
        """Individual 128-output W1/W3 tiles in the logical program."""

        return self.route_rows * (self.gated_width // self.output_tile)

    @property
    def persistent_tasks(self) -> int:
        """Compiler-emitted CTA claims over the tiled W1/W3 program."""

        return self.program_items // self.w13_claim


def compile_routed_moe_lowering(
    *,
    w13_claim: int = 8,
    warp_reduce: bool = True,
) -> RoutedMoeLowering:
    """Lower semantic routed-MoE work into the measured GB300 program."""

    return RoutedMoeLowering(
        w13_claim=w13_claim,
        warp_reduce=warp_reduce,
    )


def matrix_a_row_permutation(rows: int, *, epilogue_tile_m: int = 128) -> torch.Tensor:
    """Compile a logical row index into TRT-LLM's Matrix-A MMA row order.

    The result follows the advanced-indexing convention: ``packed = raw[perm]``.
    Keeping this load-time transform in Python leaves the persistent CUDA body
    with one immutable physical layout and no model-format branches.
    """

    block = 32 if epilogue_tile_m % 128 == 0 else 16
    if rows <= 0 or rows % block:
        raise ValueError(f"rows must be a positive multiple of {block}")
    old = torch.arange(rows, dtype=torch.long)
    if block == 32:
        destination_in_block = old.remainder(32).remainder(4) * 8 + torch.div(
            old.remainder(32), 4, rounding_mode="floor"
        )
    else:
        destination_in_block = old.remainder(16).remainder(8) * 2 + torch.div(
            old.remainder(16), 8, rounding_mode="floor"
        )
    destination = torch.div(old, block, rounding_mode="floor") * block
    destination += destination_in_block
    permutation = torch.empty_like(old)
    permutation[destination] = old
    return permutation


def gated_matrix_a_row_permutation(rows: int, *, epilogue_tile_m: int = 128) -> torch.Tensor:
    """Compose contiguous ``[up, gate]`` halves with the gated MMA shuffle."""

    if rows <= 0 or rows % 2:
        raise ValueError("gated rows must be positive and even")
    half = rows // 2
    gated = torch.empty(rows, dtype=torch.long)
    logical = torch.arange(half, dtype=torch.long)
    gated[0::2] = logical
    gated[1::2] = logical + half
    return gated[matrix_a_row_permutation(rows, epilogue_tile_m=epilogue_tile_m)]


def pack_topk_ids_and_weights(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Compile top-k ids and BF16 route weights into the routed-MoE ABI."""

    if topk_ids.shape != topk_weights.shape or topk_ids.ndim < 1:
        raise ValueError("top-k ids and weights must have the same non-scalar shape")
    if topk_ids.dtype != torch.int32:
        raise TypeError("top-k ids must be int32")
    if topk_weights.dtype != torch.float32:
        raise TypeError("top-k weights must be float32")
    ids = topk_ids.contiguous()
    weights = topk_weights.contiguous()
    weight_bits = weights.bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    return (ids << 16) | weight_bits


@functools.lru_cache(maxsize=1)
def _load_route_pack_extension():
    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="deepspec_dspark_route_pack_sm103_v8",
        sources=[
            str(source / "dspark_route_pack.cpp"),
            str(source / "dspark_route_pack_kernel.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"],
        extra_cuda_cflags=[
            "-O3",
            "--std=c++17",
            "-DNDEBUG",
            "-lineinfo",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        with_cuda=True,
        verbose=False,
    )


def markov_add_argmax_cuda(
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    draft_tokens: torch.Tensor,
    *,
    draft_token_column: int,
    argmax_slot: torch.Tensor,
) -> torch.Tensor:
    """Fuse one Markov base-logit add with direct greedy token publication."""

    if base_logits.shape != (1, 129280) or bias.shape != base_logits.shape:
        raise ValueError("Markov logits and bias must have shape [1, 129280]")
    if base_logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Markov base logits must be BF16 or FP32")
    if bias.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Markov bias must be BF16 or FP32")
    if draft_tokens.ndim != 2 or draft_tokens.shape[0] != 1:
        raise ValueError("draft tokens must have shape [1, gamma]")
    if draft_tokens.dtype != torch.int64:
        raise TypeError("draft tokens must be int64")
    if not 0 <= draft_token_column < draft_tokens.shape[1]:
        raise ValueError("draft token column is out of range")
    if argmax_slot.shape != (1,) or argmax_slot.dtype != torch.int64:
        raise ValueError("argmax slot must be one int64 word")
    tensors = (base_logits, bias, draft_tokens, argmax_slot)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Markov add/argmax tensors must be contiguous CUDA tensors")
    _load_route_pack_extension().markov_add_argmax(
        base_logits,
        bias,
        draft_tokens,
        argmax_slot,
        draft_token_column,
    )
    return draft_tokens[:, draft_token_column]


def pack_topk_ids_and_weights_cuda(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lower top-k results into FlashInfer's packed route ABI in one kernel."""

    if topk_ids.shape != topk_weights.shape or topk_ids.ndim < 1:
        raise ValueError("top-k ids and weights must have the same non-scalar shape")
    if not topk_ids.is_cuda or not topk_weights.is_cuda:
        raise ValueError("fused route packing requires CUDA tensors")
    if topk_ids.dtype != torch.int32:
        raise TypeError("top-k ids must be int32")
    if topk_weights.dtype != torch.float32:
        raise TypeError("top-k weights must be float32")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("top-k inputs must be contiguous")
    if output is None:
        output = torch.empty_like(topk_ids)
    if output.shape != topk_ids.shape or output.dtype != torch.int32:
        raise ValueError("packed route output must be int32 with the input shape")
    if not output.is_cuda or not output.is_contiguous():
        raise ValueError("packed route output must be contiguous CUDA storage")
    _load_route_pack_extension().pack(topk_ids, topk_weights, output)
    return output


def target_tap_mean_concat_cuda(
    taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Lower three semantic stream means and their concat into one leaf."""

    if any(tap.shape != (6, 4, 4096) for tap in taps):
        raise ValueError("target taps must have shape [6, 4, 4096]")
    if output.shape != (6, 3 * 4096):
        raise ValueError("target hidden output must have shape [6, 12288]")
    if any(tap.dtype != torch.bfloat16 or not tap.is_cuda for tap in taps):
        raise ValueError("target taps must be CUDA BF16 tensors")
    if output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("target hidden output must be a CUDA BF16 tensor")
    if any(not tap.is_contiguous() for tap in (*taps, output)):
        raise ValueError("target tap tensors must be contiguous")
    _load_route_pack_extension().target_tap_mean_concat(*taps, output)
    return output


def target_hc_head_rmsnorm_cuda(
    streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    output: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Lower the fixed six-row HC head and final RMSNorm into one sm_103 leaf."""

    rows = streams.shape[0]
    if not 1 <= rows <= 6 or streams.shape != (rows, 4, 4096):
        raise ValueError("HC-head streams must be BF16 [1..6, 4, 4096]")
    if streams.dtype != torch.bfloat16:
        raise ValueError("HC-head streams must be BF16")
    if hc_fn.shape != (4, 4 * 4096) or hc_fn.dtype != torch.float32:
        raise ValueError("target HC coefficients must be FP32 [4, 16384]")
    if hc_scale.shape != (1,) or hc_base.shape != (4,):
        raise ValueError("target HC scale/base shapes are invalid")
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise TypeError("target HC scale/base tensors must be FP32")
    if norm_weight.shape != (4096,) or norm_weight.dtype != torch.bfloat16:
        raise ValueError("target norm weight must be BF16 [4096]")
    if output.shape != (rows, 4096) or output.dtype != torch.bfloat16:
        raise ValueError(f"HC-head output must be BF16 [{rows}, 4096]")
    tensors = (streams, hc_fn, hc_scale, hc_base, norm_weight, output)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("target HC-head tensors must be contiguous CUDA tensors")
    _load_route_pack_extension().target_hc_head_rmsnorm(
        streams,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        output,
        norm_eps,
        hc_eps,
    )
    return output


def q_lora_rmsnorm_quant_cuda(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower fixed target Q-LoRA RMSNorm plus UE8M0 quantization."""

    if input.shape != (6, 1024) or input.dtype != torch.bfloat16:
        raise ValueError("Q-LoRA input must be BF16 [6, 1024]")
    if norm_weight.shape != (1024,) or norm_weight.dtype != torch.bfloat16:
        raise ValueError("Q-LoRA norm weight must be BF16 [1024]")
    if quantized.shape != input.shape or quantized.dtype != torch.uint8:
        raise ValueError("Q-LoRA quantized output must be uint8 [6, 1024]")
    if scales.shape != (6, 2) or scales.dtype != torch.int32:
        raise ValueError("Q-LoRA scales must use packed int32 shape [6, 2]")
    if scales.stride() != (1, 8):
        raise ValueError("Q-LoRA scales must use the TMA-aligned column-major ABI")
    tensors = (input, norm_weight, quantized, scales)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Q-LoRA tensors must be CUDA")
    if not input.is_contiguous() or not norm_weight.is_contiguous():
        raise ValueError("Q-LoRA input and norm weight must be contiguous")
    if not quantized.is_contiguous():
        raise ValueError("Q-LoRA quantized output must be contiguous")
    _load_route_pack_extension().q_lora_rmsnorm_quant(
        input,
        norm_weight,
        quantized,
        scales,
        norm_eps,
    )
    return quantized, scales


def embedding_hc_expand_cuda(
    token_ids: torch.Tensor,
    embedding: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Gather six checkpoint rows and broadcast each into four HC streams."""

    if token_ids.shape != (6,) or token_ids.dtype != torch.int64:
        raise ValueError("embedding expansion requires six int64 token ids")
    if embedding.ndim != 2 or embedding.shape[1] != 4096:
        raise ValueError("embedding table must have width 4096")
    if output.shape != (6, 4, 4096):
        raise ValueError("embedding output must have shape [6, 4, 4096]")
    if embedding.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise TypeError("embedding expansion data must be BF16")
    if any(not tensor.is_cuda for tensor in (token_ids, embedding, output)):
        raise ValueError("embedding expansion tensors must be CUDA")
    if any(not tensor.is_contiguous() for tensor in (token_ids, embedding, output)):
        raise ValueError("embedding expansion tensors must be contiguous")
    _load_route_pack_extension().embedding_hc_expand(token_ids, embedding, output)
    return output


def prepare_graft_inputs_cuda(
    bonus: torch.Tensor,
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    *,
    mask_token_id: int,
) -> None:
    """Fill the fixed graft block and forward the target bonus in one leaf."""

    if bonus.numel() < 1 or input_ids.numel() != 5 or candidates.numel() != 6:
        raise ValueError(
            "graft input tensors have invalid shapes: "
            f"bonus={tuple(bonus.shape)}, input_ids={tuple(input_ids.shape)}, "
            f"candidates={tuple(candidates.shape)}"
        )
    if any(tensor.dtype != torch.int64 for tensor in (bonus, input_ids, candidates)):
        raise TypeError("graft input tensors must be int64")
    if any(not tensor.is_cuda for tensor in (bonus, input_ids, candidates)):
        raise ValueError("graft input tensors must be CUDA")
    if any(not tensor.is_contiguous() for tensor in (bonus, input_ids, candidates)):
        raise ValueError("graft input tensors must be contiguous")
    _load_route_pack_extension().prepare_graft_inputs(
        bonus,
        input_ids,
        candidates,
        int(mask_token_id),
    )


def prepare_target_inject_layout_cuda(
    commit_len: torch.Tensor,
    prefix_len: torch.Tensor,
    *,
    swa_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Build SGLang's batch-one committed-row injection layout on device."""

    if commit_len.numel() != 1 or prefix_len.numel() != 1:
        raise ValueError("target injection lengths must be scalar tensors")
    if swa_loc.numel() != 6 or positions.numel() != 6:
        raise ValueError("target injection layout requires six verify rows")
    if commit_len.dtype != torch.int32 or swa_loc.dtype != torch.int32:
        raise TypeError("commit length and SWA locations must be int32")
    if prefix_len.dtype != torch.int64 or positions.dtype != torch.int64:
        raise TypeError("prefix length and positions must be int64")
    tensors = (commit_len, prefix_len, swa_loc, positions)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("target injection layout tensors must be CUDA")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("target injection layout tensors must be contiguous")
    _load_route_pack_extension().prepare_target_inject_layout(
        commit_len,
        prefix_len,
        swa_loc,
        positions,
    )


def fused_moe_hc_post_cuda(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fuse BF16 routed/shared combine into the target MHC post leaf."""

    if any(not tensor.is_cuda for tensor in (routed, shared, residual, post, comb, output)):
        raise ValueError("fused MoE MHC post requires CUDA tensors")
    if routed.shape != (6, 4096) or shared.shape != routed.shape:
        raise ValueError("routed and shared outputs must have shape (6, 4096)")
    if residual.shape != (6, 4, 4096) or output.shape != residual.shape:
        raise ValueError("MHC streams must have shape (6, 4, 4096)")
    if post.numel() != 24 or comb.shape != (6, 4, 4):
        raise ValueError("MHC post/comb tensors have the wrong shape")
    if any(tensor.dtype != torch.bfloat16 for tensor in (routed, shared, residual, output)):
        raise TypeError("MoE and stream tensors must be BF16")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise TypeError("MHC coefficients must be FP32")
    if any(not tensor.is_contiguous() for tensor in (routed, shared, residual, post, comb, output)):
        raise ValueError("fused MoE MHC post tensors must be contiguous")
    _load_route_pack_extension().moe_hc_post(
        routed,
        shared,
        residual,
        post,
        comb,
        output,
    )
    return output


def expert_hc_post_cuda(
    expert_output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fuse a complete routed-plus-shared expert result into draft MHC post."""

    rows = int(expert_output.shape[0]) if expert_output.ndim == 2 else -1
    tensors = (expert_output, residual, post, comb, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("expert MHC post requires CUDA tensors")
    if expert_output.shape != (rows, 4096) or not 1 <= rows <= 6:
        raise ValueError("expert output must have shape [1..6, 4096]")
    if residual.shape != (rows, 4, 4096) or output.shape != residual.shape:
        raise ValueError("MHC streams must have shape [rows, 4, 4096]")
    if post.shape != (rows, 4) or comb.shape != (rows, 4, 4):
        raise ValueError("MHC coefficients have the wrong shape")
    if any(tensor.dtype != torch.bfloat16 for tensor in (expert_output, residual, output)):
        raise TypeError("expert and stream tensors must be BF16")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise TypeError("MHC coefficients must be FP32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("expert MHC post tensors must be contiguous")
    _load_route_pack_extension().expert_hc_post(
        expert_output,
        residual,
        post,
        comb,
        output,
    )
    return output


def fused_moe_finalize_hc_post_cuda(
    gemm2: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_to_permuted: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fuse the unfinalized FlashInfer routed result into target HC post."""

    tensors = (
        gemm2,
        expert_weights,
        expanded_to_permuted,
        shared,
        residual,
        post,
        comb,
        output,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused MoE finalize/HC post requires CUDA tensors")
    if gemm2.ndim != 2 or gemm2.shape[1] < 4096:
        raise ValueError("gemm2 output must have shape [padded_tokens, >=4096]")
    if expert_weights.shape != (6, 6) or expanded_to_permuted.numel() != 36:
        raise ValueError("fused finalize requires batch6/topk6 routing metadata")
    if shared.shape != (6, 4096):
        raise ValueError("shared output must have shape (6, 4096)")
    if residual.shape != (6, 4, 4096) or output.shape != residual.shape:
        raise ValueError("MHC streams must have shape (6, 4, 4096)")
    if post.numel() != 24 or comb.shape != (6, 4, 4):
        raise ValueError("MHC post/comb tensors have the wrong shape")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (gemm2, expert_weights, shared, residual, output)
    ):
        raise TypeError("MoE data and stream tensors must be BF16")
    if expanded_to_permuted.dtype != torch.int32:
        raise TypeError("expanded-to-permuted map must be int32")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise TypeError("MHC coefficients must be FP32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused MoE finalize/HC post tensors must be contiguous")
    _load_route_pack_extension().moe_finalize_hc_post(
        gemm2,
        expert_weights,
        expanded_to_permuted,
        shared,
        residual,
        post,
        comb,
        output,
    )
    return output
