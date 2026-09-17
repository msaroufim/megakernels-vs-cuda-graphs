"""FP32 fused-epilogue Llama 1B candidate; numerical validation precedes tuning.

The globals interface is the pinned Hazy latency Globals with interleaved Q/K.
No model weights, prompt-dependent activations, or outputs are precomputed.
"""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time

import torch
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class CandidateOptions:
    """Explicit runtime/numerical choices, fixed before evaluating prompts."""

    pdl: bool = True
    attention: str = "sdpa"
    native_bf16_fma: bool = True
    attention_tile: int = 16
    attention_warps: int = 4


def load_attention():
    """Load the sibling Triton kernel without depending on caller sys.path."""
    path = Path(__file__).with_name("attention.py")
    spec = importlib.util.spec_from_file_location("phase3_hf_attention", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pack(weight, rows):
    """Losslessly rearrange weight rows for contiguous 16-byte CTA loads."""
    return (
        weight.view(-1, rows, weight.shape[-1] // 8, 8)
        .transpose(1, 2)
        .contiguous()
        .reshape(weight.shape)
    )


def prepare(g, *, hf_config=None, options=CandidateOptions()):
    """Compile, pack immutable weights, and create FP32 interleaved RoPE tables.

    Passing the independent HF model's config constructs its FP32 tables on
    this GPU with Transformers' LlamaRotaryEmbedding. Without config, existing
    interleaved globals tables remain FP32; the receipt distinguishes
    that fallback. This function performs no input-dependent model work.
    """
    if options.attention not in ("sdpa", "eager"):
        raise ValueError("attention must be 'sdpa' or 'eager'")
    if options.attention_tile not in (16, 32, 64):
        raise ValueError("attention_tile must evenly divide head dimension 64")
    device = g.hidden_states.device
    capability = torch.cuda.get_device_capability(device)
    if options.pdl and capability < (9, 0):
        raise ValueError("PDL requires Hopper or newer hardware")
    if options.native_bf16_fma and capability < (10, 0):
        raise ValueError("Native fma.rn.f32.bf16 requires Blackwell")
    flags = ["-O3", "-lineinfo", "--use_fast_math"]
    if options.native_bf16_fma:
        flags.append("-DUSE_NATIVE_BF16_FMA=1")
    extension = load(
        name="llama_phase3_fp32epi_headout_fastmath_" + ("native" if options.native_bf16_fma else "portable"),
        sources=[str(Path(__file__).with_name("projection.cu"))],
        extra_cuda_cflags=flags,
        verbose=True,
    )
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    before = torch.cuda.memory_allocated(device)
    weights = {}
    for name, rows in (
        ("qkv_proj_weights", 2),
        ("o_proj_weights", 2),
        ("up_proj_weights", 2),
        ("gate_proj_weights", 2),
        ("down_proj_weights", 2),
    ):
        weights[name] = [pack(weight, rows) for weight in getattr(g, name)]
    weights["lm_head_weights"] = pack(g.lm_head_weights, 2)
    max_len = g.k_cache.shape[2]
    if hf_config is None:
        cosine = g.rope_cos[:max_len].to(torch.float32).contiguous()
        sine = g.rope_sin[:max_len].to(torch.float32).contiguous()
        rope_source = "existing interleaved FP32 globals tables"
    else:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        if isinstance(hf_config, dict):
            hf_config = LlamaConfig.from_dict(hf_config)

        rope = LlamaRotaryEmbedding(hf_config, device=device)
        positions = torch.arange(max_len, device=device)[None]
        dummy = torch.empty(0, device=device, dtype=torch.float32)
        cosine, sine = rope(dummy, positions)
        permutation = torch.arange(64, device=device).reshape(2, 32).T.reshape(-1)
        cosine = cosine[0, :, permutation].contiguous()
        sine = sine[0, :, permutation].contiguous()
        rope_source = "HF LlamaRotaryEmbedding on device; FP32 then interleaved"
    torch.cuda.synchronize(device)
    return {
        "output_logits": torch.empty_like(g.logits,dtype=torch.float32),
        "normalized": torch.empty_like(g.hidden_states),
        "down_partial": torch.empty((4,2048),device=g.hidden_states.device,dtype=torch.float32),
        "ext": extension,
        "attention_module": load_attention(),
        "weights": weights,
        "rope_cos": cosine,
        "rope_sin": sine,
        "options": options,
        "rope_source": rope_source,
        "pack_and_rope_seconds": time.perf_counter() - start,
        "extra_allocated_bytes": torch.cuda.memory_allocated(device) - before,
        "numerical_status": "FP32 fused epilogues with BF16 stored activations; GPU validation pending",
    }


def operations(g, position, prepared):
    """Return 81 named ordinary kernel calls in model order for diagnostics."""
    ext = prepared["ext"]
    weights = prepared["weights"]
    options = prepared["options"]
    attention = prepared["attention_module"]
    ops = []
    if not 0 <= position < g.k_cache.shape[2]:
        raise ValueError("position outside cache allocation")
    for layer in range(16):
        key = g.k_cache[layer, 0]
        value = g.v_cache[layer, 0]

        def projection(x, weight, weight2, norm, output, mode, key=key, value=value):
            """Bind one layer's buffers without retaining mutable loop state."""
            return lambda: ext.project(
                x, weight, weight2, norm, output, key, value,
                prepared["rope_cos"], prepared["rope_sin"],
                position, g.rms_norm_eps, mode, options.pdl,
            )

        qkv = weights["qkv_proj_weights"][layer]
        output_weight = weights["o_proj_weights"][layer]
        down = weights["down_proj_weights"][layer]
        ops.append(("qkv", projection(g.hidden_states, qkv, qkv, g.attn_ln_weights[layer], g.post_ln_rope_q, 0)))
        ops.append(("attention", attention.make(
            g.post_ln_rope_q, key, value, g.attn_out, position,
            options.attention_tile, options.attention_warps,
            options.pdl, options.attention == "eager",
        )))
        ops.append(("o", projection(g.attn_out, output_weight, output_weight, g.attn_ln_weights[layer], g.hidden_states, 1)))
        ops.append(("upgate", projection(g.hidden_states, weights["up_proj_weights"][layer], weights["gate_proj_weights"][layer], g.mlp_ln_weights[layer], g.silu_out, 2)))
        ops.append(("down_split", lambda weight=down: ext.down_split(g.silu_out,weight,prepared["down_partial"],g.hidden_states,options.pdl)))
    head = weights["lm_head_weights"]
    ops.append(("head_norm", lambda:ext.head_norm(g.hidden_states,g.lm_head_norm_weights,prepared["normalized"],g.rms_norm_eps,options.pdl)))
    ops.append(("lmhead", projection(prepared["normalized"], head, head, g.lm_head_norm_weights, prepared["output_logits"], 5)))
    return ops


def build(g, position, prepared):
    """Build a no-argument full-model step reading hidden_states/writing logits."""
    ops = operations(g, position, prepared)

    def step():
        """Run all layers and the vocabulary head with no host/GPU synchronization."""
        for _, operation in ops:
            operation()

    return step


def output_logits(g,prepared):
    """Expose FP32 vocabulary logits; BF16 model weights/activations unchanged."""
    return prepared["output_logits"]
