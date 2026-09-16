"""Thin adapter to unmodified public MegaKittens Llama1B decode.

GPU selection belongs to the process environment before importing torch.
This adapter does not load model weights, create an allocation, or choose tokens.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch

class UpstreamDecoder:
    def __init__(self, source_root, hidden_states, weights, k_cache, v_cache,
                 rope_cos, rope_sin, eps=1e-5, *, cluster_size=1,
                 use_jit_cache=False, verbose=True):
        import megakittens
        from megakittens.jit.cuda_utils import initialize_cuda_context
        initialize_cuda_context()
        path = Path(source_root)/'examples/llama1b/compiled_decode.py'
        spec = importlib.util.spec_from_file_location('public_llama1b_decode', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.position = torch.zeros(1, device=hidden_states.device, dtype=torch.int32)
        self.scale = torch.tensor([0.125], device=hidden_states.device, dtype=torch.float32)
        self.eps = torch.tensor([eps], device=hidden_states.device, dtype=torch.float32)
        self.hidden_states = hidden_states
        self.k_cache, self.v_cache = k_cache, v_cache
        names = ('qkv_weights','o_weights','attn_norm_weights','mlp_norm_weights',
                 'up_weights','gate_weights','down_weights','lm_head_norm_weight','lm_head_weight')
        self.args = (hidden_states, *(weights[name] for name in names),
                     k_cache, v_cache, rope_cos, rope_sin, self.position, self.scale, self.eps)
        self.compiled = megakittens.compile(module.decode, cluster_size=cluster_size,
                                            use_jit_cache=use_jit_cache, verbose=verbose)
        self.last_logits = None

    def run(self, fixed_position=None):
        """Return actual output tensor; mutate hidden/KV and increment position.

        In sequence mode, initialize position once and call without a position.
        For fixed-state graphs, fixed_position's fill belongs inside timed replay.
        Input hidden state and any sequence reset belong to the external harness.
        """
        if fixed_position is not None:
            self.position.fill_(fixed_position)
        self.last_logits = self.compiled(*self.args)
        return self.last_logits


def from_legacy_globals(g, source_root, **kwargs):
    """Map already-interleaved weights/cache views; no mathematical conversion."""
    names = {
        'qkv_weights':'qkv_proj_weights', 'o_weights':'o_proj_weights',
        'attn_norm_weights':'attn_ln_weights','mlp_norm_weights':'mlp_ln_weights',
        'up_weights':'up_proj_weights','gate_weights':'gate_proj_weights',
        'down_weights':'down_proj_weights','lm_head_norm_weight':'lm_head_norm_weights',
        'lm_head_weight':'lm_head_weights',
    }
    weights = {key:getattr(g,value) for key,value in names.items()}
    seq_len = g.k_cache.numel() // (16*8*64)
    return UpstreamDecoder(source_root, g.hidden_states, weights,
                           g.k_cache.view(16,seq_len,8,64),
                           g.v_cache.view(16,seq_len,8,64),
                           g.rope_cos, g.rope_sin, g.rms_norm_eps, **kwargs)
