from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, cast

import torch

OFFICIAL_V4_FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
OFFICIAL_V4_FLASH_REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
OFFICIAL_V4_FLASH_SHARDS = (
    "model-00001-of-00048.safetensors",
    "model-00045-of-00048.safetensors",
    "model-00046-of-00048.safetensors",
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)


def _is_proposal_tensor(name: str) -> bool:
    return name in {"embed.weight", "head.weight"} or name.startswith("mtp.")


def selected_weight_map(index: Mapping[str, Any]) -> dict[str, str]:
    """Return the official checkpoint entries needed by the proposal-only model."""

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("model.safetensors.index.json has no weight_map object")
    selected: dict[str, str] = {}
    for raw_name, raw_shard in weight_map.items():
        if not isinstance(raw_name, str) or not isinstance(raw_shard, str):
            raise ValueError("checkpoint weight_map entries must be string pairs")
        if _is_proposal_tensor(raw_name):
            selected[raw_name] = raw_shard
    if "embed.weight" not in selected or "head.weight" not in selected:
        raise ValueError("checkpoint is missing the shared embedding or LM head")
    if not any(name.startswith("mtp.0.") for name in selected):
        raise ValueError("checkpoint has no mtp.0 proposal tensors")
    return selected


def required_checkpoint_shards(weight_map: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(set(weight_map.values())))


def official_snapshot_patterns() -> tuple[str, ...]:
    return (
        "model.safetensors.index.json",
        "inference/config.json",
        "inference/model.py",
        "inference/kernel.py",
        "inference/convert.py",
        *OFFICIAL_V4_FLASH_SHARDS,
    )


def download_official_v4_flash_snapshot(*, cache_dir: str | Path | None = None) -> Path:
    """Download only the pinned V4-Flash proposal code and five required shards."""

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=OFFICIAL_V4_FLASH_MODEL_ID,
        revision=OFFICIAL_V4_FLASH_REVISION,
        allow_patterns=list(official_snapshot_patterns()),
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return Path(path)


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official DeepSeek source from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@lru_cache(maxsize=2)
def _import_official_model(inference_dir_string: str) -> ModuleType:
    inference_dir = Path(inference_dir_string)
    kernel_path = inference_dir / "kernel.py"
    model_path = inference_dir / "model.py"
    if not kernel_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"official V4 inference sources are missing under {inference_dir}")

    kernel_name = "_deepspec_official_v4_flash_kernel"
    model_name = "_deepspec_official_v4_flash_model"
    kernel_module = _load_source_module(kernel_name, kernel_path)
    previous_kernel = sys.modules.get("kernel")
    sys.modules["kernel"] = kernel_module
    try:
        return _load_source_module(model_name, model_path)
    finally:
        if previous_kernel is None:
            sys.modules.pop("kernel", None)
        else:
            sys.modules["kernel"] = previous_kernel


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class OfficialV4FlashProposal(torch.nn.Module):
    """Proposal-only wrapper around DeepSeek's pinned V4 reference classes."""

    def __init__(self, source: ModuleType, args: Any, *, device: torch.device):
        super().__init__()
        self.source = source
        self.args = args
        setattr(source, "world_size", 1)
        setattr(source, "rank", 0)
        setattr(source, "default_dtype", torch.float8_e4m3fn)
        setattr(source, "scale_fmt", "ue8m0")
        setattr(source, "scale_dtype", torch.float8_e8m0fnu)

        with torch.device(device), _default_dtype(torch.bfloat16):
            self.embed = source.ParallelEmbedding(args.vocab_size, args.dim)
            self.head = source.ParallelHead(
                args.vocab_size,
                args.dim,
                args.norm_eps,
                args.hc_eps,
            )
            self.mtp = torch.nn.ModuleList()
            for stage_id in range(args.n_mtp_layers):
                block = source.DSparkBlock(args.n_layers + stage_id, args)
                block.embed = self.embed
                block.head = self.head
                self.mtp.append(block)

    @torch.inference_mode()
    def forward_main_projection(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """Expose the first official intermediate for custom-kernel parity."""

        with torch.device(main_hidden.device), _default_dtype(torch.bfloat16):
            first_layer = cast(Any, self.mtp[0])
            return first_layer.main_norm(first_layer.main_proj(main_hidden))

    @torch.inference_mode()
    def forward_front(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose official HC-expanded embeddings and projected target hidden."""

        with torch.device(main_hidden.device), _default_dtype(torch.bfloat16):
            first_layer = cast(Any, self.mtp[0])
            return first_layer.forward_embed(main_hidden, input_ids)

    @torch.inference_mode()
    def forward_main_kv(
        self,
        main_projected: torch.Tensor,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the three post-QAT target-KV rows and their RoPE values."""

        rows = []
        rope = None
        with torch.device(main_projected.device), _default_dtype(torch.bfloat16):
            for raw_layer in self.mtp:
                layer = cast(Any, raw_layer)
                attention = layer.attn
                freqs = attention.freqs_cis[start_pos : start_pos + 1]
                kv = attention.kv_norm(attention.wkv(main_projected))
                self.source.apply_rotary_emb(kv[..., -attention.rope_head_dim :], freqs)
                self.source.act_quant(
                    kv[..., : -attention.rope_head_dim],
                    64,
                    self.source.scale_fmt,
                    self.source.scale_dtype,
                    True,
                )
                rows.append(kv)
                if rope is None:
                    all_freqs = attention.freqs_cis[start_pos : start_pos + 6]
                    rope = torch.view_as_real(all_freqs).float().contiguous()
        assert rope is not None
        return torch.stack(rows), rope

    @torch.inference_mode()
    def forward_attn_hc(
        self,
        hidden_streams: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose one official attention-HC preconditioner boundary."""

        with torch.device(hidden_streams.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            shape = hidden_streams.shape
            flattened = hidden_streams.flatten(2).float()
            inverse_rms = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + layer.norm_eps)
            mixes = torch.nn.functional.linear(flattened, layer.hc_attn_fn) * inverse_rms
            pre, post, comb = self.source.hc_split_sinkhorn(
                mixes,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.hc_mult,
                layer.hc_sinkhorn_iters,
                layer.hc_eps,
            )
            reduced = torch.sum(
                pre.unsqueeze(-1) * flattened.view(shape),
                dim=2,
            ).to(hidden_streams.dtype)
            normalized = layer.attn_norm(reduced)
            return mixes, pre, post, comb, normalized

    @torch.inference_mode()
    def forward_attn_projections(
        self,
        normalized_hidden: torch.Tensor,
        *,
        start_pos: int,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose one layer's normalized Q-LoRA, queries, and draft KV."""

        with torch.device(normalized_hidden.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            attention = layer.attn
            freqs = attention.freqs_cis[start_pos + 1 : start_pos + 1 + normalized_hidden.size(1)]
            q_lora = attention.q_norm(attention.wq_a(normalized_hidden))
            query_projection = attention.wq_b(q_lora).unflatten(
                -1,
                (attention.n_local_heads, attention.head_dim),
            )
            query_inverse_rms = torch.rsqrt(
                query_projection.square().mean(-1, keepdim=True) + attention.eps
            )
            queries = query_projection * query_inverse_rms
            self.source.apply_rotary_emb(
                queries[..., -attention.rope_head_dim :],
                freqs,
            )
            draft_kv = attention.kv_norm(attention.wkv(normalized_hidden))
            self.source.apply_rotary_emb(
                draft_kv[..., -attention.rope_head_dim :],
                freqs,
            )
            self.source.act_quant(
                draft_kv[..., : -attention.rope_head_dim],
                64,
                self.source.scale_fmt,
                self.source.scale_dtype,
                True,
            )
            return (
                q_lora,
                query_projection,
                query_inverse_rms.squeeze(-1),
                queries,
                draft_kv,
            )

    @torch.inference_mode()
    def forward_sparse_attention(
        self,
        queries: torch.Tensor,
        draft_kv: torch.Tensor,
        target_kv_cache: torch.Tensor,
        *,
        start_pos: int,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose sparse attention before and after inverse RoPE."""

        with torch.device(queries.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            attention = layer.attn
            target_kv = target_kv_cache[layer_index]
            combined_kv = torch.cat([target_kv, draft_kv], dim=1)
            topk = self.source.get_dspark_topk_idxs(
                attention.window_size,
                queries.size(0),
                queries.size(1),
                start_pos,
            )
            raw = self.source.sparse_attn(
                queries,
                combined_kv,
                attention.attn_sink,
                topk,
                attention.softmax_scale,
            )
            output = raw.clone()
            freqs = attention.freqs_cis[start_pos + 1 : start_pos + 1 + queries.size(1)]
            self.source.apply_rotary_emb(
                output[..., -attention.rope_head_dim :],
                freqs,
                True,
            )
            return raw, output

    @torch.inference_mode()
    def forward_attention_output(
        self,
        attention_values: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose grouped output projection and the following HC update."""

        with torch.device(attention_values.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            attention = layer.attn
            batch, sequence, _, _ = attention_values.shape
            grouped = attention_values.view(
                batch,
                sequence,
                attention.n_local_groups,
                -1,
            )
            wo_a = attention.wo_a.weight.view(
                attention.n_local_groups,
                attention.o_lora_rank,
                -1,
            )
            output_lora = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
            output = attention.wo_b(output_lora.flatten(2))
            streams = layer.hc_post(output, residual, post, comb)
            return output_lora, output, streams

    @torch.inference_mode()
    def forward_hc_post(
        self,
        output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        """Expose HC-post independently of the preceding output projections."""

        with torch.device(output.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            return layer.hc_post(output, residual, post, comb)

    @torch.inference_mode()
    def forward_ffn_hc(
        self,
        hidden_streams: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose one official FFN-HC preconditioner boundary."""

        with torch.device(hidden_streams.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            shape = hidden_streams.shape
            flattened = hidden_streams.flatten(2).float()
            inverse_rms = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + layer.norm_eps)
            mixes = torch.nn.functional.linear(flattened, layer.hc_ffn_fn) * inverse_rms
            pre, post, comb = self.source.hc_split_sinkhorn(
                mixes,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                layer.hc_mult,
                layer.hc_sinkhorn_iters,
                layer.hc_eps,
            )
            reduced = torch.sum(
                pre.unsqueeze(-1) * flattened.view(shape),
                dim=2,
            ).to(hidden_streams.dtype)
            normalized = layer.ffn_norm(reduced)
            return mixes, pre, post, comb, normalized

    @torch.inference_mode()
    def forward_router(
        self,
        normalized_hidden: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose unbiased scores, selected indices, and normalized top-6 weights."""

        with torch.device(normalized_hidden.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            gate = layer.ffn.gate
            flattened = normalized_hidden.view(-1, gate.dim)
            logits = torch.nn.functional.linear(flattened.float(), gate.weight.float())
            scores = torch.nn.functional.softplus(logits).sqrt()
            selection = scores if gate.bias is None else scores + gate.bias
            indices = selection.topk(gate.topk, dim=-1)[1]
            weights = scores.gather(1, indices)
            weights /= weights.sum(dim=-1, keepdim=True)
            weights *= gate.route_scale
            shape = normalized_hidden.shape[:-1]
            return (
                scores.view(*shape, -1),
                indices.view(*shape, -1),
                weights.view(*shape, -1),
            )

    @torch.inference_mode()
    def forward_activation_quant(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the official per-128 FP8/E8M0 activation boundary."""

        with torch.device(hidden.device), _default_dtype(torch.bfloat16):
            return cast(
                tuple[torch.Tensor, torch.Tensor],
                self.source.act_quant(
                    hidden,
                    self.source.block_size,
                    self.source.scale_fmt,
                    self.source.scale_dtype,
                ),
            )

    @torch.inference_mode()
    def forward_routed_w13_probe(
        self,
        normalized_hidden: torch.Tensor,
        *,
        layer_index: int,
        expert_index: int,
        row_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one official routed expert's W1/W3 projections in isolation."""

        with torch.device(normalized_hidden.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            ffn = layer.ffn
            flattened = normalized_hidden.view(-1, ffn.dim)
            if not 0 <= row_index < flattened.size(0):
                raise IndexError(f"row_index {row_index} is outside {flattened.size(0)} rows")
            if not ffn.experts_start_idx <= expert_index < ffn.experts_end_idx:
                raise IndexError(
                    f"expert_index {expert_index} is outside "
                    f"[{ffn.experts_start_idx}, {ffn.experts_end_idx})"
                )
            expert = ffn.experts[expert_index]
            hidden = flattened[row_index : row_index + 1]
            return expert.w1(hidden), expert.w3(hidden)

    @torch.inference_mode()
    def forward_routed_w13_matrix_probe(
        self,
        hidden: torch.Tensor,
        *,
        layer_index: int,
        expert_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a full matrix through one routed expert's W1/W3 projections."""

        with torch.device(hidden.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            ffn = layer.ffn
            flattened = hidden.view(-1, ffn.dim)
            if not ffn.experts_start_idx <= expert_index < ffn.experts_end_idx:
                raise IndexError(
                    f"expert_index {expert_index} is outside "
                    f"[{ffn.experts_start_idx}, {ffn.experts_end_idx})"
                )
            expert = ffn.experts[expert_index]
            return expert.w1(flattened), expert.w3(flattened)

    def routed_fp4_kernel_source(self) -> str:
        """Compile and return the pinned official W1/W3 TileLang CUDA source."""

        fp4_gemm = cast(Any, self.source.fp4_gemm)
        globals_ = fp4_gemm.__globals__
        factory = cast(Any, globals_["fp4_gemm_kernel"])
        scale_dtype = globals_["FE8M0"]
        return cast(
            str,
            factory.get_kernel_source(
                2048,
                4096,
                scale_dtype=scale_dtype,
            ),
        )

    @torch.inference_mode()
    def forward_experts(
        self,
        normalized_hidden: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        residual_streams: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, ...]:
        """Expose routed/shared expert arithmetic and the following HC update."""

        with torch.device(normalized_hidden.device), _default_dtype(torch.bfloat16):
            layer = cast(Any, self.mtp[layer_index])
            ffn = layer.ffn
            shape = normalized_hidden.shape
            flattened = normalized_hidden.view(-1, ffn.dim)
            flat_indices = indices.view(-1, ffn.n_activated_experts)
            flat_weights = weights.view(-1, ffn.n_activated_experts)
            rows = flattened.size(0)
            intermediate = ffn.shared_experts.w1.out_features

            routed_w13 = torch.full(
                (rows, ffn.n_activated_experts, 2, intermediate),
                fill_value=float("nan"),
                dtype=flattened.dtype,
                device=flattened.device,
            )
            routed_swiglu = torch.full(
                (rows, ffn.n_activated_experts, intermediate),
                fill_value=float("nan"),
                dtype=flattened.dtype,
                device=flattened.device,
            )
            routed_partials = torch.full(
                (rows, ffn.n_activated_experts, ffn.dim),
                fill_value=float("nan"),
                dtype=flattened.dtype,
                device=flattened.device,
            )
            routed_filled = torch.zeros(
                rows,
                ffn.n_activated_experts,
                dtype=torch.bool,
                device=flattened.device,
            )
            routed_sum = torch.zeros_like(flattened, dtype=torch.float32)
            for expert_index in range(ffn.experts_start_idx, ffn.experts_end_idx):
                row, top = torch.where(flat_indices == expert_index)
                if row.numel() == 0:
                    continue
                expert = ffn.experts[expert_index]
                gate = expert.w1(flattened[row])
                up = expert.w3(flattened[row])
                routed_w13[row, top, 0] = gate
                routed_w13[row, top, 1] = up
                gate_float = gate.float()
                up_float = up.float()
                if expert.swiglu_limit > 0:
                    up_float = torch.clamp(
                        up_float,
                        min=-expert.swiglu_limit,
                        max=expert.swiglu_limit,
                    )
                    gate_float = torch.clamp(gate_float, max=expert.swiglu_limit)
                activation = (
                    torch.nn.functional.silu(gate_float) * up_float * flat_weights[row, top, None]
                ).to(flattened.dtype)
                routed_swiglu[row, top] = activation
                output = expert.w2(activation)
                routed_partials[row, top] = output
                routed_sum[row] += output
                routed_filled[row, top] = True

            if not bool(routed_filled.all().item()):
                missing = torch.nonzero(~routed_filled, as_tuple=False).cpu().tolist()
                raise AssertionError(
                    f"official routed expert capture left slots unfilled: {missing}"
                )

            shared = ffn.shared_experts
            shared_gate = shared.w1(flattened)
            shared_up = shared.w3(flattened)
            shared_gate_float = shared_gate.float()
            shared_up_float = shared_up.float()
            if shared.swiglu_limit > 0:
                shared_up_float = torch.clamp(
                    shared_up_float,
                    min=-shared.swiglu_limit,
                    max=shared.swiglu_limit,
                )
                shared_gate_float = torch.clamp(
                    shared_gate_float,
                    max=shared.swiglu_limit,
                )
            shared_swiglu = (torch.nn.functional.silu(shared_gate_float) * shared_up_float).to(
                flattened.dtype
            )
            shared_output = shared.w2(shared_swiglu)
            expert_output = (routed_sum + shared_output).to(flattened.dtype)
            expert_output = expert_output.view(*shape)
            updated_streams = layer.hc_post(
                expert_output,
                residual_streams,
                post,
                comb,
            )
            return (
                routed_w13.view(*shape[:-1], ffn.n_activated_experts, 2, intermediate),
                torch.stack((shared_gate, shared_up), dim=-2).view(
                    *shape[:-1],
                    2,
                    intermediate,
                ),
                routed_swiglu.view(
                    *shape[:-1],
                    ffn.n_activated_experts,
                    intermediate,
                ),
                shared_swiglu.view(*shape[:-1], intermediate),
                routed_partials.view(
                    *shape[:-1],
                    ffn.n_activated_experts,
                    ffn.dim,
                ),
                routed_sum.view(*shape),
                shared_output.view(*shape),
                expert_output,
                updated_streams,
            )

    @torch.inference_mode()
    def forward_head_intermediates(
        self,
        hidden_streams: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose the final HC reduction, RMSNorm, and five FP32 LM rows."""

        with torch.device(hidden_streams.device), _default_dtype(torch.bfloat16):
            last_layer = cast(Any, self.mtp[-1])
            hidden = last_layer.hc_head(
                hidden_streams,
                last_layer.hc_head_fn,
                last_layer.hc_head_scale,
                last_layer.hc_head_base,
            )
            normalized = last_layer.norm(hidden)
            assert last_layer.head is not None
            logits = last_layer.head(normalized, full_logits=True)
            return hidden, normalized, logits

    @torch.inference_mode()
    def forward_tail_intermediates(
        self,
        base_logits: torch.Tensor,
        head_hidden: torch.Tensor,
        anchor: torch.Tensor,
        *,
        sampling_temperature: float = 0.0,
        uniforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Expose controlled Markov correction, sampling, and confidence."""

        if sampling_temperature < 0.0:
            raise ValueError("sampling_temperature must be non-negative")
        if sampling_temperature >= 1.0e-5 and uniforms is None:
            raise ValueError("positive sampling_temperature requires uniforms")

        with torch.device(base_logits.device), _default_dtype(torch.bfloat16):
            last_layer = cast(Any, self.mtp[-1])
            output_ids = torch.empty(
                anchor.size(0),
                self.args.dspark_block_size + 1,
                dtype=anchor.dtype,
                device=anchor.device,
            )
            output_ids[:, 0] = anchor
            corrected = base_logits.clone()
            markov_embeddings = []
            markov_logits = []
            probability_rows = []
            for step in range(self.args.dspark_block_size):
                bias, embedding = last_layer.markov_head(output_ids[:, step])
                corrected[:, step].add_(bias)
                if sampling_temperature < 1.0e-5:
                    token = corrected[:, step].argmax(dim=-1)
                    probabilities = torch.zeros_like(corrected[:, step])
                    probabilities.scatter_(1, token[:, None], 1.0)
                else:
                    assert uniforms is not None
                    probabilities = torch.softmax(
                        corrected[:, step] / sampling_temperature,
                        dim=-1,
                    )
                    token = (
                        torch.searchsorted(
                            probabilities.cumsum(dim=-1),
                            uniforms[:, step, None],
                            right=False,
                        )
                        .squeeze(-1)
                        .clamp_max(corrected.size(-1) - 1)
                    )
                output_ids[:, step + 1] = token
                markov_embeddings.append(embedding)
                markov_logits.append(bias)
                probability_rows.append(probabilities)
            stacked_embeddings = torch.stack(markov_embeddings, dim=1)
            confidence = last_layer.confidence_head(head_hidden, stacked_embeddings)
            return (
                output_ids,
                corrected,
                torch.stack(probability_rows, dim=1),
                confidence,
                stacked_embeddings,
                torch.stack(markov_logits, dim=1),
            )

    @torch.inference_mode()
    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # DeepSeek's generate.py sets the process-wide default dtype to BF16.
        # Its TileLang wrappers allocate GEMM outputs from that default, so the
        # proposal-only wrapper must recreate the same dynamic context.
        # generate.py also calls torch.set_default_device("cuda"); helpers such
        # as get_dspark_topk_idxs otherwise create CPU tensors implicitly.
        with torch.device(input_ids.device), _default_dtype(torch.bfloat16):
            first_layer = cast(Any, self.mtp[0])
            hidden, main_x = first_layer.forward_embed(main_hidden, input_ids)
            for raw_layer in self.mtp:
                layer = cast(Any, raw_layer)
                hidden = layer(hidden, start_pos, input_ids, main_x)
            if start_pos == 0:
                return None
            last_layer = cast(Any, self.mtp[-1])
            return last_layer.forward_head(hidden, input_ids)

    @torch.inference_mode()
    def reset_kv_cache(self) -> None:
        for raw_layer in self.mtp:
            layer = cast(Any, raw_layer)
            layer.attn.kv_cache.zero_()

    def kv_cache_snapshot(self) -> torch.Tensor:
        layers = [cast(Any, layer) for layer in self.mtp]
        return torch.stack([layer.attn.kv_cache.detach().clone() for layer in layers])


def build_official_v4_flash_proposal(
    snapshot: str | Path,
    *,
    device: torch.device | str,
    max_batch_size: int = 1,
    max_seq_len: int = 4096,
    temperature: float = 0.0,
) -> OfficialV4FlashProposal:
    snapshot = Path(snapshot)
    source = _import_official_model(str((snapshot / "inference").resolve()))
    with (snapshot / "inference" / "config.json").open() as stream:
        config = json.load(stream)
    args = source.ModelArgs(**config)
    args.max_batch_size = max_batch_size
    args.max_seq_len = max_seq_len
    args.temperature = temperature
    proposal = OfficialV4FlashProposal(source, args, device=torch.device(device))
    return proposal.eval()


@dataclass(frozen=True)
class OfficialV4LoadReport:
    loaded_tensors: int
    loaded_bytes: int
    checkpoint_shards: tuple[str, ...]


def _dequantize_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        (
            weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
            * scale[:, None, :, None].float()
        )
        .flatten(2, 3)
        .flatten(0, 1)
        .bfloat16()
    )


@torch.inference_mode()
def load_official_v4_flash_weights(
    proposal: OfficialV4FlashProposal,
    snapshot: str | Path,
) -> OfficialV4LoadReport:
    """Load selected raw HF tensors using DeepSeek's conversion semantics."""

    from safetensors import safe_open

    snapshot = Path(snapshot)
    with (snapshot / "model.safetensors.index.json").open() as stream:
        index = json.load(stream)
    weight_map = selected_weight_map(index)
    shards = required_checkpoint_shards(weight_map)
    if shards != OFFICIAL_V4_FLASH_SHARDS:
        raise ValueError(f"unexpected V4-Flash proposal shards: {shards}")

    parameters = dict(proposal.named_parameters())
    loaded: set[str] = set()
    loaded_bytes = 0
    for shard in shards:
        names = sorted(name for name, mapped_shard in weight_map.items() if mapped_shard == shard)
        with safe_open(snapshot / shard, framework="pt", device="cpu") as reader:
            for name in names:
                if name.endswith(".wo_a.scale"):
                    continue
                target = parameters.get(name)
                if target is None:
                    raise ValueError(
                        f"official proposal has no parameter for checkpoint tensor {name}"
                    )
                source = reader.get_tensor(name)
                loaded_bytes += source.numel() * source.element_size()
                if name.endswith(".wo_a.weight"):
                    scale_name = name.replace("weight", "scale")
                    scale = reader.get_tensor(scale_name)
                    loaded_bytes += scale.numel() * scale.element_size()
                    source = _dequantize_wo_a(source, scale)
                elif ".experts." in name and name.endswith(".weight"):
                    if source.dtype != torch.int8:
                        raise ValueError(
                            f"expected packed int8 storage for {name}, got {source.dtype}"
                        )
                    source = source.view(torch.float4_e2m1fn_x2)
                source = source.to(device=target.device, dtype=target.dtype)
                target.copy_(source)
                loaded.add(name)

    missing = sorted(set(parameters) - loaded)
    if missing:
        raise ValueError(f"official proposal parameters were not loaded: {missing[:8]}")
    expected_source = set(weight_map)
    consumed_source = loaded | {name for name in expected_source if name.endswith(".wo_a.scale")}
    unexpected = sorted(expected_source - consumed_source)
    if unexpected:
        raise ValueError(f"official checkpoint tensors were not consumed: {unexpected[:8]}")
    return OfficialV4LoadReport(
        loaded_tensors=len(consumed_source),
        loaded_bytes=loaded_bytes,
        checkpoint_shards=shards,
    )
