from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


class UnsupportedMegaKernelConfig(ValueError):
    """Raised when a model cannot use the specialized megakernel path."""


@dataclass(frozen=True)
class DSparkMegaKernelSpec:
    """Frozen legacy Qwen3 reference megakernel contract.

    This path remains useful for small-checkpoint primitive validation. It is
    not the DeepSeek-V4 production specialization.
    """

    block_size: int = 7
    hidden_size: int = 2560
    intermediate_size: int = 9728
    vocab_size: int = 151936
    num_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    markov_rank: int = 256
    target_feature_count: int = 5
    dtype: str = "bfloat16"
    model_type: str = "qwen3"
    markov_head_type: str = "vanilla"

    @property
    def target_feature_width(self) -> int:
        return self.target_feature_count * self.hidden_size

    @classmethod
    def from_model(cls, model) -> "DSparkMegaKernelSpec":
        config = model.config
        spec = cls(
            block_size=int(model.block_size),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            vocab_size=int(config.vocab_size),
            num_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            markov_rank=int(config.markov_rank),
            target_feature_count=len(config.target_layer_ids),
            dtype="bfloat16",
            model_type=str(config.model_type),
            markov_head_type=str(config.markov_head_type),
        )
        spec.require_supported()
        return spec

    def incompatibilities(self) -> list[str]:
        expected = type(self)()
        fields = (
            "block_size",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "markov_rank",
            "target_feature_count",
            "dtype",
            "model_type",
            "markov_head_type",
        )
        return [
            f"{name}={getattr(self, name)!r}, expected {getattr(expected, name)!r}"
            for name in fields
            if getattr(self, name) != getattr(expected, name)
        ]

    def require_supported(self) -> None:
        problems = self.incompatibilities()
        if problems:
            raise UnsupportedMegaKernelConfig(
                "unsupported DSpark megakernel configuration: " + "; ".join(problems)
            )


@dataclass(frozen=True)
class DeepSeekV4MegaKernelSpec:
    """Fail-closed contract for the official V4-Flash DSpark checkpoint."""

    variant: str = "flash"
    model_type: str = "deepseek_v4"
    block_size: int = 5
    noise_token_id: int = 128799
    hidden_size: int = 4096
    vocab_size: int = 129280
    num_target_layers: int = 43
    target_layer_ids: tuple[int, ...] = (40, 41, 42)
    num_draft_layers: int = 3
    num_attention_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    output_groups: int = 8
    output_lora_rank: int = 1024
    window_size: int = 128
    num_routed_experts: int = 256
    num_shared_experts: int = 1
    num_activated_experts: int = 6
    expert_intermediate_size: int = 2048
    score_function: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    markov_rank: int = 256
    hc_multiplier: int = 4
    hc_sinkhorn_iterations: int = 20
    rms_epsilon: float = 1e-6
    hc_epsilon: float = 1e-6
    dense_dtype: str = "fp8_e4m3"
    dense_scale_format: str = "ue8m0"
    expert_dtype: str = "fp4_e2m1"

    @property
    def target_feature_width(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def mhc_width(self) -> int:
        return self.hc_multiplier * self.hidden_size

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.head_dim

    @property
    def maximum_sparse_kv(self) -> int:
        return self.window_size + self.block_size

    @staticmethod
    def _read(config: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    @classmethod
    def from_inference_config(cls, config: Mapping[str, Any] | Any) -> "DeepSeekV4MegaKernelSpec":
        read = cls._read
        hidden_size = int(read(config, "dim"))
        num_target_layers = int(read(config, "n_layers"))
        num_routed_experts = int(read(config, "n_routed_experts"))
        markov_rank = int(read(config, "dspark_markov_rank"))
        signature = (
            hidden_size,
            num_target_layers,
            num_routed_experts,
            markov_rank,
        )
        if signature == (4096, 43, 256, 256):
            variant = "flash"
        elif signature == (7168, 61, 384, 512):
            variant = "pro"
        else:
            variant = "unknown"
        dtype = str(read(config, "dtype"))
        expert_dtype = str(read(config, "expert_dtype"))
        return cls(
            variant=variant,
            block_size=int(read(config, "dspark_block_size")),
            noise_token_id=int(read(config, "dspark_noise_token_id")),
            hidden_size=hidden_size,
            vocab_size=int(read(config, "vocab_size")),
            num_target_layers=num_target_layers,
            target_layer_ids=tuple(int(value) for value in read(config, "dspark_target_layer_ids")),
            num_draft_layers=int(read(config, "n_mtp_layers")),
            num_attention_heads=int(read(config, "n_heads")),
            head_dim=int(read(config, "head_dim")),
            rope_head_dim=int(read(config, "rope_head_dim")),
            q_lora_rank=int(read(config, "q_lora_rank")),
            output_groups=int(read(config, "o_groups")),
            output_lora_rank=int(read(config, "o_lora_rank")),
            window_size=int(read(config, "window_size")),
            num_routed_experts=num_routed_experts,
            num_shared_experts=int(read(config, "n_shared_experts")),
            num_activated_experts=int(read(config, "n_activated_experts")),
            expert_intermediate_size=int(read(config, "moe_inter_dim")),
            score_function=str(read(config, "score_func")),
            route_scale=float(read(config, "route_scale")),
            swiglu_limit=float(read(config, "swiglu_limit")),
            markov_rank=markov_rank,
            hc_multiplier=int(read(config, "hc_mult")),
            hc_sinkhorn_iterations=int(read(config, "hc_sinkhorn_iters")),
            rms_epsilon=float(read(config, "norm_eps", 1e-6)),
            hc_epsilon=float(read(config, "hc_eps", 1e-6)),
            dense_dtype={"fp8": "fp8_e4m3"}.get(dtype, dtype),
            dense_scale_format=str(read(config, "scale_fmt")),
            expert_dtype={"fp4": "fp4_e2m1"}.get(expert_dtype, expert_dtype),
        )

    def incompatibilities(self) -> list[str]:
        expected = type(self)()
        return [
            f"{field.name}={getattr(self, field.name)!r}, "
            f"expected {getattr(expected, field.name)!r}"
            for field in fields(self)
            if getattr(self, field.name) != getattr(expected, field.name)
        ]

    def require_supported(self) -> None:
        problems = self.incompatibilities()
        if problems:
            raise UnsupportedMegaKernelConfig(
                "unsupported DeepSeek-V4-Flash DSpark megakernel configuration: "
                + "; ".join(problems)
            )
