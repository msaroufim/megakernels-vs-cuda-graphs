from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DSparkTailOutput:
    token_ids: torch.Tensor
    corrected_logits: torch.Tensor
    draft_probs: torch.Tensor
    confidence_logits: torch.Tensor
    prefix_length: torch.Tensor


def _confident_prefix_length(
    confidence_logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    proposal_len = confidence_logits.numel()
    if threshold <= 0.0:
        return torch.tensor(
            proposal_len,
            dtype=torch.int32,
            device=confidence_logits.device,
        )
    below = confidence_logits.sigmoid() < float(threshold)
    indices = torch.nonzero(below, as_tuple=False)
    length = proposal_len if indices.numel() == 0 else int(indices[0, 0])
    return torch.tensor(length, dtype=torch.int32, device=confidence_logits.device)


def dspark_tail_reference(
    base_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    first_prev_token_id: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    confidence_weight: torch.Tensor,
    confidence_bias: torch.Tensor,
    *,
    temperature: float = 0.0,
    uniforms: torch.Tensor | None = None,
    confidence_threshold: float = 0.0,
) -> DSparkTailOutput:
    """Reference for the sequential DSpark Markov/confidence tail.

    This matches the released vanilla Markov head. Positive-temperature tests
    pass controlled uniforms so a CUDA implementation can be compared without
    depending on PyTorch's private multinomial RNG implementation.
    """

    if base_logits.ndim != 2:
        raise ValueError("base_logits must be [proposal_len, vocab_size]")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [proposal_len, hidden_size]")
    proposal_len, vocab_size = base_logits.shape
    if hidden_states.shape[0] != proposal_len:
        raise ValueError("base_logits and hidden_states proposal lengths differ")
    if markov_w1.ndim != 2 or markov_w1.shape[0] != vocab_size:
        raise ValueError("markov_w1 must be [vocab_size, markov_rank]")
    if markov_w2.shape != markov_w1.shape:
        raise ValueError("markov_w2 must match markov_w1 shape")
    if confidence_weight.shape != (
        1,
        hidden_states.shape[1] + markov_w1.shape[1],
    ):
        raise ValueError("confidence_weight must be [1, hidden_size + markov_rank]")
    if confidence_bias.numel() != 1:
        raise ValueError("confidence_bias must contain one value")
    if first_prev_token_id.numel() != 1:
        raise ValueError("first_prev_token_id must contain one token")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if uniforms is not None:
        if uniforms.shape != (proposal_len,):
            raise ValueError("uniforms must have shape [proposal_len]")
        if not bool(torch.all((uniforms >= 0.0) & (uniforms < 1.0))):
            raise ValueError("uniforms must be in [0, 1)")
    elif temperature >= 1e-5:
        raise ValueError("positive-temperature reference sampling requires uniforms")

    corrected_steps = []
    probability_steps = []
    confidence_steps = []
    sampled_steps = []
    prev_token = first_prev_token_id.reshape(()).long()
    for step in range(proposal_len):
        prev_embedding = markov_w1[prev_token]
        bias = F.linear(prev_embedding, markov_w2)
        step_logits = base_logits[step] + bias
        if temperature < 1e-5:
            token = torch.argmax(step_logits, dim=-1)
            probs = torch.zeros_like(step_logits, dtype=torch.float32)
            probs.scatter_(0, token.reshape(1), 1.0)
        else:
            assert uniforms is not None
            probs = torch.softmax(step_logits.float() / float(temperature), dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            token = torch.searchsorted(
                cdf,
                uniforms[step].to(device=cdf.device, dtype=cdf.dtype),
                right=False,
            ).clamp_max(vocab_size - 1)
        confidence_features = torch.cat([hidden_states[step], prev_embedding])
        confidence = F.linear(
            confidence_features,
            confidence_weight,
            confidence_bias,
        ).float()

        corrected_steps.append(step_logits)
        probability_steps.append(probs)
        confidence_steps.append(confidence.reshape(()))
        sampled_steps.append(token)
        prev_token = token

    corrected_logits = torch.stack(corrected_steps, dim=0)
    confidence_logits = torch.stack(confidence_steps, dim=0)
    token_ids = torch.stack(sampled_steps, dim=0)
    return DSparkTailOutput(
        token_ids=token_ids,
        corrected_logits=corrected_logits,
        draft_probs=torch.stack(probability_steps, dim=0),
        confidence_logits=confidence_logits,
        prefix_length=_confident_prefix_length(
            confidence_logits,
            float(confidence_threshold),
        ),
    )
