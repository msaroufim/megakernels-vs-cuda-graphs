from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open

from deepspec.megakernel.device_graph_runtime import captured_node_types
from deepspec.megakernel.full_loop_compiler import (
    GB300GraftLowering,
    compile_gb300_graft_lowering,
)
from deepspec.megakernel.full_loop_moe_compiler import (
    markov_add_argmax_cuda,
    prepare_target_inject_layout_cuda,
    target_hc_head_rmsnorm_cuda,
    target_tap_mean_concat_cuda,
)


class _SharedEmbedding(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, cast(torch.Tensor, self.weight))


class _SharedLmHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.org_vocab_size = int(weight.shape[0])
        self.tp_size = 1
        self.num_embeddings_per_partition = self.org_vocab_size
        self.num_embeddings_padded = self.org_vocab_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=self.org_vocab_size,
        )


def _load_global(snapshot: Path, name: str) -> torch.Tensor:
    with (snapshot / "model.safetensors.index.json").open() as stream:
        shard = json.load(stream)["weight_map"][name]
    with safe_open(snapshot / shard, framework="pt", device="cpu") as reader:
        return reader.get_tensor(name).cuda()


def _find_captured_graph(runner) -> torch.cuda.CUDAGraph:
    candidates = []
    for owner_name, owner in (("runner", runner), *vars(runner).items()):
        graphs = getattr(owner, "_graphs", None)
        if isinstance(graphs, dict):
            candidates.extend((owner_name, key, graph) for key, graph in graphs.items())
    if len(candidates) != 1:
        summary = [(owner, repr(key), type(graph).__name__) for owner, key, graph in candidates]
        raise RuntimeError(f"expected one captured draft graph, found {summary}")
    _, _, graph = candidates[0]
    if not isinstance(graph, torch.cuda.CUDAGraph):
        raise TypeError(f"captured graft is {type(graph).__name__}")
    return graph


def _require_device_launchable(graph: torch.cuda.CUDAGraph, *, label: str) -> tuple[int, ...]:
    node_types = captured_node_types(graph)
    unsupported = tuple(kind for kind in node_types if kind not in (0, 1))
    if unsupported:
        raise RuntimeError(f"{label} graph contains unsupported nodes: {unsupported}")
    return node_types


def split_greedy_argmax(step_logits: torch.Tensor, step_idx: int = 0) -> torch.Tensor:
    """Use SGLang's split-CTA top-1 reduction for one graft sampling step."""

    del step_idx
    from sglang.kernels.ops.speculative.topk1 import (  # ty: ignore[unresolved-import]
        draft_topk1_postprocess,
    )

    scratch_positions = torch.empty(
        (step_logits.shape[0],), dtype=torch.int64, device=step_logits.device
    )
    _, topk_index = draft_topk1_postprocess(step_logits, scratch_positions)
    return topk_index.flatten()


def split_greedy_argmax_into(
    step_logits: torch.Tensor,
    step_idx: int,
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Reduce top-1 and publish it directly into the compiler-owned draft buffer."""

    from sglang.kernels.ops.speculative.topk1 import (  # ty: ignore[unresolved-import]
        draft_topk1_postprocess,
    )

    scratch_positions = torch.empty(
        (step_logits.shape[0],), dtype=torch.int64, device=step_logits.device
    )
    _, topk_index = draft_topk1_postprocess(
        step_logits,
        scratch_positions,
        draft_tokens=draft_tokens,
        draft_token_column=step_idx,
    )
    return topk_index.flatten()


@dataclass
class CapturedGraftBand:
    """Compiler-owned handle to SGLang's complete optimized graft executor."""

    graph: torch.cuda.CUDAGraph
    bundle: Any
    runner: Any
    sampler: Any
    node_types: tuple[int, ...]
    embed_weight: torch.Tensor
    head_weight: torch.Tensor
    control_graph: torch.cuda.CUDAGraph | None = None
    control_runner: Any | None = None
    head_normalized: torch.Tensor | None = None
    head_logits: torch.Tensor | None = None
    head_streams: torch.Tensor | None = None

    @property
    def input_ids(self) -> torch.Tensor:
        return self.runner.buffers.input_ids[:6]

    @property
    def proposal_tokens(self) -> torch.Tensor:
        return self.sampler.out[:5]

    @property
    def seq_lens(self) -> torch.Tensor:
        return self.runner.buffers.seq_lens[:1]

    @property
    def positions(self) -> torch.Tensor:
        return self.runner.buffers.positions[:5]

    @property
    def out_cache_loc(self) -> torch.Tensor:
        return self.runner.buffers.out_cache_loc[:5]


@dataclass
class CapturedTargetInjectBand:
    """Target-tap projection and committed SWA-KV writes for the graft."""

    graph: torch.cuda.CUDAGraph
    node_types: tuple[int, ...]
    target_hidden: torch.Tensor
    swa_loc: torch.Tensor
    positions: torch.Tensor


def capture_sglang_graft_band(
    snapshot: Path,
    *,
    dist_port: int = 29631,
    embed_weight: torch.Tensor | None = None,
    head_weight: torch.Tensor | None = None,
    head_weight_kn: torch.Tensor | None = None,
    proposal_output: torch.Tensor | None = None,
    lowering: GB300GraftLowering | None = None,
    fused_markov_argmax: bool = False,
    capture_markov_control: bool = False,
) -> CapturedGraftBand:
    """Capture the exact batch-one/verify-six SGLang DSpark graft leaf."""

    lowering = lowering or compile_gb300_graft_lowering()
    if lowering.autotune:
        raise ValueError("the standalone GB300 graft lowering forbids autotuning")
    if lowering.executor != "sglang_cuda_graph":
        raise ValueError(f"unsupported graft executor: {lowering.executor}")
    if lowering.greedy_argmax != "sglang_split_topk1":
        raise ValueError(f"unsupported greedy argmax: {lowering.greedy_argmax}")
    if lowering.hc_head != "sm103_fused_hc_rmsnorm":
        raise ValueError(f"unsupported graft HC head: {lowering.hc_head}")
    if lowering.lm_head_layout != "kn_contiguous":
        raise ValueError(f"unsupported graft LM-head layout: {lowering.lm_head_layout}")
    if not lowering.direct_proposal_store:
        raise ValueError("the GB300 graft requires direct proposal publication")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError("GB300 sm_103 is required")

    from sglang.srt.configs.model_config import ModelConfig  # ty: ignore[unresolved-import]
    from sglang.srt.distributed import bootstrap  # ty: ignore[unresolved-import]
    from sglang.srt.distributed.parallel_state_wrapper import (  # ty: ignore[unresolved-import]
        ParallelState,
    )
    from sglang.srt.layers.moe.utils import (  # ty: ignore[unresolved-import]
        initialize_moe_config,
        speculative_moe_a2a_backend_context,
        speculative_moe_backend_context,
    )
    from sglang.srt.model_executor.pool_configurator import (  # ty: ignore[unresolved-import]
        MemoryPoolConfig,
    )
    from sglang.srt.runtime_context import get_context  # ty: ignore[unresolved-import]
    from sglang.srt.server_args import ServerArgs  # ty: ignore[unresolved-import]
    from sglang.srt.speculative.draft_worker_common import (  # ty: ignore[unresolved-import]
        build_draft_tp_worker,
    )
    from sglang.srt.speculative.dspark_components import (  # ty: ignore[unresolved-import]
        dspark_draft,
    )
    from sglang.srt.speculative.dspark_components.dspark_config import (  # ty: ignore[unresolved-import]
        DSV4_DRAFT_ATTENTION_BACKEND,
    )

    server_args = ServerArgs(
        model_path=str(snapshot),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        tp_size=1,
        speculative_algorithm="DSPARK",
        speculative_num_draft_tokens=6,
        mem_fraction_static=0.35,
        max_running_requests=1,
        max_total_tokens=1024,
        cuda_graph_max_bs_decode=1,
        disable_flashinfer_autotune=True,
        disable_radix_cache=True,
        # The pinned DSpark loader does not remap shared weights into fused
        # routed-expert slots. Keep the checkpoint's FP8 shared experts separate
        # so their gate/up/down weights are loaded instead of silently skipped.
        disable_shared_experts_fusion=True,
        speculative_moe_runner_backend="flashinfer_mxfp4",
    )
    get_context().set_server_args(server_args)
    initialize_moe_config(server_args)
    target_config = ModelConfig(
        str(snapshot),
        trust_remote_code=True,
        speculative_algorithm="DSPARK",
    )
    parallel_state = ParallelState.trivial()
    bootstrap.init_torch_distributed(
        server_args=server_args,
        model_config=target_config,
        device="cuda",
        ps=parallel_state,
        dist_port=dist_port,
        is_draft_worker=False,
        local_omp_cpuid=None,
    )
    with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=0,
            ps=parallel_state,
            nccl_port=dist_port,
            target_model_config=target_config,
            algo_label="DSPARK",
            attention_backend_override=DSV4_DRAFT_ATTENTION_BACKEND,
        )
    for stage in bundle.draft_model.stages:
        if stage.mlp.num_fused_shared_experts != 0 or not hasattr(stage.mlp, "shared_experts"):
            raise RuntimeError("DSpark graph baseline requires separately loaded shared experts")
    if embed_weight is None:
        embed_weight = _load_global(snapshot, "embed.weight")
    if head_weight is None:
        head_weight = _load_global(snapshot, "head.weight")
    if head_weight_kn is None:
        head_weight_kn = head_weight.T.contiguous()
    bundle.draft_model.attach_shared_modules(
        embed_tokens=_SharedEmbedding(embed_weight),
        lm_head=_SharedLmHead(head_weight),
    )
    with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
        bundle.draft_worker.alloc_memory_pool(
            memory_pool_config=MemoryPoolConfig(
                max_total_num_tokens=1024,
                max_running_requests=1,
                full_max_total_num_tokens=1024,
                swa_max_total_num_tokens=1024,
            )
        )
        bundle.draft_worker.init_attention_backends()
    sampler = dspark_draft.maybe_build_draft_sampler(
        draft_model=bundle.draft_model,
        gamma=5,
        max_bs=1,
        device=bundle.draft_model_runner.device,
        tp_rank=0,
        out=proposal_output,
    )
    if sampler is None:
        raise RuntimeError("SGLang refused to fold the greedy draft sampler")
    fused_head_normalized = torch.empty((sampler.gamma, 4096), dtype=torch.bfloat16, device="cuda")
    fused_head_logits = torch.empty(
        (sampler.gamma, int(head_weight.shape[0])),
        dtype=torch.bfloat16,
        device="cuda",
    )
    markov_argmax_slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    markov_head = sampler.markov_head
    markov_shard = getattr(markov_head, "_tp_shard", None)
    if fused_markov_argmax:
        if markov_shard is None or int(markov_shard.tp_size) != 1:
            raise RuntimeError("fused Markov argmax currently requires the TP1 sharded ABI")
        markov_weight = markov_head.markov_w2.weight[
            int(markov_shard.org_vocab_start) : int(markov_shard.org_vocab_end)
        ]

    retained_head_streams: torch.Tensor | None = None

    def fused_base_logits(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal retained_head_streams
        retained_head_streams = hidden_states
        last = bundle.draft_model.stages[-1]
        target_hc_head_rmsnorm_cuda(
            hidden_states,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
            last.norm.weight,
            output=fused_head_normalized,
            norm_eps=float(bundle.draft_model.norm_eps),
            hc_eps=float(bundle.draft_model.hc_eps),
        )
        torch.mm(fused_head_normalized, head_weight_kn, out=fused_head_logits)
        return fused_head_logits, fused_head_normalized

    use_fused_markov_argmax = fused_markov_argmax

    def capture_direct_sampler(_runner, model_output, forward_batch, _num_tokens):
        hidden_states = model_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("draft forward has no hidden states for folded sampling")
        batch = hidden_states.shape[0] // sampler.gamma
        base_logits, _ = fused_base_logits(hidden_states)
        base_logits = base_logits.view(batch, sampler.gamma, -1)
        anchor = forward_batch.input_ids.view(batch, sampler.gamma)[:, 0]
        output = sampler.out.view(batch, sampler.gamma)

        def direct_step(logits: torch.Tensor, step: int) -> torch.Tensor:
            return split_greedy_argmax_into(logits, step, output)

        if use_fused_markov_argmax:
            prev_tokens = anchor
            for step in range(sampler.gamma):
                latent = markov_head.get_prev_embeddings(prev_tokens)
                bias = F.linear(
                    latent.to(markov_weight.dtype),
                    markov_weight,
                )
                prev_tokens = markov_add_argmax_cuda(
                    base_logits[:, step, :],
                    bias,
                    output,
                    draft_token_column=step,
                    argmax_slot=markov_argmax_slot,
                )
        else:
            sampler.markov_head.sample_block(
                base_logits,
                first_prev_tokens=anchor,
                hidden_states=hidden_states.view(batch, sampler.gamma, -1),
                sampler=direct_step,
            )

    bundle.draft_model_runner.capture_tail_hooks.append(capture_direct_sampler)

    def capture_current_tail() -> tuple[Any, torch.cuda.CUDAGraph]:
        original_cuda_graph = torch.cuda.CUDAGraph
        original_greedy_step_sampler = dspark_draft.greedy_step_sampler
        torch.cuda.CUDAGraph = lambda: original_cuda_graph(  # ty: ignore[invalid-assignment]
            keep_graph=True
        )
        dspark_draft.greedy_step_sampler = split_greedy_argmax
        try:
            with (
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                bundle.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=True)
        finally:
            torch.cuda.CUDAGraph = original_cuda_graph
            dspark_draft.greedy_step_sampler = original_greedy_step_sampler
        runner = bundle.draft_model_runner.decode_cuda_graph_runner
        return runner, _find_captured_graph(runner)

    control_runner = None
    control_graph = None
    if capture_markov_control:
        if not fused_markov_argmax:
            raise ValueError("paired Markov capture requires fused_markov_argmax=True")
        use_fused_markov_argmax = False
        control_runner, control_graph = capture_current_tail()
        use_fused_markov_argmax = True
    runner, graph = capture_current_tail()
    return CapturedGraftBand(
        graph=graph,
        bundle=bundle,
        runner=runner,
        sampler=sampler,
        node_types=_require_device_launchable(graph, label="graft"),
        embed_weight=embed_weight,
        head_weight=head_weight,
        control_graph=control_graph,
        control_runner=control_runner,
        head_normalized=fused_head_normalized,
        head_logits=fused_head_logits,
        head_streams=retained_head_streams,
    )


def capture_target_inject_band(
    graft: CapturedGraftBand,
    taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    commit_len: torch.Tensor,
    prefix_len: torch.Tensor,
) -> CapturedTargetInjectBand:
    """Lower three completed target HC taps into the draft's SWA KV pool."""

    if any(tap.shape != (6, 4, 4096) for tap in taps):
        raise ValueError("target injection requires three [6, 4, 4096] taps")
    if any(tap.dtype != torch.bfloat16 or not tap.is_cuda for tap in taps):
        raise ValueError("target taps must be CUDA BF16 tensors")
    target_hidden = torch.empty((6, 3 * 4096), dtype=torch.bfloat16, device=taps[0].device)
    swa_loc = torch.empty(6, dtype=torch.int32, device=taps[0].device)
    positions = torch.empty(6, dtype=torch.int64, device=taps[0].device)

    def inject_once() -> None:
        target_tap_mean_concat_cuda(taps, output=target_hidden)
        prepare_target_inject_layout_cuda(
            commit_len,
            prefix_len,
            swa_loc=swa_loc,
            positions=positions,
        )
        graft.bundle.draft_model.write_target_hidden_kv(
            main_hidden=target_hidden,
            swa_loc=swa_loc,
            positions=positions,
            pool=graft.bundle.draft_model_runner.token_to_kv_pool,
        )

    for _ in range(3):
        inject_once()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        inject_once()
    return CapturedTargetInjectBand(
        graph=graph,
        node_types=_require_device_launchable(graph, label="target inject"),
        target_hidden=target_hidden,
        swa_loc=swa_loc,
        positions=positions,
    )
