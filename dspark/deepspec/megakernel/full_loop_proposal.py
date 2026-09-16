from __future__ import annotations

import ctypes
import math
import os
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from scripts.megakernel.boltin_pack import BoltinPackedWeights

from deepspec.megakernel.device_graph_runtime import (
    captured_node_types,
    prepare_proposal_megakernel,
    publish_proposal_candidates,
)
from deepspec.megakernel.extension import (
    V4_EXECUTE_ALL,
    V4_EXECUTE_ATTENTION_OUTPUT,
    V4_EXECUTE_ATTN_HC,
    V4_EXECUTE_ATTN_PROJECTIONS,
    V4_EXECUTE_EMBEDDING,
    V4_EXECUTE_FFN_ROUTER,
    V4_EXECUTE_HEAD,
    V4_EXECUTE_MAIN,
    V4_EXECUTE_MAIN_KV,
    V4_EXECUTE_SPARSE_ATTENTION,
    V4_EXECUTE_TAIL,
    V4ProposalOutputs,
    allocate_v4_proposal_outputs,
    allocate_v4_scheduler_workspace,
    run_v4_front_kv_stage,
    view_v4_workspace_region,
)
from deepspec.megakernel.full_loop_compiler import GB300GraftLowering
from deepspec.megakernel.full_loop_moe_compiler import expert_hc_post_cuda

_BLOCK = 5
_DEVICE_SCALARS_BYTES = 16


def _require_device_launchable(
    graph: torch.cuda.CUDAGraph,
    *,
    label: str,
) -> tuple[int, ...]:
    node_types = captured_node_types(graph)
    unsupported = tuple(kind for kind in node_types if kind not in (0, 1))
    if unsupported:
        raise RuntimeError(f"{label} graph contains unsupported nodes: {unsupported}")
    return node_types


@dataclass
class CapturedProposalMegakernelBand:
    """Final GB300 proposal scheduler adapted to the full-loop graft ABI."""

    input_graph: torch.cuda.CUDAGraph
    graph: torch.cuda.CUDAGraph
    input_node_types: tuple[int, ...]
    node_types: tuple[int, ...]
    keepalive: tuple[Any, ...]
    uncaptured_proposal: Callable[[], None] | None = None
    recapture_proposal: Callable[[ModuleType], CapturedProposalMegakernelBand] | None = None
    proposal_outputs: V4ProposalOutputs | None = None
    io_binding: Any | None = None


_DRAFT_BODY_MASK = (
    V4_EXECUTE_ATTN_HC
    | V4_EXECUTE_ATTN_PROJECTIONS
    | V4_EXECUTE_SPARSE_ATTENTION
    | V4_EXECUTE_ATTENTION_OUTPUT
    | V4_EXECUTE_FFN_ROUTER
)


def capture_proposal_megakernel_band(
    snapshot: Path,
    *,
    graft: Any,
    target_hidden: torch.Tensor,
    bonus: torch.Tensor,
    commit_len: torch.Tensor,
    new_seq_len: torch.Tensor,
    candidates: torch.Tensor,
    lowering: GB300GraftLowering,
    instrumented: bool = False,
    weight_cache: dict[tuple[object, ...], BoltinPackedWeights] | None = None,
    extension_module: ModuleType | None = None,
    integrated_io: bool = False,
    greedy_tail: bool = True,
    sampling_temperature: float = 0.0,
    uniforms: torch.Tensor | None = None,
    sts_temperatures: torch.Tensor | None = None,
    steps_per_second: torch.Tensor | None = None,
) -> CapturedProposalMegakernelBand:
    """Capture accepted-state staging and the final proposal megakernel.

    The target-inject band remains the single writer of committed draft SWA
    state. The input graph gathers that authoritative ring, selects the newest
    committed target feature, advances a device-owned scheduler epoch, and
    builds RoPE. The proposal graph then runs the persistent 152-CTA scheduler
    and publishes anchor plus five draft IDs directly into the next verifier.
    ``extension_module`` optionally binds a precompiled scheduler for matched
    experiments through the existing launch API; normal calls use its loader.

    Set ``greedy_tail=False`` to capture the existing normalized relaxed tail.
    Positive temperature requires caller-owned controlled ``uniforms`` [1,5].
    Their addresses remain fixed for replay; update their values before the input
    graph, keeping uniforms in [0,1]. Optional STS temperatures [5] and SPS curve
    [7] retain the existing unit-temperature/example-curve defaults. They are
    validation inputs, not calibrated serving policy. The returned
    ``proposal_outputs`` exposes probabilities and prefix results; those fields
    remain unwritten in the default greedy mode. This capture is not stochastic
    certification or an integration with the full-loop rejection sampler.

    Non-greedy capture uses the normal loader, which rejects greedy-only build
    settings. Precompiled modules have no tail-capability query, so they remain
    available only for the existing greedy experiment.

    ``integrated_io=True`` requires the experimental native I/O ABI. It captures
    one kernel that also stages inputs, gathers KV and publishes candidates;
    its input graph is empty. Each band owns a private workspace and must only
    run nonoverlapping invocations. ``io_binding.verify_resident()`` audits its
    immutable bindings and completed epochs outside capture and timing.
    """

    io_support = None
    if integrated_io:
        if extension_module is None or instrumented or not greedy_tail:
            raise ValueError("Integrated I/O requires an explicit, uninstrumented greedy module")
        from deepspec.megakernel import proposal_io as io_support

        io_support.image_receipt(extension_module)
    if not math.isfinite(sampling_temperature) or sampling_temperature < 0.0:
        raise ValueError("sampling_temperature must be finite and non-negative")
    if greedy_tail and sampling_temperature >= 1.0e-5:
        raise ValueError("positive sampling_temperature requires greedy_tail=False")
    if sampling_temperature >= 1.0e-5 and uniforms is None:
        raise ValueError("positive sampling_temperature requires controlled uniforms")
    if not greedy_tail and extension_module is not None:
        raise ValueError("non-greedy capture requires the normal extension loader")
    library = ctypes.CDLL(extension_module.__file__) if extension_module is not None else None
    retired_markers = (
        "dspark_full_proposal_probe_abi",
        "dspark_full_proposal_group32_abi",
        "dspark_full_proposal_group32_probe_abi",
        "dspark_full_proposal_stream_boundaries_abi",
        "dspark_full_proposal_layer0_boundaries_abi",
        "dspark_full_proposal_attention_boundaries_abi",
    )
    if library is not None and any(hasattr(library, name) for name in retired_markers):
        raise ValueError("Archived proposal variants require their original host bindings")
    for name, tensor, shape in (
        ("uniforms", uniforms, (1, _BLOCK)),
        ("sts_temperatures", sts_temperatures, (_BLOCK,)),
        ("steps_per_second", steps_per_second, (_BLOCK + 2,)),
    ):
        if tensor is not None and (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.device != target_hidden.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous float32 {shape} on the target device")
    if lowering.executor != "proposal_megakernel":
        raise ValueError(f"unsupported proposal executor: {lowering.executor}")
    if target_hidden.shape != (6, 3 * 4096) or target_hidden.dtype != torch.bfloat16:
        raise ValueError("proposal target_hidden must be BF16 [6,12288]")
    if candidates.shape != (6,) or candidates.dtype != torch.int64:
        raise ValueError("proposal candidates must be int64 [6]")
    properties = torch.cuda.get_device_properties(target_hidden.device)
    if (properties.major, properties.minor) != (10, 3) or properties.multi_processor_count != 152:
        raise RuntimeError("the proposal megakernel requires a 152-SM GB300/sm_103")

    from scripts.megakernel.boltin_kv_accessor import (
        bind_full_loop_graft_kv_accessor,
    )
    from scripts.megakernel.boltin_pack import pack_boltin_arena

    layout = dict(
        lm_bulk_staging=os.environ.get("DSPARK_LM_BULK_STAGING") == "1",
        woa_bulk_staging=os.environ.get("DSPARK_WOA_BULK_STAGING") == "1",
        qb_block_scaled=os.environ.get("DSPARK_QB_BLOCK_SCALED") in ("2", "4"),
        dense_block_scaled=os.environ.get("DSPARK_DENSE_BLOCK_SCALED") in ("2", "4"),
        markov_bulk_staging=os.environ.get("DSPARK_MARKOV_BULK_K") in ("128", "256"),
        routed_packed_sfa=os.environ.get("DSPARK_ROUTED_PACKED_SFA") in ("1", "2"),
        routed_bulk_weights=os.environ.get("DSPARK_ROUTED_BULK_WEIGHTS") == "1",
        routed_interleave=os.environ.get("DSPARK_ROUTED_INTERLEAVE") == "1",
        shared_bulk_staging=os.environ.get("DSPARK_SHARED_BULK_STAGING") == "1",
    )
    # Candidate rings with the same immutable layout share weights. Capture
    # and timing still use separate workspaces and the same warm replay policy.
    cache_key = (str(snapshot.resolve()), str(target_hidden.device), *layout.values())
    packed = weight_cache.get(cache_key) if weight_cache is not None else None
    if packed is None:
        packed = pack_boltin_arena(snapshot, device=target_hidden.device, **layout)
        if weight_cache is not None:
            weight_cache[cache_key] = packed
    accessor = bind_full_loop_graft_kv_accessor(graft)
    workspace = allocate_v4_scheduler_workspace(
        device=target_hidden.device,
        extra_scratch_bytes=(
            io_support.EXTRA_BYTES if io_support is not None else _DEVICE_SCALARS_BYTES
        ),
    )
    if integrated_io:
        from deepspec.megakernel.v4_abi import V4LaunchShape, build_v4_launch_abi

        base_bytes = build_v4_launch_abi(shape=V4LaunchShape(batch_size=1)).workspace.total_bytes
        start_pos = workspace.scratch[base_bytes : base_bytes + 8].view(torch.int64)
        epoch = workspace.scratch[base_bytes + 8 : base_bytes + 16].view(torch.int64)
    else:
        start_pos = workspace.scratch[-16:-8].view(torch.int64)
        epoch = workspace.scratch[-8:].view(torch.int64)
    start_pos.fill_(1)
    epoch.zero_()
    outputs = allocate_v4_proposal_outputs(device=target_hidden.device)
    anchor = torch.empty(1, dtype=torch.int32, device=target_hidden.device)
    main_hidden = torch.empty((1, 1, 3 * 4096), dtype=torch.bfloat16, device=target_hidden.device)
    main_output = torch.empty((1, 1, 4096), dtype=torch.bfloat16, device=target_hidden.device)
    embedding_output = torch.empty(
        (1, _BLOCK, 4, 4096), dtype=torch.bfloat16, device=target_hidden.device
    )
    rope = torch.empty((_BLOCK + 1, 32, 2), dtype=torch.float32, device=target_hidden.device)
    freqs_real = torch.view_as_real(packed.freqs_cis)
    if not freqs_real.is_contiguous():
        freqs_real = freqs_real.contiguous()
    request_index = torch.zeros(1, dtype=torch.int64, device=target_hidden.device)
    if uniforms is None:
        uniforms = torch.zeros((1, _BLOCK), dtype=torch.float32, device=target_hidden.device)
    if sts_temperatures is None:
        sts_temperatures = torch.ones(_BLOCK, dtype=torch.float32, device=target_hidden.device)
    if steps_per_second is None:
        steps_per_second = torch.tensor(
            [0.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2],
            dtype=torch.float32,
            device=target_hidden.device,
        )

    io_binding = None
    if integrated_io:
        assert io_support is not None
        io_binding = io_support.Binding(
            torch,
            extension_module,
            workspace,
            accessor,
            bonus=bonus,
            commit_len=commit_len,
            new_seq_len=new_seq_len,
            target_hidden=target_hidden,
            freqs_real=freqs_real,
            anchor=anchor,
            main_hidden=main_hidden,
            rope=rope,
            start_pos=start_pos,
            epoch=epoch,
            request_index=request_index,
            candidates=candidates,
            base_bytes=base_bytes,
        )

    def prepare_once() -> None:
        if integrated_io:
            return
        prepare_proposal_megakernel(
            bonus=bonus,
            commit_len=commit_len,
            new_seq_len=new_seq_len,
            target_hidden=target_hidden,
            freqs_real=freqs_real,
            anchor=anchor,
            main_hidden=main_hidden,
            rope=rope,
            start_pos=start_pos,
            epoch=epoch,
        )
        accessor.gather(request_index, new_seq_len, expected_pos=start_pos)

    def propose_with_module(module: ModuleType | None) -> None:
        """Launch one scheduler against the owned workspace and accepted inputs."""
        if integrated_io:
            assert io_binding is not None
            if module is not extension_module:
                raise ValueError("An integrated workspace cannot be rebound to another module")
            io_binding.verify_host()
        run_v4_front_kv_stage(
            anchor,
            main_hidden,
            rope,
            packed.arena,
            packed.offsets,
            accessor.out,
            start_pos=-1,
            draft_layer_count=3,
            workspace=workspace,
            proposal_epoch=-1,
            main_output=main_output,
            embedding_output=embedding_output,
            proposal_outputs=outputs,
            uniforms=uniforms,
            sampling_temperature=sampling_temperature,
            sts_temperatures=sts_temperatures,
            steps_per_second=steps_per_second,
            relaxed_dag=True,
            greedy_tail=greedy_tail,
            full_loop_device_epoch=True,
            instrumented=instrumented,
            compiled_execution_mask=(
                V4_EXECUTE_ALL if os.environ.get("DSPARK_SPECIALIZE_ALL_PHASES") == "1" else None
            ),
            compiled_draft_layer_mask=(
                0x7 if os.environ.get("DSPARK_SPECIALIZE_ALL_PHASES") == "1" else None
            ),
            extension_module=module,
        )
        if not integrated_io:
            publish_proposal_candidates(outputs.output_ids, candidates)

    def propose_once() -> None:
        """Launch the initially selected scheduler module."""
        propose_with_module(extension_module)

    for _ in range(2):
        prepare_once()
        propose_once()
    torch.cuda.synchronize(target_hidden.device)
    if integrated_io:
        assert io_binding is not None
        io_binding.verify_resident()

    input_graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(input_graph):
        prepare_once()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        propose_once()
    if integrated_io and (
        captured_node_types(input_graph) != () or captured_node_types(graph) != (0,)
    ):
        raise RuntimeError("Integrated proposal must capture an empty input graph and one kernel")

    input_graph.replay()
    graph.replay()
    torch.cuda.synchronize(target_hidden.device)
    if integrated_io:
        assert io_binding is not None
        io_binding.verify_resident()
    if int(torch.count_nonzero(accessor.status).item()) != 0:
        status_counts = torch.bincount(accessor.status.flatten(), minlength=6).tolist()
        raise RuntimeError(
            "proposal megakernel KV gather failed its position/layout audit: "
            f"status_counts={status_counts}"
        )
    if bool(torch.any(candidates < 0).item()) or bool(torch.any(candidates >= 129280).item()):
        raise RuntimeError("proposal megakernel published an out-of-vocabulary token")

    band = CapturedProposalMegakernelBand(
        input_graph=input_graph,
        graph=graph,
        input_node_types=_require_device_launchable(input_graph, label="proposal input"),
        # Instrumentation adds counter-reset memset nodes. Its host-only
        # diagnostic bypasses device-graph composition; production stays strict.
        node_types=(
            captured_node_types(graph)
            if instrumented
            else _require_device_launchable(graph, label="proposal megakernel")
        ),
        uncaptured_proposal=propose_once,
        proposal_outputs=outputs,
        io_binding=io_binding,
        keepalive=(
            packed,
            accessor,
            workspace,
            outputs,
            anchor,
            main_hidden,
            rope,
            freqs_real,
            start_pos,
            epoch,
            request_index,
            main_output,
            embedding_output,
            uniforms,
            sts_temperatures,
            steps_per_second,
        ),
    )

    band_owner = weakref.ref(band)

    def recapture_proposal(module: ModuleType) -> CapturedProposalMegakernelBand:
        """Capture an ABI-compatible candidate using these exact tensor owners.

        This deliberately shares outputs. Callers must snapshot reference
        evidence before replaying another arm; comparing live views afterward
        would be vacuous. The caller must also preserve the packing layout.
        """
        if instrumented:
            raise ValueError("shared-workspace recapture excludes instrumentation")
        original_band = band_owner()
        if original_band is None:
            raise RuntimeError("the original proposal band has been released")

        def candidate_once() -> None:
            """Retain the candidate module while its native launch is captured."""
            propose_with_module(module)

        for _ in range(2):
            prepare_once()
            candidate_once()
        torch.cuda.synchronize(target_hidden.device)
        candidate_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(candidate_graph):
            candidate_once()
        return replace(
            original_band,
            graph=candidate_graph,
            node_types=_require_device_launchable(candidate_graph, label="proposal candidate"),
            uncaptured_proposal=candidate_once,
            recapture_proposal=None,
        )

    if extension_module is not None and not integrated_io:
        band.recapture_proposal = recapture_proposal
    return band


def capture_hybrid_proposal_band(
    snapshot: Path,
    *,
    graft: Any,
    target_hidden: torch.Tensor,
    bonus: torch.Tensor,
    commit_len: torch.Tensor,
    new_seq_len: torch.Tensor,
    candidates: torch.Tensor,
) -> CapturedProposalMegakernelBand:
    """Capture segmented persistent stages around exact SGLang FlashInfer MLPs.

    This is a diagnostic composition experiment, not the literal one-launch
    proposal megakernel.  It retains megakernel attention, HC, head, and tail
    bodies while replacing each complete routed-plus-shared expert branch with
    the exact graph-capturable SGLang/FlashInfer top-7 MLP leaf.
    """

    if target_hidden.shape != (6, 3 * 4096) or target_hidden.dtype != torch.bfloat16:
        raise ValueError("proposal target_hidden must be BF16 [6,12288]")
    if candidates.shape != (6,) or candidates.dtype != torch.int64:
        raise ValueError("proposal candidates must be int64 [6]")
    properties = torch.cuda.get_device_properties(target_hidden.device)
    if (properties.major, properties.minor) != (10, 3) or properties.multi_processor_count != 152:
        raise RuntimeError("the hybrid proposal requires a 152-SM GB300/sm_103")

    from scripts.megakernel.boltin_kv_accessor import bind_full_loop_graft_kv_accessor
    from scripts.megakernel.boltin_pack import pack_boltin_arena

    stages = tuple(graft.bundle.draft_model.stages)
    if len(stages) != 3 or any(not hasattr(stage, "mlp") for stage in stages):
        raise RuntimeError("the hybrid proposal requires exactly three SGLang draft MLPs")

    packed = pack_boltin_arena(snapshot, device=target_hidden.device)
    accessor = bind_full_loop_graft_kv_accessor(graft)
    workspace = allocate_v4_scheduler_workspace(
        device=target_hidden.device,
        extra_scratch_bytes=_DEVICE_SCALARS_BYTES,
    )
    start_pos = workspace.scratch[-16:-8].view(torch.int64)
    epoch = workspace.scratch[-8:].view(torch.int64)
    start_pos.fill_(1)
    epoch.zero_()
    outputs = allocate_v4_proposal_outputs(device=target_hidden.device)
    anchor = torch.empty(1, dtype=torch.int32, device=target_hidden.device)
    main_hidden = torch.empty((1, 1, 3 * 4096), dtype=torch.bfloat16, device=target_hidden.device)
    main_output = torch.empty((1, 1, 4096), dtype=torch.bfloat16, device=target_hidden.device)
    embedding_output = torch.empty(
        (1, _BLOCK, 4, 4096), dtype=torch.bfloat16, device=target_hidden.device
    )
    rope = torch.empty((_BLOCK + 1, 32, 2), dtype=torch.float32, device=target_hidden.device)
    freqs_real = torch.view_as_real(packed.freqs_cis)
    if not freqs_real.is_contiguous():
        freqs_real = freqs_real.contiguous()
    request_index = torch.zeros(1, dtype=torch.int64, device=target_hidden.device)
    uniforms = torch.zeros((1, _BLOCK), dtype=torch.float32, device=target_hidden.device)
    sts_temperatures = torch.ones(_BLOCK, dtype=torch.float32, device=target_hidden.device)
    steps_per_second = torch.tensor(
        [0.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2],
        dtype=torch.float32,
        device=target_hidden.device,
    )
    normalized = view_v4_workspace_region(workspace, "ffn_normalized_hidden").view(_BLOCK, 4096)
    residual = view_v4_workspace_region(workspace, "attention_hidden_streams").view(_BLOCK, 4, 4096)
    post = view_v4_workspace_region(workspace, "ffn_hc_post").view(_BLOCK, 4)
    comb = view_v4_workspace_region(workspace, "ffn_hc_comb").view(_BLOCK, 4, 4)
    hidden_streams = view_v4_workspace_region(workspace, "hidden_streams").view(_BLOCK, 4, 4096)
    expert_outputs: list[torch.Tensor | None] = [None, None, None]

    def prepare_once() -> None:
        prepare_proposal_megakernel(
            bonus=bonus,
            commit_len=commit_len,
            new_seq_len=new_seq_len,
            target_hidden=target_hidden,
            freqs_real=freqs_real,
            anchor=anchor,
            main_hidden=main_hidden,
            rope=rope,
            start_pos=start_pos,
            epoch=epoch,
        )
        accessor.gather(request_index, new_seq_len, expected_pos=start_pos)

    def persistent_segment(mask: int, layer_mask: int, proposal_epoch: int) -> None:
        run_v4_front_kv_stage(
            anchor,
            main_hidden,
            rope,
            packed.arena,
            packed.offsets,
            accessor.out,
            start_pos=-1,
            draft_layer_count=3,
            workspace=workspace,
            proposal_epoch=proposal_epoch,
            main_output=main_output,
            embedding_output=embedding_output,
            proposal_outputs=outputs,
            uniforms=uniforms,
            sampling_temperature=0.0,
            sts_temperatures=sts_temperatures,
            steps_per_second=steps_per_second,
            relaxed_dag=True,
            greedy_tail=True,
            execution_mask=mask,
            draft_layer_mask=layer_mask,
            compiled_execution_mask=mask,
            compiled_draft_layer_mask=layer_mask,
        )

    def propose_once() -> None:
        first_mask = V4_EXECUTE_MAIN | V4_EXECUTE_EMBEDDING | V4_EXECUTE_MAIN_KV | _DRAFT_BODY_MASK
        persistent_segment(first_mask, 0x1, 1)
        for layer, stage in enumerate(stages):
            if layer:
                persistent_segment(_DRAFT_BODY_MASK, 1 << layer, layer + 1)
            expert_output = stage.mlp.forward_normal(normalized)
            expert_outputs[layer] = expert_output
            expert_hc_post_cuda(
                expert_output,
                residual,
                post,
                comb,
                output=hidden_streams,
            )
        persistent_segment(V4_EXECUTE_HEAD | V4_EXECUTE_TAIL, 0x7, 4)
        publish_proposal_candidates(outputs.output_ids, candidates)

    for _ in range(2):
        prepare_once()
        propose_once()
    torch.cuda.synchronize(target_hidden.device)

    input_graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(input_graph):
        prepare_once()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        propose_once()

    input_graph.replay()
    graph.replay()
    torch.cuda.synchronize(target_hidden.device)
    if int(torch.count_nonzero(accessor.status).item()) != 0:
        status_counts = torch.bincount(accessor.status.flatten(), minlength=6).tolist()
        raise RuntimeError(
            "hybrid proposal KV gather failed its position/layout audit: "
            f"status_counts={status_counts}"
        )
    if bool(torch.any(candidates < 0).item()) or bool(torch.any(candidates >= 129280).item()):
        raise RuntimeError("hybrid proposal published an out-of-vocabulary token")

    return CapturedProposalMegakernelBand(
        input_graph=input_graph,
        graph=graph,
        input_node_types=_require_device_launchable(input_graph, label="hybrid input"),
        node_types=_require_device_launchable(graph, label="hybrid proposal"),
        keepalive=(
            packed,
            accessor,
            workspace,
            outputs,
            anchor,
            main_hidden,
            rope,
            freqs_real,
            start_pos,
            epoch,
            request_index,
            main_output,
            embedding_output,
            uniforms,
            sts_temperatures,
            steps_per_second,
            normalized,
            residual,
            post,
            comb,
            hidden_streams,
            stages,
            tuple(expert_outputs),
        ),
    )
