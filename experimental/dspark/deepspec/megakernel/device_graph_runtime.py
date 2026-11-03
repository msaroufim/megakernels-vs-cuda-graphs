from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@functools.lru_cache(maxsize=1)
def load_device_graph_runtime_extension():
    """Build the CUDA-13 bridge used by AOT-composed numerical graphs."""

    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="deepspec_dspark_device_graph_runtime_cuda13_v11",
        sources=[
            str(source / "dspark_device_graph_runtime.cpp"),
            str(source / "dspark_device_graph_runtime_kernel.cu"),
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


def append_tail_relaunch(iteration: torch.Tensor, *, iterations: int) -> None:
    """Append the compiler's commit/back-edge kernel during graph capture."""

    load_device_graph_runtime_extension().tail_marker(iteration, iterations)


@dataclass
class FullLoopStateBuffers:
    """Shared dynamic metadata emitted by the persistent-loop controller."""

    iteration: torch.Tensor
    prefix_len: torch.Tensor
    positions: torch.Tensor
    window_locations: torch.Tensor
    c4_plan_c: torch.Tensor
    c4_plan_w: torch.Tensor
    c128_plan_c: torch.Tensor
    c128_plan_w: torch.Tensor
    c4_target_out_locations: torch.Tensor
    c4_index_out_locations: torch.Tensor
    c4_index_context_lens: torch.Tensor
    c128_target_out_locations: torch.Tensor
    c128_extra_topk_lengths: torch.Tensor
    graft_seq_lens: torch.Tensor
    graft_positions: torch.Tensor
    graft_out_cache_loc: torch.Tensor

    @classmethod
    def allocate(cls, *, device: torch.device | str = "cuda") -> "FullLoopStateBuffers":
        return cls(
            iteration=torch.zeros(1, dtype=torch.int32, device=device),
            prefix_len=torch.empty(1, dtype=torch.int64, device=device),
            positions=torch.empty(6, dtype=torch.int32, device=device),
            window_locations=torch.empty(6, dtype=torch.int32, device=device),
            c4_plan_c=torch.empty((6, 16), dtype=torch.uint8, device=device),
            c4_plan_w=torch.empty((6, 8), dtype=torch.uint8, device=device),
            c128_plan_c=torch.empty((6, 16), dtype=torch.uint8, device=device),
            c128_plan_w=torch.empty((6, 8), dtype=torch.uint8, device=device),
            c4_target_out_locations=torch.empty(6, dtype=torch.int64, device=device),
            c4_index_out_locations=torch.empty(6, dtype=torch.int64, device=device),
            c4_index_context_lens=torch.empty((6, 1), dtype=torch.int32, device=device),
            c128_target_out_locations=torch.empty(6, dtype=torch.int64, device=device),
            c128_extra_topk_lengths=torch.empty(6, dtype=torch.int32, device=device),
            graft_seq_lens=torch.empty(1, dtype=torch.int64, device=device),
            graft_positions=torch.empty(5, dtype=torch.int64, device=device),
            graft_out_cache_loc=torch.empty(5, dtype=torch.int64, device=device),
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(getattr(self, field) for field in self.__dataclass_fields__)


def append_full_loop_state_update(
    new_seq_len: torch.Tensor,
    buffers: FullLoopStateBuffers,
    *,
    iterations: int = 0,
) -> None:
    """Advance metadata, optionally publishing a device-tail control probe."""

    load_device_graph_runtime_extension().full_loop_state_update(
        buffers.iteration,
        iterations,
        new_seq_len,
        *buffers.tensors()[1:],
    )


def prepare_proposal_megakernel(
    *,
    bonus: torch.Tensor,
    commit_len: torch.Tensor,
    new_seq_len: torch.Tensor,
    target_hidden: torch.Tensor,
    freqs_real: torch.Tensor,
    anchor: torch.Tensor,
    main_hidden: torch.Tensor,
    rope: torch.Tensor,
    start_pos: torch.Tensor,
    epoch: torch.Tensor,
) -> None:
    """Stage accepted state into the fixed-address proposal-megakernel ABI."""

    load_device_graph_runtime_extension().prepare_proposal_megakernel(
        bonus,
        commit_len,
        new_seq_len,
        target_hidden,
        freqs_real,
        anchor,
        main_hidden,
        rope,
        start_pos,
        epoch,
        os.environ.get("DSPARK_PREPARE_VECTOR_COPY") == "1",
    )


def publish_proposal_candidates(
    output_ids: torch.Tensor,
    candidates: torch.Tensor,
) -> None:
    """Publish anchor plus five draft IDs directly into the next verify round."""

    load_device_graph_runtime_extension().publish_proposal_candidates(
        output_ids,
        candidates,
    )


@dataclass
class DeviceGraphExecutable:
    """Own one executable assembled from captured numerical child graphs."""

    handle: int
    _closed: bool = False

    @classmethod
    def instantiate(cls, graph: torch.cuda.CUDAGraph) -> "DeviceGraphExecutable":
        handle = load_device_graph_runtime_extension().instantiate(graph.raw_cuda_graph())
        return cls(int(handle))

    @classmethod
    def compose(
        cls,
        graphs: tuple[torch.cuda.CUDAGraph, ...],
        *,
        device_launch: bool = True,
    ) -> "DeviceGraphExecutable":
        """Link ordered bands; retain device-tail eligibility unless explicitly disabled."""

        if not graphs:
            raise ValueError("composed CUDA graph requires at least one child")
        if not isinstance(device_launch, bool):
            raise TypeError("device_launch must be a bool")
        handles = [graph.raw_cuda_graph() for graph in graphs]
        extension = load_device_graph_runtime_extension()
        instantiate = (
            extension.instantiate_composed if device_launch else extension.instantiate_composed_host
        )
        handle = instantiate(handles)
        return cls(int(handle))

    def launch(self) -> None:
        if self._closed:
            raise RuntimeError("device graph executable is closed")
        load_device_graph_runtime_extension().launch(self.handle)

    def close(self) -> None:
        if not self._closed:
            load_device_graph_runtime_extension().destroy(self.handle)
            self._closed = True

    def __enter__(self) -> "DeviceGraphExecutable":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def captured_node_types(graph: torch.cuda.CUDAGraph) -> tuple[int, ...]:
    """Expose node kinds so device-launch restrictions fail closed in tests."""

    values = load_device_graph_runtime_extension().node_types(graph.raw_cuda_graph())
    return tuple(int(value) for value in values)
