"""Fail-closed access to SGLang v0.5.16's authoritative DSpark SWA KV.

The hot path is deliberately small: ``BoltinKVAccessor.gather`` forwards nine
already-bound tensors to one CUDA launch.  It performs no allocation, scalar
readback, synchronization, indexing, or layout conversion.  All structural
checks and the two output allocations happen once in :meth:`bind`.

This module is separate from the production bolt-in glue until the standalone
B200 oracle and latency lane has passed.  The admitted layout is the pinned
DeepSeek-V4 layout only; an unfamiliar SGLang version, GPU, pool variant, or
tensor metadata raises :class:`BoltinKVAdmissionError` instead of guessing.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

EXPECTED_SGLANG_VERSION = "0.5.16"
LAYERS = 3
RING_ROWS = 128
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK = 64
SCALE_PAD = 1
PAGE_TOKENS = 256
TOKEN_DATA_BYTES = 576
SCALE_BYTES = 8
PAGE_BYTES = 149_760

STATUS_OK = 0
STATUS_INVALID_REQUEST = 1
STATUS_LOGICAL_OUT_OF_BOUNDS = 2
STATUS_INVALID_FULL_LOCATION = 3
STATUS_INVALID_SWA_LOCATION = 4


class BoltinKVAdmissionError(RuntimeError):
    """The live SGLang objects do not match the pinned accessor contract."""


def logical_ring_position(last_position: int, ring_offset: int) -> int:
    """Latest logical position at or before ``last_position`` for one ring row."""

    if not 0 <= ring_offset < RING_ROWS:
        raise ValueError(f"ring_offset must be in [0,{RING_ROWS}), got {ring_offset}")
    return last_position - ((last_position - ring_offset) & (RING_ROWS - 1))


def _attribute(owner: Any, name: str, owner_name: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError as error:
        raise BoltinKVAdmissionError(f"{owner_name} has no {name!r}") from error


def _integer_attribute(owner: Any, name: str, expected: int, owner_name: str) -> None:
    value = _attribute(owner, name, owner_name)
    if isinstance(value, bool) or value != expected:
        raise BoltinKVAdmissionError(f"{owner_name}.{name}={value!r}, expected exactly {expected}")


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise BoltinKVAdmissionError("object is not tensor-shaped") from error


def _check_tensor(
    tensor: Any,
    *,
    name: str,
    dtype: torch.dtype,
    ndim: int,
    device: torch.device | None = None,
) -> tuple[int, ...]:
    shape = _shape(tensor)
    if len(shape) != ndim:
        raise BoltinKVAdmissionError(f"{name} must be {ndim}D, got shape {shape}")
    if getattr(tensor, "dtype", None) != dtype:
        raise BoltinKVAdmissionError(
            f"{name} must have dtype {dtype}, got {getattr(tensor, 'dtype', None)}"
        )
    tensor_device = torch.device(getattr(tensor, "device", "cpu"))
    if tensor_device.type != "cuda" or not bool(getattr(tensor, "is_cuda", False)):
        raise BoltinKVAdmissionError(f"{name} must be a CUDA tensor")
    if device is not None and tensor_device != device:
        raise BoltinKVAdmissionError(
            f"{name} is on {tensor_device}, expected bound device {device}"
        )
    try:
        contiguous = bool(tensor.is_contiguous())
    except AttributeError as error:
        raise BoltinKVAdmissionError(f"{name} has no contiguity metadata") from error
    if not contiguous:
        raise BoltinKVAdmissionError(f"{name} must be contiguous")
    return shape


def _installed_sglang_version() -> str:
    try:
        import sglang
    except ImportError as error:
        raise BoltinKVAdmissionError("SGLang is not importable") from error
    return str(getattr(sglang, "__version__", "unknown"))


@dataclass(frozen=True)
class BoltinKVAdmission:
    """Resolved immutable tensor references for an admitted injector."""

    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    req_to_token: torch.Tensor
    full_to_swa: torch.Tensor
    device: torch.device
    pages: int


def inspect_boltin_kv_admission(
    injector: Any,
    *,
    tp_rank: int,
    batch_size: int,
    sglang_version: str | None = None,
) -> BoltinKVAdmission:
    """Resolve and validate the exact pinned SGLang target-KV surface.

    ``sglang_version`` is an explicit seam for CPU unit tests.  Production
    callers omit it, so the imported package version is always checked.
    """

    actual_version = _installed_sglang_version() if sglang_version is None else sglang_version
    if actual_version != EXPECTED_SGLANG_VERSION:
        raise BoltinKVAdmissionError(
            f"SGLang {actual_version!r} is unsupported; expected {EXPECTED_SGLANG_VERSION!r}"
        )
    if isinstance(tp_rank, bool) or tp_rank != 0:
        raise BoltinKVAdmissionError(f"authoritative KV access is rank-0-only, got {tp_rank=}")
    if isinstance(batch_size, bool) or batch_size != 1:
        raise BoltinKVAdmissionError(f"authoritative KV access requires bs=1, got {batch_size=}")

    draft_runner = _attribute(injector, "draft_model_runner", "injector")
    pool = _attribute(draft_runner, "token_to_kv_pool", "injector.draft_model_runner")
    if _attribute(pool, "_unified_kv", "draft token_to_kv_pool") is not False:
        raise BoltinKVAdmissionError("unified DeepSeek-V4 KV pools are not supported")
    if getattr(pool, "unified_kv_pool", None) is not None:
        raise BoltinKVAdmissionError("unified DeepSeek-V4 KV storage must be absent")
    _integer_attribute(pool, "sliding_window", RING_ROWS, "draft token_to_kv_pool")
    _integer_attribute(pool, "swa_page_size", PAGE_TOKENS, "draft token_to_kv_pool")

    swa_pool = _attribute(pool, "swa_kv_pool", "draft token_to_kv_pool")
    if swa_pool is None:
        raise BoltinKVAdmissionError("draft token_to_kv_pool.swa_kv_pool is absent")
    _integer_attribute(swa_pool, "page_size", PAGE_TOKENS, "swa_kv_pool")
    _integer_attribute(swa_pool, "quantize_block_size", QUANT_BLOCK, "swa_kv_pool")
    _integer_attribute(swa_pool, "scale_pad", SCALE_PAD, "swa_kv_pool")
    _integer_attribute(swa_pool, "qk_nope_head_dim", NOPE_DIM, "swa_kv_pool")
    _integer_attribute(swa_pool, "qk_rope_head_dim", ROPE_DIM, "swa_kv_pool")
    _integer_attribute(swa_pool, "kv_cache_total_dim", 584, "swa_kv_pool")
    _integer_attribute(swa_pool, "bytes_per_page_padded", PAGE_BYTES, "swa_kv_pool")
    if _attribute(swa_pool, "store_dtype", "swa_kv_pool") != torch.uint8:
        raise BoltinKVAdmissionError("swa_kv_pool.store_dtype must be torch.uint8")

    raw_buffers = _attribute(swa_pool, "kv_buffer", "swa_kv_pool")
    if not isinstance(raw_buffers, (list, tuple)) or len(raw_buffers) != LAYERS:
        raise BoltinKVAdmissionError(f"swa_kv_pool.kv_buffer must contain exactly {LAYERS} layers")
    first = raw_buffers[0]
    first_shape = _check_tensor(first, name="swa0", dtype=torch.uint8, ndim=2)
    if first_shape[0] <= 0 or first_shape[1] != PAGE_BYTES:
        raise BoltinKVAdmissionError(
            f"swa0 must have shape [P,{PAGE_BYTES}] with P>0, got {first_shape}"
        )
    size = _attribute(swa_pool, "size", "swa_kv_pool")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise BoltinKVAdmissionError(f"swa_kv_pool.size must be a positive int, got {size!r}")
    expected_pages = (size + PAGE_TOKENS + 1) // PAGE_TOKENS
    if first_shape[0] != expected_pages:
        raise BoltinKVAdmissionError(
            f"swa0 has {first_shape[0]} pages, expected {expected_pages} for size={size}"
        )
    device = torch.device(first.device)
    buffers = tuple(raw_buffers)
    for layer, buffer in enumerate(buffers[1:], start=1):
        shape = _check_tensor(
            buffer,
            name=f"swa{layer}",
            dtype=torch.uint8,
            ndim=2,
            device=device,
        )
        if shape != first_shape:
            raise BoltinKVAdmissionError(
                f"swa{layer} shape {shape} does not match swa0 {first_shape}"
            )

    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    properties = torch.cuda.get_device_properties(index)
    device_name = str(properties.name)
    sms = int(properties.multi_processor_count)
    if capability != (10, 0) or "B200" not in device_name.upper() or sms != 148:
        raise BoltinKVAdmissionError(
            "authoritative KV access requires a 148-SM B200/sm_100; "
            f"got {device_name!r}, capability={capability}, sms={sms}"
        )

    model_runner = _attribute(injector, "model_runner", "injector")
    req_pool = _attribute(model_runner, "req_to_token_pool", "injector.model_runner")
    req_to_token = _attribute(req_pool, "req_to_token", "target req_to_token_pool")
    req_shape = _check_tensor(
        req_to_token,
        name="req_to_token",
        dtype=torch.int32,
        ndim=2,
        device=device,
    )
    if req_shape[0] <= 0 or req_shape[1] <= 0:
        raise BoltinKVAdmissionError(
            f"req_to_token must have positive [R,C] dimensions, got {req_shape}"
        )

    full_to_swa = _attribute(pool, "full_to_swa_index_mapping", "draft token_to_kv_pool")
    full_shape = _check_tensor(
        full_to_swa,
        name="full_to_swa",
        dtype=torch.int64,
        ndim=1,
        device=device,
    )
    if full_shape[0] <= 1:
        raise BoltinKVAdmissionError(
            f"full_to_swa must reserve null index 0 and contain live slots, got {full_shape}"
        )

    return BoltinKVAdmission(
        buffers=(buffers[0], buffers[1], buffers[2]),
        req_to_token=req_to_token,
        full_to_swa=full_to_swa,
        device=device,
        pages=first_shape[0],
    )


@functools.lru_cache(maxsize=1)
def load_boltin_kv_extension():
    """JIT the isolated one-launch accessor extension."""

    source_dir = Path(__file__).resolve().parents[2] / "deepspec" / "megakernel" / "csrc"
    return load(
        name="deepspec_dspark_boltin_kv_v1",
        sources=[
            str(source_dir / "dspark_boltin_kv.cpp"),
            str(source_dir / "dspark_boltin_kv.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--std=c++17"],
        with_cuda=True,
        verbose=False,
    )


@dataclass
class BoltinKVAccessor:
    """Pre-bound, allocation-free runtime accessor."""

    extension: Any
    admission: BoltinKVAdmission
    out: torch.Tensor
    status: torch.Tensor
    staged_req_index: torch.Tensor
    staged_seq_len: torch.Tensor
    allow_zero_locations: bool = False

    @classmethod
    def from_admission(
        cls,
        admission: BoltinKVAdmission,
        *,
        extension: Any | None = None,
        allow_zero_locations: bool = False,
    ) -> BoltinKVAccessor:
        module = load_boltin_kv_extension() if extension is None else extension
        if not callable(getattr(module, "gather_target_kv", None)):
            raise BoltinKVAdmissionError("accessor extension has no gather_target_kv entry point")
        return cls(
            extension=module,
            admission=admission,
            out=torch.empty(
                LAYERS,
                1,
                RING_ROWS,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=admission.device,
            ),
            status=torch.empty(
                LAYERS,
                RING_ROWS,
                dtype=torch.int32,
                device=admission.device,
            ),
            staged_req_index=torch.empty(1, dtype=torch.int64, device=admission.device),
            staged_seq_len=torch.empty(1, dtype=torch.int64, device=admission.device),
            allow_zero_locations=allow_zero_locations,
        )

    @classmethod
    def bind(
        cls,
        injector: Any,
        *,
        tp_rank: int,
        batch_size: int,
        extension: Any | None = None,
        sglang_version: str | None = None,
    ) -> BoltinKVAccessor:
        admission = inspect_boltin_kv_admission(
            injector,
            tp_rank=tp_rank,
            batch_size=batch_size,
            sglang_version=sglang_version,
        )
        return cls.from_admission(admission, extension=extension)

    def gather(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        expected_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enqueue one gather/dequant launch and return the reusable outputs.

        The batch tensors are staged through persistent int64 [1] device
        tensors with fixed-shape ``copy_`` so any integer dtype SGLang uses is
        accepted without a per-step allocation or a host synchronization.
        ``expected_pos`` (int64 [1], device) is the commit-tracked position;
        when given, the kernel poisons every status word unless it equals
        ``seq_lens - 1``, so downstream consumers fail closed on a divergence
        between the two position sources.
        """

        if req_pool_indices.numel() != 1 or seq_lens.numel() != 1:
            raise RuntimeError(
                "authoritative KV gather requires single-request batch tensors, got "
                f"req_pool_indices.numel()={req_pool_indices.numel()} "
                f"seq_lens.numel()={seq_lens.numel()}"
            )
        self.staged_req_index.copy_(req_pool_indices.reshape(1))
        self.staged_seq_len.copy_(seq_lens.reshape(1))
        buffers = self.admission.buffers
        self.extension.gather_target_kv(
            buffers[0],
            buffers[1],
            buffers[2],
            self.admission.req_to_token,
            self.admission.full_to_swa,
            self.staged_req_index,
            self.staged_seq_len,
            expected_pos,
            self.out,
            self.status,
            self.allow_zero_locations,
        )
        return self.out, self.status


def bind_boltin_kv_accessor(
    injector: Any,
    *,
    tp_rank: int,
    batch_size: int,
) -> BoltinKVAccessor:
    """Production convenience wrapper with no test seams exposed."""

    return BoltinKVAccessor.bind(
        injector,
        tp_rank=tp_rank,
        batch_size=batch_size,
    )


def bind_full_loop_graft_kv_accessor(graft: Any) -> BoltinKVAccessor:
    """Bind the standalone GB300 full-loop graft pool without server objects.

    The full-loop experiment owns the request table and the three-layer draft
    pool directly, so it does not have SGLang's target-side injector wrapper.
    Keep this constructor explicit instead of weakening the production B200
    admission checks used by :func:`bind_boltin_kv_accessor`.
    """

    runner = _attribute(graft.bundle, "draft_model_runner", "graft.bundle")
    pool = _attribute(runner, "token_to_kv_pool", "graft draft_model_runner")
    swa_pool = _attribute(pool, "swa_kv_pool", "graft token_to_kv_pool")
    raw_buffers = _attribute(swa_pool, "kv_buffer", "graft swa_kv_pool")
    if not isinstance(raw_buffers, (list, tuple)) or len(raw_buffers) != LAYERS:
        raise BoltinKVAdmissionError("full-loop graft must expose exactly three SWA buffers")
    first_shape = _check_tensor(raw_buffers[0], name="swa0", dtype=torch.uint8, ndim=2)
    if first_shape[1] != PAGE_BYTES:
        raise BoltinKVAdmissionError(
            f"full-loop swa0 page width is {first_shape[1]}, expected {PAGE_BYTES}"
        )
    device = torch.device(raw_buffers[0].device)
    buffers = tuple(raw_buffers)
    for layer, buffer in enumerate(buffers[1:], start=1):
        shape = _check_tensor(
            buffer,
            name=f"swa{layer}",
            dtype=torch.uint8,
            ndim=2,
            device=device,
        )
        if shape != first_shape:
            raise BoltinKVAdmissionError(f"swa{layer} shape {shape} != {first_shape}")
    req_pool = _attribute(runner, "req_to_token_pool", "graft draft_model_runner")
    req_to_token = _attribute(req_pool, "req_to_token", "graft req_to_token_pool")
    _check_tensor(
        req_to_token,
        name="req_to_token",
        dtype=torch.int32,
        ndim=2,
        device=device,
    )
    full_to_swa = _attribute(pool, "full_to_swa_index_mapping", "graft token_to_kv_pool")
    _check_tensor(
        full_to_swa,
        name="full_to_swa",
        dtype=torch.int64,
        ndim=1,
        device=device,
    )
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (10, 3) or properties.multi_processor_count != 152:
        raise BoltinKVAdmissionError("full-loop proposal megakernel requires a 152-SM GB300/sm_103")
    admission = BoltinKVAdmission(
        buffers=(buffers[0], buffers[1], buffers[2]),
        req_to_token=req_to_token,
        full_to_swa=full_to_swa,
        device=device,
        pages=first_shape[0],
    )
    return BoltinKVAccessor.from_admission(admission, allow_zero_locations=True)
