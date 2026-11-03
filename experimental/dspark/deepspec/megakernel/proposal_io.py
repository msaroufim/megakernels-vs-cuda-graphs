"""Bind live proposal I/O to the experimental single-kernel V4 entry.

The original workspace ABI is unchanged. Two existing device scalars precede
an appended 160-byte Context. Its first 152 bytes are immutable; the final
arrival counter is owned by the kernel. Owners must never execute overlapping
invocations. Receipt checks run outside capture and timing.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path
from typing import Any

ABI = 0x4453494F000100A0
POINTERS = (
    "bonus",
    "commit_len",
    "new_seq_len",
    "target_hidden",
    "freqs_real",
    "swa0",
    "swa1",
    "swa2",
    "req_to_token",
    "full_to_swa",
    "req_pool_indices",
    "candidates",
    "status",
)
DIMENSIONS = (
    "pages",
    "req_rows",
    "req_cols",
    "full_locations",
    "allow_zero_locations",
    "freq_rows",
)


class Context(ct.Structure):
    """Exact fixed-width layout shared with dspark_proposal_io.cuh."""

    _fields_ = (
        [(name, ct.c_uint64) for name in POINTERS]
        + [(name, ct.c_int64) for name in DIMENSIONS]
        + [("arrivals", ct.c_uint64)]
    )


CONTEXT_BYTES = ct.sizeof(Context)
EXTRA_BYTES = 16 + CONTEXT_BYTES
assert CONTEXT_BYTES == 160 and ct.alignment(Context) == 8
assert getattr(Context, "arrivals").offset == 152
assert [getattr(Context, name).offset for name, _ in Context._fields_] == list(range(0, 160, 8))


def image_receipt(module: Any) -> dict[str, Any]:
    """Reject ordinary or incompatible DSOs before any integrated capture/launch."""
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str):
        raise ValueError("Integrated proposal requires an explicit native V4 extension")
    path = Path(filename).resolve()
    if not path.is_file() or not callable(getattr(module, "front_kv_stage", None)):
        raise ValueError("Integrated proposal requires an explicit native V4 extension")
    library = ct.CDLL(str(path))
    marker = getattr(library, "dspark_full_proposal_io_abi", None)
    if marker is None:
        raise ValueError("Missing integrated proposal I/O ABI marker")
    marker.argtypes, marker.restype = [], ct.c_uint64
    if marker() != ABI:
        raise ValueError("Unsupported integrated proposal I/O ABI")
    return {
        "binary": str(path),
        "binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "abi": ABI,
        "context_bytes": CONTEXT_BYTES,
        "context_alignment": ct.alignment(Context),
        "field_offsets": {name: getattr(Context, name).offset for name, _ in Context._fields_},
        "workers": 152,
        "threads": 256,
    }


def tensor_identity(value: Any) -> tuple[Any, ...]:
    """Record the exact view and underlying storage without reading GPU values."""
    storage = value.untyped_storage()
    return (
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
        value.storage_offset(),
        storage.data_ptr(),
        storage.nbytes(),
    )


def require_tensor(torch: Any, value: Any, dtype: Any, shape: tuple[int, ...], device: Any) -> None:
    """Require bounded, aligned contiguous CUDA storage on the selected device."""
    if (
        not value.is_cuda
        or value.device != device
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not value.is_contiguous()
    ):
        raise ValueError(f"Invalid proposal I/O tensor: expected {dtype} {shape} on {device}")
    pointer, _, _, _, _, _, storage, extent = tensor_identity(value)
    if pointer % value.element_size() or not storage <= pointer < storage + extent:
        raise ValueError("Misaligned or out-of-storage proposal I/O view")
    if pointer + value.numel() * value.element_size() > storage + extent:
        raise ValueError("Proposal I/O view exceeds its live owner")


class Binding:
    """Retain one private workspace, immutable live-input bindings and original image."""

    def __init__(
        self,
        torch: Any,
        module: Any,
        workspace: Any,
        accessor: Any,
        *,
        bonus: Any,
        commit_len: Any,
        new_seq_len: Any,
        target_hidden: Any,
        freqs_real: Any,
        anchor: Any,
        main_hidden: Any,
        rope: Any,
        start_pos: Any,
        epoch: Any,
        request_index: Any,
        candidates: Any,
        base_bytes: int,
        probe: Any | None = None,
    ):
        """Validate every owner and upload the new Context once before the first launch."""
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Bind proposal I/O outside CUDA capture")
        if getattr(workspace, "_proposal_io_binding", None) is not None:
            raise ValueError("Integrated I/O requires an unbound private workspace")
        self.torch, self.module, self.workspace, self.accessor = torch, module, workspace, accessor
        self.image = image_receipt(module)
        self.launch = module.front_kv_stage
        self.base_bytes = base_bytes
        self.probe = probe
        if probe is not None and (
            probe.workspace is not workspace
            or probe.module is not module
            or probe.base_bytes != base_bytes
        ):
            raise ValueError("Proposal observer must own this exact image and workspace")
        self.extra_bytes = EXTRA_BYTES + (probe.trailer_bytes if probe is not None else 0)
        self.inputs = dict(
            bonus=bonus,
            commit_len=commit_len,
            new_seq_len=new_seq_len,
            target_hidden=target_hidden,
            freqs_real=freqs_real,
            req_pool_indices=request_index,
            candidates=candidates,
        )
        self.other = dict(
            anchor=anchor,
            main_hidden=main_hidden,
            rope=rope,
            start_pos=start_pos,
            epoch=epoch,
            scratch=workspace.scratch,
            target_kv=accessor.out,
        )
        admission = accessor.admission
        if len(admission.buffers) != 3 or type(accessor.allow_zero_locations) is not bool:
            raise ValueError("Expected admitted three-layer SWA storage")
        device = target_hidden.device
        for name, dtype, shape in (
            ("bonus", torch.int64, (1,)),
            ("commit_len", torch.int32, (1,)),
            ("new_seq_len", torch.int64, (1,)),
            ("target_hidden", torch.bfloat16, (6, 12288)),
            ("req_pool_indices", torch.int64, (1,)),
            ("candidates", torch.int64, (6,)),
        ):
            require_tensor(torch, self.inputs[name], dtype, shape, device)
        if freqs_real.numel() % 64 or freqs_real.numel() < 384:
            raise ValueError("RoPE table must contain complete 64-float position rows")
        require_tensor(torch, freqs_real, torch.float32, tuple(freqs_real.shape), device)
        require_tensor(
            torch, workspace.scratch, torch.uint8, (base_bytes + self.extra_bytes,), device
        )
        for value, dtype, shape in (
            (anchor, torch.int32, (1,)),
            (main_hidden, torch.bfloat16, (1, 1, 4096 * 3)),
            (rope, torch.float32, (6, 32, 2)),
            (start_pos, torch.int64, (1,)),
            (epoch, torch.int64, (1,)),
            (accessor.out, torch.bfloat16, (3, 1, 128, 512)),
            (accessor.status, torch.int32, (3, 128)),
        ):
            require_tensor(torch, value, dtype, shape, device)
        if any(value.data_ptr() % 16 for value in (target_hidden, main_hidden, freqs_real, rope)):
            raise ValueError("Proposal vector-copy operands require 16-byte alignment")
        self.dimensions = self.live_dimensions()
        for name, value in self.live_tensors().items():
            if name.startswith("swa"):
                require_tensor(
                    torch, value, torch.uint8, (self.dimensions["pages"], 149760), device
                )
        require_tensor(
            torch,
            admission.req_to_token,
            torch.int32,
            (self.dimensions["req_rows"], self.dimensions["req_cols"]),
            device,
        )
        require_tensor(
            torch, admission.full_to_swa, torch.int64, (self.dimensions["full_locations"],), device
        )
        base = workspace.scratch.data_ptr() + base_bytes
        if base_bytes % 8 or start_pos.data_ptr() != base or epoch.data_ptr() != base + 8:
            raise ValueError("Device scalars moved from the frozen workspace tail")
        self.context = workspace.scratch.narrow(0, base_bytes + 16, CONTEXT_BYTES)
        self.context_identity = tensor_identity(self.context)
        tensors = self.live_tensors()
        writable = {k: tensors[k] for k in ("candidates", "status")}
        writable.update(
            {k: self.other[k] for k in ("anchor", "main_hidden", "rope", "scratch", "target_kv")}
        )
        # Read/read sharing is legal. Every writable allocation is private and
        # disjoint from other I/O, including the preserved original SWA pool.
        spans = {
            name: (v.data_ptr(), v.data_ptr() + v.numel() * v.element_size())
            for name, v in {**tensors, **writable}.items()
        }
        for name in writable:
            begin, end = spans[name]
            for other, (a, b) in spans.items():
                if name != other and begin < b and a < end:
                    raise ValueError(f"Writable proposal I/O alias: {name}/{other}")
        self.expected_owners = {**tensors, **self.other}
        self.identities = {
            name: tensor_identity(value) for name, value in self.expected_owners.items()
        }
        context = Context(
            **{name: value.data_ptr() for name, value in tensors.items()},
            **self.dimensions,
            arrivals=0,
        )
        self.expected = bytes(context)[:152]
        if int(epoch.item()) != 0:
            raise ValueError("Integrated I/O requires a fresh zero epoch")
        encoded = torch.frombuffer(bytearray(bytes(context)), dtype=torch.uint8)
        self.context.copy_(encoded.to(device))
        workspace._proposal_io_binding = self
        self.verify_resident()

    def live_tensors(self) -> dict[str, Any]:
        """Resolve accessor owners again so replacing its fields cannot evade checks."""
        admission = self.accessor.admission
        return {
            **self.inputs,
            **{f"swa{i}": value for i, value in enumerate(admission.buffers)},
            "req_to_token": admission.req_to_token,
            "full_to_swa": admission.full_to_swa,
            "status": self.accessor.status,
        }

    def live_dimensions(self) -> dict[str, int]:
        """Derive exact table bounds from the retained admission, never supplied integers."""
        admission = self.accessor.admission
        result = {
            "pages": admission.pages,
            "req_rows": admission.req_to_token.shape[0],
            "req_cols": admission.req_to_token.shape[1],
            "full_locations": admission.full_to_swa.numel(),
            "allow_zero_locations": int(self.accessor.allow_zero_locations),
            "freq_rows": self.inputs["freqs_real"].numel() // 64,
        }
        if any(
            type(v) is not int or not 0 < v < 2**63
            for k, v in result.items()
            if k != "allow_zero_locations"
        ):
            raise ValueError("Invalid proposal I/O table bounds")
        return result

    def verify_host(self) -> None:
        """Check retained objects, views and dimensions without synchronization."""
        if self.probe is not None:
            self.probe.verify_host()
        if (
            self.module.front_kv_stage is not self.launch
            or self.workspace._proposal_io_binding is not self
            or tensor_identity(self.context) != self.context_identity
        ):
            raise ValueError("Proposal launch or private binding changed")
        current = {
            **self.live_tensors(),
            **self.other,
            "scratch": self.workspace.scratch,
            "target_kv": self.accessor.out,
        }
        if (
            current.keys() != self.expected_owners.keys()
            or self.live_dimensions() != self.dimensions
        ):
            raise ValueError("Proposal I/O owner layout changed")
        for name, value in current.items():
            if (
                value is not self.expected_owners[name]
                or tensor_identity(value) != self.identities[name]
            ):
                raise ValueError(f"Proposal I/O owner/view changed: {name}")

    def verify_resident(self) -> dict[str, int]:
        """Check immutable Context and completed-grid epoch accounting outside capture."""
        if self.torch.cuda.is_current_stream_capturing():
            raise ValueError("Read proposal I/O receipt outside CUDA capture")
        self.verify_host()
        raw = bytes(self.context.cpu().tolist())
        epoch = int(self.other["epoch"].item())
        arrivals = int.from_bytes(raw[152:], "little")
        if raw[:152] != self.expected or not 0 <= epoch < 0xFFFFFFFF or arrivals != epoch * 152:
            raise ValueError("Changed Context or incomplete/overlapping proposal I/O epoch")
        return {"epoch": epoch, "arrivals": arrivals}

    def record(self) -> dict[str, Any]:
        """Expose auditable immutable bindings and current completed invocation counters."""
        return {
            **self.image,
            "workspace_base_bytes": self.base_bytes,
            "context_offset": self.base_bytes + 16,
            "extra_bytes": self.extra_bytes,
            "observer": self.probe.record() if self.probe is not None else None,
            "immutable_context_hex": self.expected.hex(),
            "dimensions": self.dimensions,
            "owners": {name: list(value) for name, value in self.identities.items()},
            "completed": self.verify_resident(),
            "scope": "One private workspace, physically nonoverlapping invocations",
        }
