import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.megakernel.boltin_kv_accessor import (
    EXPECTED_SGLANG_VERSION,
    HEAD_DIM,
    LAYERS,
    PAGE_BYTES,
    PAGE_TOKENS,
    RING_ROWS,
    BoltinKVAccessor,
    BoltinKVAdmission,
    BoltinKVAdmissionError,
    inspect_boltin_kv_admission,
    logical_ring_position,
)

REPO = Path(__file__).resolve().parents[1]


class _FakeCudaTensor:
    def __init__(self, shape, dtype, *, device="cuda:0", contiguous=True):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = True
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


def _injector(
    *,
    unified=False,
    page_bytes=PAGE_BYTES,
    buffer_count=LAYERS,
    pool_size=4 * PAGE_TOKENS - PAGE_TOKENS - 1,
):
    buffers = [_FakeCudaTensor((4, page_bytes), torch.uint8) for _ in range(buffer_count)]
    swa_pool = SimpleNamespace(
        size=pool_size,
        page_size=PAGE_TOKENS,
        quantize_block_size=64,
        scale_pad=1,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        kv_cache_total_dim=584,
        bytes_per_page_padded=page_bytes,
        store_dtype=torch.uint8,
        kv_buffer=buffers,
    )
    pool = SimpleNamespace(
        _unified_kv=unified,
        unified_kv_pool=object() if unified else None,
        sliding_window=RING_ROWS,
        swa_page_size=PAGE_TOKENS,
        swa_kv_pool=swa_pool,
        full_to_swa_index_mapping=_FakeCudaTensor((4096,), torch.int64),
    )
    return SimpleNamespace(
        draft_model_runner=SimpleNamespace(token_to_kv_pool=pool),
        model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=_FakeCudaTensor((4, 512), torch.int32))
        ),
    )


@pytest.fixture
def fake_b200(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(name="NVIDIA B200", multi_processor_count=148),
    )


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (1, [0, -127, -126, -1]),
        (127, [0, 1, 2, -1]),
        (128, [0, 1, 2, 127]),
        (129, [128, 1, 2, 127]),
        (255, [128, 129, 130, 127]),
        (256, [128, 129, 130, 255]),
        (257, [256, 129, 130, 255]),
    ],
)
def test_ring_ordinal_matches_latest_logical_position(length, expected):
    offsets = (0, 1, 2, 127)
    actual = [logical_ring_position(length - 1, offset) for offset in offsets]
    assert actual == expected
    for offset, logical in enumerate(
        logical_ring_position(length - 1, value) for value in range(RING_ROWS)
    ):
        assert logical <= length - 1
        assert logical % RING_ROWS == offset


def test_ring_ordinal_rejects_invalid_offset():
    with pytest.raises(ValueError, match="ring_offset"):
        logical_ring_position(0, -1)
    with pytest.raises(ValueError, match="ring_offset"):
        logical_ring_position(0, RING_ROWS)


def test_exact_pinned_admission_accepts_only_expected_surface(fake_b200):
    admission = inspect_boltin_kv_admission(
        _injector(),
        tp_rank=0,
        batch_size=1,
        sglang_version=EXPECTED_SGLANG_VERSION,
    )
    assert admission.device == torch.device("cuda:0")
    assert admission.pages == 4
    assert len(admission.buffers) == LAYERS
    assert admission.req_to_token.dtype == torch.int32
    assert admission.full_to_swa.dtype == torch.int64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tp_rank": 1, "batch_size": 1}, "rank-0-only"),
        ({"tp_rank": 0, "batch_size": 2}, "requires bs=1"),
    ],
)
def test_admission_rejects_rank_and_batch(fake_b200, kwargs, message):
    with pytest.raises(BoltinKVAdmissionError, match=message):
        inspect_boltin_kv_admission(
            _injector(),
            sglang_version=EXPECTED_SGLANG_VERSION,
            **kwargs,
        )


def test_admission_rejects_version_unified_and_layout_drift(fake_b200):
    with pytest.raises(BoltinKVAdmissionError, match="unsupported"):
        inspect_boltin_kv_admission(_injector(), tp_rank=0, batch_size=1, sglang_version="0.5.17")
    with pytest.raises(BoltinKVAdmissionError, match="unified"):
        inspect_boltin_kv_admission(
            _injector(unified=True),
            tp_rank=0,
            batch_size=1,
            sglang_version=EXPECTED_SGLANG_VERSION,
        )
    with pytest.raises(BoltinKVAdmissionError, match="149760"):
        inspect_boltin_kv_admission(
            _injector(page_bytes=PAGE_BYTES - 1),
            tp_rank=0,
            batch_size=1,
            sglang_version=EXPECTED_SGLANG_VERSION,
        )
    with pytest.raises(BoltinKVAdmissionError, match="exactly 3"):
        inspect_boltin_kv_admission(
            _injector(buffer_count=2),
            tp_rank=0,
            batch_size=1,
            sglang_version=EXPECTED_SGLANG_VERSION,
        )
    with pytest.raises(BoltinKVAdmissionError, match="expected"):
        inspect_boltin_kv_admission(
            _injector(pool_size=PAGE_TOKENS),
            tp_rank=0,
            batch_size=1,
            sglang_version=EXPECTED_SGLANG_VERSION,
        )


def test_admission_rejects_non_b200(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(name="NVIDIA H100", multi_processor_count=132),
    )
    with pytest.raises(BoltinKVAdmissionError, match="B200"):
        inspect_boltin_kv_admission(
            _injector(),
            tp_rank=0,
            batch_size=1,
            sglang_version=EXPECTED_SGLANG_VERSION,
        )


def test_runtime_gather_forwards_preallocated_tensors_without_readback():
    calls = []

    class Extension:
        @staticmethod
        def gather_target_kv(*args):
            calls.append(args)

    buffers = (object(), object(), object())
    req_to_token = object()
    full_to_swa = object()
    admission = BoltinKVAdmission(
        buffers=buffers,
        req_to_token=req_to_token,
        full_to_swa=full_to_swa,
        device=torch.device("cpu"),
        pages=1,
    )
    out = torch.empty(LAYERS, 1, RING_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    status = torch.empty(LAYERS, RING_ROWS, dtype=torch.int32)
    staged_req_index = torch.empty(1, dtype=torch.int64)
    staged_seq_len = torch.empty(1, dtype=torch.int64)
    accessor = BoltinKVAccessor(
        Extension(), admission, out, status, staged_req_index, staged_seq_len
    )
    req_pool_indices = torch.tensor([1], dtype=torch.int64)
    seq_lens = torch.tensor([257], dtype=torch.int64)

    returned = accessor.gather(req_pool_indices, seq_lens)

    assert returned == (out, status)
    assert len(calls) == 1
    assert calls[0] == (
        *buffers,
        req_to_token,
        full_to_swa,
        staged_req_index,
        staged_seq_len,
        None,
        out,
        status,
        False,
    )
    assert staged_req_index.tolist() == req_pool_indices.tolist()
    assert staged_seq_len.tolist() == seq_lens.tolist()
    source = inspect.getsource(BoltinKVAccessor.gather)
    assert ".item(" not in source
    assert "synchronize" not in source
    assert "torch.empty" not in source
    assert "torch.zeros" not in source


def test_cuda_source_has_one_exact_launch_and_every_status_branch():
    source = (REPO / "deepspec/megakernel/csrc/dspark_boltin_kv.cu").read_text()
    assert source.count("gather_target_kv_kernel<<<") == 1
    assert "dim3(kLayers, kRingRows), 256" in source
    assert "kPageBytes = 149760" in source
    assert "kScaleSectionOffset = kPageTokens * kTokenDataBytes" in source
    assert "last - ((last - ring_offset) & (kRingRows - 1))" in source
    assert "full_location < (allow_zero_locations ? 0 : 1)" in source
    assert "swa_location < (allow_zero_locations ? 0 : 1)" in source
    assert "bool allow_zero_locations" in source
    for status in (
        "kInvalidRequest",
        "kLogicalOutOfBounds",
        "kInvalidFullLocation",
        "kInvalidSwaLocation",
    ):
        assert status in source
