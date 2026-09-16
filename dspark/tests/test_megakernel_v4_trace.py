import json

import pytest
import torch

from deepspec.megakernel import (
    DeepSeekV4MegaKernelSpec,
    V4DeviceTrace,
    V4LaunchShape,
    V4TraceFlag,
    V4WorkerRole,
    build_v4_phase_program,
)


def _program():
    return build_v4_phase_program(DeepSeekV4MegaKernelSpec(), V4LaunchShape())


def _synthetic_trace() -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.zeros(2, 4, 8, dtype=torch.int64)
    counts = torch.tensor([3, 2], dtype=torch.int32)
    # phase, role, segment, start, end, work item, ticket, flags
    raw[0, 0] = torch.tensor([0, 0, 1, 100, 110, 0, 1, 0])
    raw[0, 1] = torch.tensor([0, 2, 0, 100, 200, 7, 2, int(V4TraceFlag.LAST_TILE)])
    raw[0, 2] = torch.tensor([1, 2, 4, 200, 220, 0, 3, 0])
    raw[1, 0] = torch.tensor([0, 2, 0, 110, 190, 8, 4, int(V4TraceFlag.STOLEN)])
    raw[1, 1] = torch.tensor([1, 3, 3, 100, 220, 0, 5, 0])
    return raw, counts


def test_v4_trace_decodes_role_weighted_capacity_and_steals():
    raw, counts = _synthetic_trace()
    trace = V4DeviceTrace.from_tensors(raw, counts, _program())
    summary = trace.summarize(top_level_kernel_launches=1)
    assert summary.events == 5
    assert summary.kernel_span_ns == 120
    assert summary.warp_capacity_ns == 120 * 2 * 8
    assert summary.useful_warp_ns == 720
    assert summary.controller_warp_ns == 10
    assert summary.role_wait_warp_ns == 80
    assert summary.queue_empty_wait_warp_ns == 240
    assert summary.unaccounted_warp_ns == 870
    assert summary.top_level_kernel_launches == 1
    assert summary.inter_launch_gap_ns == 0
    phase_zero = trace.phase_summaries()[0]
    assert phase_zero.work_items == 2
    assert phase_zero.stolen_work_items == 1
    assert trace.phase_spans_ns() == {
        "main.activation_quant": 100,
        "main.projection": 120,
    }


def test_v4_trace_rejects_overlapping_role_timeline():
    raw, counts = _synthetic_trace()
    raw[0, 2, 3] = 199
    with pytest.raises(ValueError, match="overlapping role timeline"):
        V4DeviceTrace.from_tensors(raw, counts, _program())


def test_v4_trace_rejects_overflow_and_unknown_phase():
    raw, counts = _synthetic_trace()
    counts[0] = 5
    with pytest.raises(ValueError, match="exceeds capacity"):
        V4DeviceTrace.from_tensors(raw, counts, _program())
    counts[0] = 3
    raw[0, 0, 0] = 9999
    with pytest.raises(ValueError, match="unknown V4 phase"):
        V4DeviceTrace.from_tensors(raw, counts, _program())


def test_v4_trace_writes_json_and_perfetto(tmp_path):
    raw, counts = _synthetic_trace()
    trace = V4DeviceTrace.from_tensors(raw, counts, _program())
    json_path = trace.write_json(tmp_path / "trace.json")
    perfetto_path = trace.write_perfetto(tmp_path / "trace.perfetto.json")
    payload = json.loads(json_path.read_text())
    perfetto = json.loads(perfetto_path.read_text())
    assert payload["schema"] == "deepspec.deepseek_v4.device_trace.v1"
    assert payload["summary"]["top_level_kernel_launches"] == 1
    assert payload["events"][1]["role"] == V4WorkerRole.MMA.name
    assert len(perfetto["traceEvents"]) == 5
