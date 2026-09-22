"""Guard against misleading timing/coverage in the worker diagram."""

import json

import pytest

from scripts.megakernel.render_trace import SCHEMA, read_trace, task_counts


def write_trace(tmp_path, events, **overrides):
    payload = {
        "schema": SCHEMA,
        "timestamp_unit": "ns",
        "summary": {"workers": 3, "events": len(events)},
        "events": events,
    }
    payload.update(overrides)
    path = tmp_path / "phases.json"
    path.write_text(json.dumps(payload))
    return path


def event(worker, start, end, role="MMA", segment="USEFUL"):
    # Nanosecond differences must survive an epoch-scale globaltimer origin.
    origin = 1_788_544_492_322_224_512
    return {
        "worker": worker,
        "phase": "layer_0.fp4_routed_w13",
        "role": role,
        "segment": segment,
        "start_ns": origin + start,
        "end_ns": origin + end,
    }


def test_union_counts_ctas_not_roles_and_preserves_nanoseconds(tmp_path):
    events = [
        event(0, 0, 100),
        event(0, 0, 100, role="TMA"),
        event(0, 50, 150),
        event(0, 150, 200),
        event(1, 50, 100),
        event(1, 0, 250, role="CONTROLLER", segment="CONTROLLER"),
    ]
    trace = read_trace(write_trace(tmp_path, events))
    assert len(trace.spans) == 5  # Role duplicates collapse, overlaps remain visible.
    assert trace.workers == 3  # Keep workers with no recorded events.
    assert trace.duration_us == 0.25
    assert task_counts(trace) == ([0.0, 0.05, 0.1, 0.2, 0.25], [1, 2, 1, 0, 0])


@pytest.mark.parametrize(
    "change",
    [
        {"end_ns": 0},
        {"worker": 3},
        {"start_ns": 1.788e18},
        {"segment": "CONTROLLER"},
    ],
)
def test_rejects_invalid_or_entry_only_trace(tmp_path, change):
    raw = event(0, 0, 100)
    raw.update(change)
    with pytest.raises(ValueError):
        read_trace(write_trace(tmp_path, [raw]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"timestamp_unit": "us"},
        {"schema": "other"},
        {"summary": {"workers": 3, "events": 2}},
    ],
)
def test_rejects_wrong_format_or_incomplete_export(tmp_path, overrides):
    with pytest.raises(ValueError):
        read_trace(write_trace(tmp_path, [event(0, 0, 100)], **overrides))
