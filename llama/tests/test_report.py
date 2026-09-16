"""Check timing units and prevent a development receipt becoming an accepted claim."""

import json

import pytest

from tools.report import render


def test_sequence_time_converts_to_step_and_smoke_stays_unqualified(tmp_path):
  """The frozen raw timings contain 127-step sequence durations in microseconds."""
  run = tmp_path / "paired" / "dev_weather-graph-r0"
  run.mkdir(parents=True)
  (run / "results.json").write_text(
    json.dumps(
      {
        "status": "complete",
        "configuration": {"mode": "graph"},
        "gpu": {"name": "NVIDIA H200", "uuid": "GPU-one", "sms": 132},
        "timing_us": {"candidate": [127000, 127000], "control": [63500, 63500]},
      }
    )
  )
  text = render(tmp_path)
  assert "| graph | 1 | 1000.00 | 500.00 |" in text
  assert "no full-suite statistical or numerical acceptance claim" in text


def test_different_hardware_or_uuid_must_not_be_pooled(tmp_path):
  for index, gpu in enumerate(
    [
      {"name": "NVIDIA H200", "uuid": "GPU-one", "sms": 132},
      {"name": "NVIDIA H100", "uuid": "GPU-two", "sms": 132},
    ]
  ):
    run = tmp_path / "paired" / str(index)
    run.mkdir(parents=True)
    (run / "results.json").write_text(
      json.dumps(
        {
          "status": "complete",
          "configuration": {"mode": "graph"},
          "gpu": gpu,
          "timing_us": {"candidate": [127000], "control": [63500]},
        }
      )
    )
  with pytest.raises(ValueError, match="Do not pool"):
    render(tmp_path)
