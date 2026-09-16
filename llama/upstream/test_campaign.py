"""Check that H200 lane guards distinguish assigned and unrelated GPUs."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_campaign():
  """Load the standalone campaign helper without modifying global import paths."""
  spec = importlib.util.spec_from_file_location("h200_campaign", Path(__file__).parents[1] / "campaign.py")
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_assigned_process_blocks_measurement(monkeypatch):
  """An existing process on this lane must prevent a matched run."""
  campaign = load_campaign()
  monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "GPU-assigned, 42, python\n")
  with pytest.raises(RuntimeError, match="another live process"):
    campaign.idle_boundary("GPU-assigned")


def test_other_lane_does_not_block_measurement(monkeypatch):
  """Parallel agents may use other physical GPUs without invalidating a lane."""
  campaign = load_campaign()
  responses = iter(("GPU-other, 42, python\n", "GPU-assigned, NVIDIA H200, 1590, 3200, 200, 42, 0\n"))
  monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: next(responses))
  boundary = campaign.idle_boundary("GPU-assigned")
  assert boundary["assigned_processes"] == []
  assert boundary["gpu_uuid"] == "GPU-assigned"


def test_process_query_failure_is_not_idle(monkeypatch):
  """Missing process evidence must not be interpreted as an empty device."""
  campaign = load_campaign()

  def fail(*args, **kwargs):
    """Simulate a failing telemetry command."""
    raise subprocess.CalledProcessError(1, "nvidia-smi")

  monkeypatch.setattr(subprocess, "check_output", fail)
  with pytest.raises(subprocess.CalledProcessError):
    campaign.idle_boundary("GPU-assigned")


def test_one_gpu_gets_complete_primary_and_native_coverage(tmp_path, monkeypatch):
  """The frozen estimator requires all prompts in one physical-GPU group."""
  campaign = load_campaign()
  runtime = tmp_path / "runtime"
  (runtime / "phase3").mkdir(parents=True)
  (runtime / "phase3/CAMPAIGN.json").write_text(json.dumps(campaign.plan()))
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned")
  for name in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setattr(campaign, "idle_boundary", lambda uuid: {"gpu_uuid": uuid})
  monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
  output = tmp_path / "lane0"
  campaign.run_lane(
    SimpleNamespace(
      runtime=runtime,
      lane=0,
      out=output,
      model=tmp_path / "model",
      fixtures=tmp_path / "fixtures",
      mk_dir=tmp_path / "mk",
    )
  )
  rows = json.loads((output / "commands.json").read_text())
  assert len(rows) == 27
  configurations = [dict(zip(row["argv"][2::2], row["argv"][3::2], strict=True)) for row in rows]
  graph = [row for row in configurations if row["--mode"] == "graph"]
  assert len(graph) == 24
  for index, prompt in enumerate(campaign.PROMPTS):
    matches = [row for row in graph if Path(row["--fixture"]).parent.name == prompt]
    assert {int(row["--replicate"]) for row in matches} == {0, 1, 2}
    assert {int(row["--seed"]) for row in matches} == {314159 + 100 * index + replicate for replicate in range(3)}
  native = [row for row in configurations if row["--mode"] == "native"]
  assert len(native) == 3
  assert all(Path(row["--fixture"]).parent.name == campaign.PROMPTS[0] for row in native)


def test_checker_lanes_cover_every_run_once(tmp_path, monkeypatch):
  """Independent checker assignment must cover 27 receipts without duplicates."""
  campaign = load_campaign()
  runtime = tmp_path / "runtime"
  (runtime / "phase3").mkdir(parents=True)
  (runtime / "phase3/CAMPAIGN.json").write_text(json.dumps(campaign.plan()))
  results = tmp_path / "results"
  expected = set()
  for index, prompt in enumerate(campaign.PROMPTS):
    for mode in ["graph", "native"] if index == 0 else ["graph"]:
      for replicate in range(3):
        name = f"{prompt}-{mode}-r{replicate}"
        expected.add(name)
        directory = results / name
        directory.mkdir(parents=True)
        (directory / "results.json").write_text('{"status":"complete"}')
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-checker")
  monkeypatch.setattr(campaign, "idle_boundary", lambda uuid: {"gpu_uuid": uuid})
  monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
  observed = []
  for lane in (1, 2, 3):
    output = tmp_path / f"checker{lane}"
    campaign.check_lane(
      SimpleNamespace(
        runtime=runtime,
        lane=lane,
        timeout=30,
        out=output,
        model=tmp_path / "model",
        fixtures=tmp_path / "fixtures",
        results=results,
      )
    )
    commands = json.loads((output / "commands.json").read_text())
    assert len(commands) == 9
    observed.extend(Path(row["argv"][row["argv"].index("--results") + 1]).name for row in commands)
  assert len(observed) == len(set(observed)) == 27
  assert set(observed) == expected
