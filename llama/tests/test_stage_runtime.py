"""CPU tests for narrowly adapted staging and independent H200 identity guards."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import campaign

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "tools"))
freeze = importlib.import_module("freeze_runtime").freeze
h200_runtime = importlib.import_module("h200_runtime")
stage_identity = h200_runtime.stage_identity
validate_hardware = h200_runtime.validate_hardware
validate_runtime = h200_runtime.validate_runtime
stager = importlib.import_module("stage_runtime")
METADATA_PATHS, SOURCE_PATHS, adapt, stage = stager.METADATA_PATHS, stager.SOURCE_PATHS, stager.adapt, stager.stage


@pytest.fixture
def metadata(tmp_path):
  """Use no model data or GPU imports to exercise complete staging."""
  root = tmp_path / "metadata"
  for name in METADATA_PATHS:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
  (root / "phase3/CAMPAIGN.json").write_text(json.dumps(campaign.plan()))
  (root / "phase3/H200_RUNTIME.json").write_text(
    json.dumps({"schema": "llama1b-h200-runtime-v1", "hardware": {"capability": [9, 0], "name_contains": "H200"}})
  )
  return root


def test_deterministic_stage_and_frozen_bytes(metadata, tmp_path):
  first = stage(STUDY, metadata, tmp_path / "first", "a" * 40)
  second = stage(STUDY, metadata, tmp_path / "second", "a" * 40)
  assert stage_identity(first) == stage_identity(second)
  adapted = {"kernels/candidate.py", "benchmarks/matched_final.py", "reference/generate.py"}
  for source, target in SOURCE_PATHS.items():
    original = (STUDY / source).read_bytes()
    if source not in adapted:
      assert (first / target).read_bytes() == original
  candidate = (first / "phase3/fp32_headout_fastmath/candidate.py").read_text()
  assert "native_bf16_fma: bool = False" in candidate
  assert "native_bf16_fma: bool = True" in (STUDY / "kernels/candidate.py").read_text()
  protocol = (first / "phase3/PROTOCOL.md").read_text()
  assert protocol.startswith((STUDY / "docs/PROTOCOL.md").read_text())
  assert protocol.endswith((STUDY / "PROTOCOL.md").read_text())
  assert (first / "phase3/harness/run_campaign.py").read_bytes() == (STUDY / "campaign.py").read_bytes()
  generator = (first / "phase3/reference/generate.py").read_text()
  assert "b890e9f84f6ecf9af592bcda3a9ff2c424d7164daf82818ee03363104b41b92f" in generator
  for path in first.rglob("*.py"):
    compile(path.read_bytes(), str(path), "exec")


def test_refuse_drift_and_existing_output(metadata, tmp_path):
  with pytest.raises(ValueError, match="anchor"):
    adapt("kernels/candidate.py", "native_bf16_fma: bool = False")
  output = stage(STUDY, metadata, tmp_path / "runtime", "a" * 40)
  with pytest.raises(ValueError, match="new directory"):
    stage(STUDY, metadata, output, "a" * 40)
  (output / "phase3/harness/h200_runtime.py").write_text("changed")
  with pytest.raises(RuntimeError, match="changed"):
    stage_identity(output)


def test_missing_and_escaping_metadata_leave_no_output(metadata, tmp_path):
  path = metadata / METADATA_PATHS[0]
  path.unlink()
  with pytest.raises(ValueError, match="Missing"):
    stage(STUDY, metadata, tmp_path / "missing", "a" * 40)
  assert not (tmp_path / "missing").exists()
  outside = tmp_path / "outside.json"
  outside.write_text("{}")
  path.symlink_to(outside)
  with pytest.raises(ValueError, match="escaping"):
    stage(STUDY, metadata, tmp_path / "escaping", "a" * 40)


def test_freeze_binds_new_runtime_and_unchanged_evaluator(metadata, tmp_path):
  output = stage(STUDY, metadata, tmp_path / "runtime", "a" * 40)
  candidate = "portable=" + str(output / "phase3/fp32_headout_fastmath/candidate.py")
  result = freeze(output, [candidate], tmp_path / "freeze.json")
  assert result["source_commit"] == "a" * 40
  assert result["candidates"]["portable"]["expected_environment"] == {}
  assert "h200_runtime.py" in result["reference_sources"]
  assert "upstream_full_source_manifest_sha256" not in result
  assert result["h200_runtime_manifest_sha256"] == stage_identity(output)["phase3/H200_RUNTIME.json"]
  assert result["stage_manifest_sha256"] == stage_identity(output)["stage-manifest.json"]
  with pytest.raises(FileExistsError):
    freeze(output, [candidate], tmp_path / "freeze.json")
  with pytest.raises(ValueError, match="unique"):
    freeze(output, [candidate, candidate], tmp_path / "duplicate.json")
  with pytest.raises(ValueError, match="only accepts"):
    freeze(output, ["external=" + str(__file__)], tmp_path / "external.json")


def fake_torch(name="NVIDIA H200", capability=(9, 0), count=1, sms=132):
  """Minimal CUDA interface; never imports real PyTorch."""
  return SimpleNamespace(
    cuda=SimpleNamespace(
      is_available=lambda: True,
      device_count=lambda: count,
      get_device_capability=lambda: capability,
      get_device_properties=lambda _: SimpleNamespace(name=name, multi_processor_count=sms),
    )
  )


def test_hardware_guard(monkeypatch):
  monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
  validate_hardware(fake_torch())
  for invalid in [
    fake_torch(name="NVIDIA H100"),
    fake_torch(capability=(10, 3)),
    fake_torch(count=2),
    fake_torch(sms=131),
  ]:
    with pytest.raises(RuntimeError):
      validate_hardware(invalid)
  monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
  with pytest.raises(RuntimeError, match="Unset"):
    validate_hardware(fake_torch())


def test_runtime_mismatch(tmp_path):
  actual = dict.fromkeys(["python", "torch", "cuda", "triton", "transformers", "nvcc"], "pinned")
  manifest = tmp_path / "runtime.json"
  manifest.write_text(
    json.dumps(
      {
        "schema": "llama1b-h200-runtime-v1",
        "expected_runtime": actual,
        "hardware": {"capability": [9, 0], "name_contains": "H200"},
      }
    )
  )
  assert len(validate_runtime(actual, manifest)) == 64
  with pytest.raises(RuntimeError, match="Software"):
    validate_runtime({**actual, "triton": "changed"}, manifest)


def test_reference_and_harness_bind_same_helper(metadata, tmp_path):
  output = stage(STUDY, metadata, tmp_path / "runtime", "a" * 40)
  assert (output / "phase3/harness/h200_runtime.py").read_bytes() == (
    output / "phase3/reference/h200_runtime.py"
  ).read_bytes()
  # Import only the new freezer, proving the staged entry point works without Git.
  spec = importlib.util.spec_from_file_location("staged_freezer", output / "phase3/harness/freeze_final.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assert callable(module.freeze)
