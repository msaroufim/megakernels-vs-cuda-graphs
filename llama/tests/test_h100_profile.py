"""CPU-only checks that explicit H100 staging cannot weaken H200 eligibility."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import campaign
import reproduce
from tools.hardware_profile import bind_hardware
from tools.stage_runtime import METADATA_PATHS, SOURCE_PATHS, stage

ROOT = Path(__file__).resolve().parents[1]


def metadata_at(root, gpu_type):
  for template in METADATA_PATHS:
    path = root / bind_hardware(template, gpu_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
  (root / "phase3/CAMPAIGN.json").write_text(bind_hardware(json.dumps(campaign.plan()), gpu_type))
  (root / f"phase3/{gpu_type}_RUNTIME.json").write_text(
    json.dumps(
      {
        "schema": f"llama1b-{gpu_type.lower()}-runtime-v1",
        "hardware": {"capability": [9, 0], "name_contains": gpu_type},
      }
    )
  )
  return root


def load_helper(path):
  # No sys.path or sys.modules mutation: the other profile's freezer stays isolated.
  spec = importlib.util.spec_from_file_location("independent_profile_guard", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def fake_torch(name, sms=132, capability=(9, 0), count=1):
  return SimpleNamespace(
    cuda=SimpleNamespace(
      is_available=lambda: True,
      device_count=lambda: count,
      get_device_capability=lambda: capability,
      get_device_properties=lambda _: SimpleNamespace(name=name, multi_processor_count=sms),
    )
  )


@pytest.mark.parametrize("gpu_type", ["H100", "H200"])
def test_each_stage_freezes_and_preserves_original_arithmetic(tmp_path, gpu_type):
  original = {name: (ROOT / name).read_bytes() for name in SOURCE_PATHS}
  output = stage(ROOT, metadata_at(tmp_path / "metadata", gpu_type), tmp_path / "runtime", "a" * 40, gpu_type=gpu_type)
  helper_name = gpu_type.lower() + "_runtime.py"
  assert (output / "phase3/reference" / helper_name).read_bytes() == (
    output / "phase3/harness" / helper_name
  ).read_bytes()
  for name in (
    "kernels/projection.cu",
    "kernels/attention.py",
    "reference/metrics.py",
    "reference/check_execution.py",
    "benchmarks/summarize.py",
  ):
    assert (output / SOURCE_PATHS[name]).read_bytes() == original[name]
  assert (output / "phase3/fp32_headout_fastmath/candidate.py").read_text() == original[
    "kernels/candidate.py"
  ].decode().replace("native_bf16_fma: bool = True", "native_bf16_fma: bool = False")
  for name, raw in original.items():
    assert (ROOT / name).read_bytes() == raw
  assert (output / "phase3/PROTOCOL.md").read_bytes().startswith((ROOT / "docs/PROTOCOL.md").read_bytes())
  for path in output.rglob("*.py"):
    compile(path.read_bytes(), str(path), "exec")
  subprocess.run(
    [
      sys.executable,
      str(output / "phase3/harness/freeze_final.py"),
      "--candidate",
      "portable=" + str(output / "phase3/fp32_headout_fastmath/candidate.py"),
      "--out",
      str(tmp_path / "freeze.json"),
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  frozen = json.loads((tmp_path / "freeze.json").read_text())
  assert f"{gpu_type.lower()}_runtime_manifest_sha256" in frozen
  assert helper_name in frozen["reference_sources"]
  assert "phase3/harness/" + helper_name in frozen["harness_sources"]
  staged_campaign = load_helper(output / "phase3/harness/run_campaign.py")
  assert staged_campaign.plan() == json.loads((output / "phase3/CAMPAIGN.json").read_text())


@pytest.mark.parametrize("gpu_type", ["H100", "H200"])
def test_profile_hardware_guard_is_exclusive(tmp_path, monkeypatch, gpu_type):
  monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
  output = stage(ROOT, metadata_at(tmp_path / "metadata", gpu_type), tmp_path / "runtime", "a" * 40, gpu_type=gpu_type)
  guard = load_helper(output / f"phase3/harness/{gpu_type.lower()}_runtime.py")
  guard.validate_hardware(fake_torch("NVIDIA " + gpu_type))
  other = "H200" if gpu_type == "H100" else "H100"
  for bad in [
    fake_torch("NVIDIA " + other),
    fake_torch("NVIDIA " + gpu_type, sms=114),
    fake_torch("NVIDIA " + gpu_type, sms=131),
    fake_torch("NVIDIA " + gpu_type, capability=(10, 3)),
    fake_torch("NVIDIA " + gpu_type, count=2),
  ]:
    with pytest.raises(RuntimeError):
      guard.validate_hardware(bad)
  monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
  with pytest.raises(RuntimeError, match="Unset"):
    guard.validate_hardware(fake_torch("NVIDIA " + gpu_type))


@pytest.mark.parametrize("changed", ["runtime", "campaign"])
def test_h100_rejects_wrong_profile_metadata_before_writing(tmp_path, changed):
  metadata = metadata_at(tmp_path / "metadata", "H100")
  if changed == "runtime":
    path = metadata / "phase3/H100_RUNTIME.json"
    path.write_text(path.read_text().replace("H100", "H200").replace("h100", "h200"))
  else:
    (metadata / "phase3/CAMPAIGN.json").write_text(json.dumps(campaign.plan()))
  output = tmp_path / "runtime"
  with pytest.raises(ValueError):
    stage(ROOT, metadata, output, "a" * 40, gpu_type="H100")
  assert not output.exists()


def test_h100_metadata_binds_plan_and_only_selected_runtime(tmp_path):
  build = metadata_at(tmp_path / "build", "H100")
  output = tmp_path / "metadata"
  hashes = reproduce.write_metadata(build, output, gpu_type="H100")
  assert "phase3/H100_RUNTIME.json" in hashes
  assert "phase3/H200_RUNTIME.json" not in hashes
  assert json.loads((output / "phase3/CAMPAIGN.json").read_text()) == json.loads(
    bind_hardware(json.dumps(campaign.plan()), "H100")
  )


def test_invalid_profile_is_rejected():
  with pytest.raises(ValueError):
    bind_hardware("", "auto")
