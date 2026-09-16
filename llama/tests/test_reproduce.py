"""CPU-only checks of the standalone command plan and provenance boundaries."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("standalone_reproduce", ROOT / "reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def test_help_needs_no_gpu_or_private_inputs():
  result = subprocess.run([sys.executable, str(ROOT / "reproduce.py"), "--help"], capture_output=True, text=True)
  assert result.returncode == 0
  assert "--suite" in result.stdout and "--build" in result.stdout


def test_command_plans_preserve_protocol_and_serialize_one_gpu(tmp_path):
  smoke = reproduce.command_plan(tmp_path, Path("/model"), Path("/build"), "smoke")
  assert [name for name, _ in smoke] == ["freeze", "references", "matched-smoke", "actual-smoke"]
  assert smoke[1][1][-1] == "development"
  assert "--candidate-freeze" not in smoke[1][1]
  matched = dict(smoke)["matched-smoke"]
  for flag, value in [("--samples", "10"), ("--warmup", "5"), ("--control", "legacy"), ("--max-len", "16384")]:
    assert matched[matched.index(flag) + 1] == value
  full = reproduce.command_plan(tmp_path, Path("/model"), Path("/build"), "full")
  assert [name for name, _ in full] == [
    "freeze",
    "references",
    "paired-full",
    "actual-lane-1",
    "actual-lane-2",
    "actual-lane-3",
    "summary",
  ]
  assert "validation" in full[1][1] and "--candidate-freeze" in full[1][1]
  for lane in (1, 2, 3):
    command = dict(full)[f"actual-lane-{lane}"]
    assert command[command.index("--lane") + 1] == str(lane)
  assert dict(full)["summary"][-1] == str(tmp_path / "summary.json")


def test_path_arch_and_device_guards(tmp_path, monkeypatch):
  monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
  repo, model = tmp_path / "repo", tmp_path / "model"
  repo.mkdir()
  model.mkdir()
  uuid = "GPU-12345678-1234-1234-1234-123456789012"
  reproduce.validate_paths(repo, tmp_path / "output", model, uuid)
  with pytest.raises(ValueError, match="outside"):
    reproduce.validate_paths(repo, repo / "output", model, uuid)
  with pytest.raises(FileExistsError):
    reproduce.validate_paths(repo, model, model, uuid)
  with pytest.raises(ValueError, match="physical GPU"):
    reproduce.validate_paths(repo, tmp_path / "output", model, "0")
  monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "")
  with pytest.raises(ValueError, match="Unset"):
    reproduce.validate_paths(repo, tmp_path / "output", model, uuid)


def test_tracked_source_dirt_and_parent_repo_rejected(tmp_path):
  subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
  source = tmp_path / "source.py"
  source.write_text("original\n")
  subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
  subprocess.run(
    [
      "git",
      "-C",
      str(tmp_path),
      "-c",
      "user.name=Test",
      "-c",
      "user.email=test@example.org",
      "-c",
      "commit.gpgsign=false",
      "commit",
      "-qm",
      "Fixture",
    ],
    check=True,
  )
  assert len(reproduce.source_commit(tmp_path)) == 40
  source.write_text("changed\n")
  with pytest.raises(ValueError, match="tracked source"):
    reproduce.source_commit(tmp_path)
  nested = tmp_path / "nested"
  nested.mkdir()
  with pytest.raises(ValueError, match="parent repository"):
    reproduce.source_commit(nested)


def test_metadata_uses_bundled_inputs_and_preserves_source(tmp_path):
  from tools.prompt_suite import PROMPTS_JSON
  from upstream.prepare import MODEL_SHA256

  source_before = {p: reproduce.sha256(p) for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts}
  build = tmp_path / "build"
  paths = (
    "phase3/H200_RUNTIME.json",
    "phase3/upstream/legacy-runtime-sha256.json",
    "phase3/upstream/source-manifest.json",
  )
  for name in paths:
    path = build / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
  output = tmp_path / "metadata"
  hashes = reproduce.write_metadata(build, output)
  assert len(hashes) == 6
  assert json.loads((output / "phase3/reference/model-sha256.json").read_text()) == MODEL_SHA256
  prompt_bytes = PROMPTS_JSON.encode() if isinstance(PROMPTS_JSON, str) else PROMPTS_JSON
  assert (output / "phase3/reference/prompts-v1.json").read_bytes() == prompt_bytes
  assert (
    hashes["phase3/reference/prompts-v1.json"] == "b890e9f84f6ecf9af592bcda3a9ff2c424d7164daf82818ee03363104b41b92f"
  )
  assert all(reproduce.sha256(path) == value for path, value in source_before.items())


@pytest.fixture
def successful_build(tmp_path, monkeypatch):
  from upstream import prepare

  build, model = tmp_path / "build", tmp_path / "model"
  model.mkdir()
  source = build / "upstream-authors"
  source.mkdir(parents=True)
  binary = source / "mk_llama.so"
  binary.write_bytes(b"binary")
  (source / "model.py").write_text("source")
  python = build / "venv/bin/python"
  python.parent.mkdir(parents=True)
  python.write_text("interpreter")
  checked = {
    "upstream": {"commit": prepare.UPSTREAM, "tracked_sha256": {"model.py": reproduce.sha256(source / "model.py")}},
    "thunderkittens": {"commit": prepare.THUNDERKITTENS, "tracked_sha256": {}},
  }
  monkeypatch.setattr(prepare, "checked_source", lambda _: checked)
  monkeypatch.setattr(prepare, "verify_model", lambda _: {"model.safetensors": "model-hash"})
  report = dict(
    passed=True,
    device={"name": "NVIDIA H200", "sm_count": 132, "capability": [9, 0]},
    upstream_commit=prepare.UPSTREAM,
    thunderkittens_commit=prepare.THUNDERKITTENS,
    model_sha256={"model.safetensors": "model-hash"},
    source_root=str(source),
    python=str(python),
    source_before=checked,
    source_after=checked,
    binary_path=str(binary),
    binary_sha256=reproduce.sha256(binary),
  )
  (build / "setup.json").write_text(json.dumps(report))
  manifests = build / "phase3/upstream"
  manifests.mkdir(parents=True)
  (manifests / "source-manifest.json").write_text(json.dumps(report))
  (build / "phase3/H200_RUNTIME.json").write_text(
    json.dumps({"schema": "llama1b-h200-runtime-v1", "hardware": {"capability": [9, 0], "name_contains": "H200"}})
  )
  identity = {**checked["upstream"]["tracked_sha256"], "mk_llama.so": report["binary_sha256"]}
  (manifests / "legacy-runtime-sha256.json").write_text(json.dumps(identity))
  return build, model, binary


def test_existing_build_checks_binary_and_complete_manifest(successful_build):
  build, model, binary = successful_build
  assert reproduce.validate_build(build, model)["passed"]
  manifest = build / "phase3/upstream/legacy-runtime-sha256.json"
  saved = manifest.read_bytes()
  manifest.write_text("{}")
  with pytest.raises(ValueError, match="incomplete"):
    reproduce.validate_build(build, model)
  manifest.write_bytes(saved)
  binary.write_bytes(b"changed")
  with pytest.raises(ValueError, match="binary identity"):
    reproduce.validate_build(build, model)


def test_existing_build_rejects_failed_or_wrong_pin(successful_build):
  build, model, _ = successful_build
  path = build / "setup.json"
  data = json.loads(path.read_text())
  data["passed"] = False
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="finish successfully"):
    reproduce.validate_build(build, model)
  data.update(passed=True, upstream_commit="different")
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="revisions"):
    reproduce.validate_build(build, model)


def test_process_failure_keeps_manifest_and_command_log(successful_build, tmp_path, monkeypatch):
  build, model, _ = successful_build
  monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

  def checkout_commit(repo):
    assert repo == ROOT.parent
    return "a" * 40

  monkeypatch.setattr(reproduce, "source_commit", checkout_commit)
  observed = []

  def failed_process(argv, **kwargs):
    observed.append((argv, kwargs["env"]))
    kwargs["stdout"].write("deliberate CPU-test failure\n")
    return subprocess.CompletedProcess(argv, 7)

  monkeypatch.setattr(reproduce.subprocess, "run", failed_process)
  output = tmp_path / "failure"
  status = reproduce.main(
    [
      "--model",
      str(model),
      "--workdir",
      str(output),
      "--build",
      str(build),
      "--gpu-uuid",
      "GPU-12345678-1234-1234-1234-123456789012",
    ]
  )
  assert status == 1
  report = json.loads((output / "reproduction.json").read_text())
  assert report["status"] == "failed" and report["commands"][0]["returncode"] == 7
  assert "deliberate CPU-test failure" in (output / report["commands"][0]["log"]).read_text()
  assert "TORCH_CUDA_ARCH_LIST" not in observed[0][1]
  assert observed[0][1]["PATH"].startswith(str(build / "venv/bin"))


def test_existing_build_rejects_other_hardware_profile(successful_build):
  build, model, _ = successful_build
  with pytest.raises(ValueError, match="hardware"):
    reproduce.validate_build(build, model, gpu_type="H100")
  setup = build / "setup.json"
  source_manifest = build / "phase3/upstream/source-manifest.json"
  data = json.loads(setup.read_text())
  data["device"]["name"] = "NVIDIA H100 80GB HBM3"
  setup.write_text(json.dumps(data))
  source_manifest.write_text(json.dumps(data))
  (build / "phase3/H100_RUNTIME.json").write_text(
    json.dumps({"schema": "llama1b-h100-runtime-v1", "hardware": {"capability": [9, 0], "name_contains": "H100"}})
  )
  assert reproduce.validate_build(build, model, gpu_type="H100")["passed"]
  with pytest.raises(ValueError, match="hardware"):
    reproduce.validate_build(build, model)
  data["device"]["capability"] = [10, 3]
  setup.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="hardware"):
    reproduce.validate_build(build, model, gpu_type="H100")


def test_existing_build_rejects_profile_manifest_and_device_provenance_drift(successful_build):
  build, model, _ = successful_build
  runtime = build / "phase3/H200_RUNTIME.json"
  saved = runtime.read_bytes()
  runtime.write_text(
    json.dumps({"schema": "llama1b-h100-runtime-v1", "hardware": {"capability": [9, 0], "name_contains": "H100"}})
  )
  with pytest.raises(ValueError, match="runtime manifest"):
    reproduce.validate_build(build, model)
  runtime.write_bytes(saved)
  provenance = build / "phase3/upstream/source-manifest.json"
  data = json.loads(provenance.read_text())
  data["device"]["sm_count"] = 114
  provenance.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="provenance"):
    reproduce.validate_build(build, model)


def test_runner_rejects_artifacts_in_sibling_study(tmp_path, monkeypatch):
  """Moving the runner must not allow generated outputs elsewhere in the checkout."""
  monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
  checkout = tmp_path / "checkout"
  study = checkout / "llama"
  sibling = checkout / "dspark"
  model = tmp_path / "model"
  for path in (study, sibling, model):
    path.mkdir(parents=True)
  monkeypatch.setattr(reproduce, "__file__", str(study / "reproduce.py"))
  output = sibling / "generated"
  with pytest.raises(ValueError, match="outside the source checkout"):
    reproduce.main(
      [
        "--model",
        str(model),
        "--workdir",
        str(output),
        "--gpu-uuid",
        "GPU-12345678-1234-1234-1234-123456789012",
      ]
    )
  assert not output.exists()
