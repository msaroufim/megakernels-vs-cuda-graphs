"""CPU tests for the upstream setup's input and source integrity checks."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("h200_prepare", Path(__file__).with_name("prepare.py"))
assert spec is not None and spec.loader is not None
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_model_mutation_rejected(tmp_path, monkeypatch):
  """A previously hashed model file cannot be changed silently."""
  model = tmp_path / "model"
  model.mkdir()
  file = model / "weights.bin"
  file.write_bytes(b"checkpoint")
  expected = {file.name: prepare.sha256(file)}
  monkeypatch.setattr(prepare, "MODEL_SHA256", expected)
  assert prepare.verify_model(model) == expected
  file.write_bytes(b"different checkpoint")
  with pytest.raises(ValueError, match="identity mismatch"):
    prepare.verify_model(model)


def test_pristine_revision_rejects_tracked_edit(tmp_path, monkeypatch):
  """The same HEAD is insufficient when build inputs have changed."""
  root = tmp_path / "source"
  tk = root / "ThunderKittens"
  tk.mkdir(parents=True)
  for path, attribute in ((tk, "THUNDERKITTENS"), (root, "UPSTREAM")):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "kernel.cuh").write_text("original\n")
    subprocess.run(["git", "add", "kernel.cuh"], cwd=path, check=True)
    subprocess.run(
      [
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
      ],
      cwd=path,
      check=True,
    )
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    monkeypatch.setattr(prepare, attribute, pin)
  prepare.checked_source(root)
  (tk / "kernel.cuh").write_text("changed\n")
  with pytest.raises(subprocess.CalledProcessError):
    prepare.checked_source(root)


def test_revision_mismatch_rejected(tmp_path, monkeypatch):
  """Revision mismatch is rejected before a build can start."""
  subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
  (tmp_path / "kernel.cu").write_text("source\n")
  subprocess.run(["git", "add", "kernel.cu"], cwd=tmp_path, check=True)
  subprocess.run(
    [
      "git",
      "-c",
      "user.name=Test",
      "-c",
      "user.email=test@example.invalid",
      "commit",
      "-qm",
      "fixture",
    ],
    cwd=tmp_path,
    check=True,
  )
  monkeypatch.setattr(prepare, "UPSTREAM", "0" * 40)
  with pytest.raises(ValueError, match="expected"):
    prepare.checked_source(tmp_path)
