"""Editable CUDA sources replace upstream trees without stale build inputs."""

import subprocess

import pytest

from upstream import prepare, vendor


def tree(root):
  (root / "include").mkdir(parents=True)
  for name, value in {
    "Makefile": "all:\n",
    "llama.cu": '#include "new.cuh"\n',
    "include/new.cuh": "// editable\n",
    "LICENSE": "license\n",
  }.items():
    (root / name).write_text(value)
  return root


def test_overlay_replaces_trees_and_preserves_runtime(tmp_path):
  source, local = tmp_path / "upstream", tree(tmp_path / "vendor")
  (source / vendor.DEMO).mkdir(parents=True)
  (source / "include").mkdir()
  (source / "runtime.py").write_text("untouched\n")
  for name in [f"{vendor.DEMO}/obsolete.cu", "include/obsolete.cuh"]:
    (source / name).write_text("old\n")
  receipt = vendor.overlay(source, local)
  assert receipt == vendor.hashes(local)
  assert (source / "runtime.py").read_text() == "untouched\n"
  assert not (source / "include/obsolete.cuh").exists()
  assert not (source / vendor.DEMO / "obsolete.cu").exists()
  assert (source / vendor.DEMO / "llama.cu").read_bytes() == (local / "llama.cu").read_bytes()
  assert "LICENSE" not in receipt
  (source / vendor.DEMO / "mk_llama.so").write_bytes(b"generated binary")
  assert vendor.verify(source, local) == receipt


@pytest.mark.parametrize("change", ["modify", "remove", "add", "checkout_changed"])
def test_stale_or_mutated_build_is_rejected(tmp_path, change):
  source, local = tmp_path / "upstream", tree(tmp_path / "vendor")
  vendor.overlay(source, local)
  if change == "modify":
    (source / "include/new.cuh").write_text("mutated")
  elif change == "remove":
    (source / "include/new.cuh").unlink()
  elif change == "add":
    (source / "include/extra.cuh").write_text("unrecorded")
  else:
    (local / "llama.cu").write_text("new submission")
  with pytest.raises(ValueError, match="fresh build"):
    vendor.verify(source, local)


def test_unsupported_or_linked_sources_rejected_before_replacement(tmp_path):
  source, local = tmp_path / "upstream", tree(tmp_path / "vendor")
  vendor.overlay(source, local)
  (local / "secret.cuh").symlink_to(local / "llama.cu")
  with pytest.raises(ValueError, match="Linked"):
    vendor.overlay(source, local)
  assert (source / vendor.DEMO / "llama.cu").exists()
  (local / "secret.cuh").unlink()
  (local / "setup.py").write_text("raise RuntimeError()")
  with pytest.raises(ValueError, match="Unsupported"):
    vendor.overlay(source, local)


def git(path, *args):
  return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_local_source_check_accepts_vendor_edits_but_rejects_runtime_edits(tmp_path, monkeypatch):
  source, local = tmp_path / "upstream", tree(tmp_path / "vendor")
  for path, pin in [(source, "UPSTREAM"), (source / "ThunderKittens", "THUNDERKITTENS")]:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / "runtime.py").write_text("original\n")
    git(path, "add", "runtime.py")
    git(path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    monkeypatch.setattr(prepare, pin, git(path, "rev-parse", "HEAD"))
  receipt = vendor.overlay(source, local)
  checked = prepare.checked_source(source, local)
  assert receipt.items() <= checked["upstream"]["tracked_sha256"].items()
  (source / "runtime.py").write_text("unexpected\n")
  with pytest.raises(subprocess.CalledProcessError):
    prepare.checked_source(source, local)
