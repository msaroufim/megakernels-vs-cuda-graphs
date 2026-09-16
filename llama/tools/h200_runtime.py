"""Strict runtime and file-identity checks for the H200 hardware replication."""

import hashlib
import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
  """Hash complete bytes without loading large files into memory."""
  value = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      value.update(chunk)
  return value.hexdigest()


def validate_hardware(torch: Any) -> None:
  """Require one actual H200 and let PyTorch resolve its compilation target."""
  if os.environ.get("TORCH_CUDA_ARCH_LIST"):
    raise RuntimeError("Unset TORCH_CUDA_ARCH_LIST; use the visible-device resolver")
  if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Expose exactly one assigned H200")
  if list(torch.cuda.get_device_capability()) != [9, 0]:
    raise RuntimeError("This hardware revision requires compute capability 9.0")
  if "H200" not in torch.cuda.get_device_properties(0).name:
    raise RuntimeError("This hardware revision requires an H200")
  if torch.cuda.get_device_properties(0).multi_processor_count != 132:
    raise RuntimeError("This hardware revision requires all 132 H200 SMs")


def runtime_values(torch: Any, transformers: Any) -> dict[str, str]:
  """Capture the software identity shared by references and measured runs."""
  triton = importlib.import_module("triton")

  return {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "triton": triton.__version__,
    "transformers": transformers.__version__,
    "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
  }


def validate_runtime(actual: dict, manifest: Path) -> str:
  """Reject software drift or a manifest for a different hardware revision."""
  data = json.loads(manifest.read_text())
  if data.get("schema") != "llama1b-h200-runtime-v1":
    raise ValueError("Unexpected runtime manifest schema")
  if data.get("hardware") != {"capability": [9, 0], "name_contains": "H200"}:
    raise ValueError("Runtime manifest does not require H200")
  expected = data["expected_runtime"]
  keys = {"python", "torch", "cuda", "triton", "transformers", "nvcc"}
  if set(expected) != keys or any(actual.get(key) != expected[key] for key in keys):
    raise RuntimeError("Software differs from the recorded H200 runtime")
  return digest(manifest)


def stage_identity(root: Path) -> dict[str, str]:
  """Verify all staged inputs and return their immutable start identities."""
  manifest = root / "stage-manifest.json"
  contents = json.loads(manifest.read_text())
  files = contents["staged_sha256"]
  for name, expected in files.items():
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
      raise RuntimeError(f"Staged input changed: {name}")
  return {**files, "stage-manifest.json": digest(manifest)}
