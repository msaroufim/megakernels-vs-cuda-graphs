"""Stage editable CUDA sources into the pinned upstream Python runtime."""

import hashlib
import shutil
from pathlib import Path

DEMO = "demos/low-latency-llama"
SUFFIXES = {".cu", ".cuh", ".h", ".hpp"}


def sources(vendor: Path) -> dict[str, Path]:
  if vendor.is_symlink() or not vendor.is_dir():
    raise ValueError("Missing or linked megakernel source directory")
  result = {}
  for path in sorted(vendor.rglob("*")):
    if path.is_symlink():
      raise ValueError("Linked megakernel sources are unsupported")
    if not path.is_file():
      continue
    name = path.relative_to(vendor).as_posix()
    if name in {"LICENSE", "ORIGIN.md"}:
      continue
    if name != "Makefile" and path.suffix not in SUFFIXES:
      raise ValueError(f"Unsupported megakernel source: {name}")
    target = name if name.startswith("include/") else f"{DEMO}/{name}"
    result[target] = path
  if not {f"{DEMO}/Makefile", f"{DEMO}/llama.cu"} <= result.keys():
    raise ValueError("Megakernel sources require Makefile and llama.cu")
  return result


def hashes(vendor: Path) -> dict[str, str]:
  return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources(vendor).items()}


def verify(source: Path, vendor: Path) -> dict[str, str]:
  """Reject stale, added, removed or mutated CUDA inputs after staging/building."""
  expected = hashes(vendor)
  actual = {}
  for area in (DEMO, "include"):
    root = source / area
    if root.is_symlink():
      raise ValueError("Linked upstream source directory")
    for path in root.rglob("*"):
      if path.is_symlink():
        raise ValueError("Linked upstream source")
      if path.is_file() and (path.suffix in SUFFIXES or path == source / DEMO / "Makefile"):
        actual[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
  if actual != expected:
    raise ValueError("Built megakernel sources differ from this checkout; make a fresh build")
  return actual


def overlay(source: Path, vendor: Path) -> dict[str, str]:
  files = sources(vendor)  # Validate before replacing either upstream tree.
  for area in (DEMO, "include"):
    target = source / area
    if target.is_symlink():
      raise ValueError("Linked upstream source directory")
    if target.exists():
      shutil.rmtree(target)
  for name, path in files.items():
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
  return verify(source, vendor)
