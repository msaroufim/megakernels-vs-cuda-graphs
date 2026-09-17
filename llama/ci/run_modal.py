"""Trusted controller for credential-free, one-H100 PR benchmarks."""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shlex
import selectors
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
from upstream.prepare import DEPENDENCIES, MODEL_SHA256  # noqa: E402

VOLUME = "megakernels-llama-weights"
IMAGE = "nvcr.io/nvidia/pytorch:25.03-py3"
TIMEOUT = 90 * 60
MAX_ARCHIVE = 32 * 1024 * 1024
MAX_REPORT = 8 * 1024 * 1024
UPSTREAMS = {
  "UPSTREAM": ("Megakernels.git", "https://github.com/HazyResearch/Megakernels.git"),
  "THUNDERKITTENS": ("ThunderKittens.git", "https://github.com/HazyResearch/ThunderKittens.git"),
}


def checked_sha(value: str) -> str:
  if not re.fullmatch(r"[0-9a-f]{40}", value):
    raise ValueError("Expected a full lowercase Git SHA")
  return value


def selected_file(name: str) -> bool:
  path = PurePosixPath(name)
  if path.is_absolute() or ".." in path.parts:
    raise ValueError("Unsafe source path")
  return (
    name.startswith("llama/kernels/")
    and path.suffix in {".py", ".cu", ".cuh", ".h", ".hpp"}
    or name == "llama/upstream/prepare.py"
    or name == "llama/megakernel/Makefile"
    or name.startswith("llama/megakernel/")
    and path.suffix in {".cu", ".cuh", ".h", ".hpp"}
  )


def extract_snapshot(raw: bytes, destination: Path) -> None:
  """Copy only regular allowlisted files; never execute repository archive metadata."""
  size, total, count, seen = 0, 0, 0, set()
  with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
    for member in archive:
      total += member.size
      count += 1
      if member.size < 0 or total > 128 * 1024 * 1024 or count > 10000:
        raise ValueError("Expanded repository archive exceeds limits")
      parts = PurePosixPath(member.name).parts
      if member.name.startswith("/") or ".." in parts:
        raise ValueError("Unsafe archive path")
      if len(parts) < 2:
        continue
      name = "/".join(parts[1:])
      if not selected_file(name):
        continue
      if not member.isfile() or name in seen:
        raise ValueError("Source files must be unique regular files")
      seen.add(name)
      size += member.size
      if member.size < 0 or size > MAX_ARCHIVE:
        raise ValueError("Source snapshot exceeds size limit")
      source = archive.extractfile(member)
      if source is None:
        raise ValueError("Missing archive file")
      target = destination / name
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(source.read())
  if not (destination / "llama/kernels/candidate.py").is_file():
    raise ValueError("Missing candidate entry point")
  vendor = destination / "llama/megakernel"
  if vendor.exists() and not all((vendor / name).is_file() for name in ("Makefile", "llama.cu")):
    raise ValueError("Vendored megakernel requires both Makefile and llama.cu")
  read_pins(destination)


def read_pins(snapshot: Path) -> dict[str, str]:
  tree = ast.parse((snapshot / "llama/upstream/prepare.py").read_text())
  pins = {}
  for node in tree.body:
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name) and target.id in UPSTREAMS:
          if target.id in pins or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            raise ValueError("Upstream pins must be single literal assignments")
          pins[target.id] = checked_sha(node.value.value)
  if set(pins) != set(UPSTREAMS):
    raise ValueError("Missing upstream source pins")
  return pins


def github_json(endpoint: str) -> dict:
  return json.loads(subprocess.check_output(["gh", "api", endpoint], timeout=60))


def bounded_output(command: list[str], maximum: int, timeout: int) -> bytes:
  """Stop downloads at the limit rather than filling the credentialed runner's disk."""
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
  assert process.stdout is not None
  deadline, output = time.monotonic() + timeout, bytearray()
  try:
    with selectors.DefaultSelector() as selector:
      selector.register(process.stdout, selectors.EVENT_READ)
      while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not selector.select(remaining):
          raise TimeoutError("Source download timed out")
        chunk = os.read(process.stdout.fileno(), min(65536, maximum + 1 - len(output)))
        if not chunk:
          break
        output.extend(chunk)
        if len(output) > maximum:
          raise ValueError("Repository archive exceeds size limit")
    if process.wait(timeout=max(0.01, deadline - time.monotonic())):
      raise RuntimeError("Source download failed")
    return bytes(output)
  finally:
    if process.poll() is None:
      process.kill()
    process.wait()
    process.stdout.close()


def fetch_snapshots(repo: str, pr: int, base_sha: str, head_sha: str, destination: Path) -> dict:
  if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or pr <= 0:
    raise ValueError("Invalid repository or PR")
  pull = github_json(f"repos/{repo}/pulls/{pr}")
  if pull["base"]["sha"] != checked_sha(base_sha) or pull["head"]["sha"] != checked_sha(head_sha):
    raise ValueError("PR changed since this run was requested")
  if pull["state"] != "open" or pull["draft"] or pull["base"]["repo"]["full_name"] != repo:
    raise ValueError("PR must be open, ready, and target this repository")
  for side, sha in (("base", base_sha), ("head", head_sha)):
    owner = pull[side]["repo"]["full_name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", owner):
      raise ValueError("Invalid source repository")
    # Download stays on the trusted host; no GitHub credential is sent to Modal.
    raw = bounded_output(["gh", "api", f"repos/{owner}/tarball/{sha}"], MAX_ARCHIVE, 180)
    extract_snapshot(raw, destination / side)
  return {"repository": repo, "pr": pr, "base_sha": base_sha, "head_sha": head_sha}


def upstream_cache(snapshots: Path, target: Path) -> None:
  """Fetch only public pinned upstreams before disabling sandbox networking."""
  pins = [read_pins(snapshots / side) for side in ("base", "head")]
  target.mkdir()
  for key, (name, url) in UPSTREAMS.items():
    mirror = target / name
    subprocess.run(["git", "init", "--bare", "--quiet", str(mirror)], check=True)
    for sha in sorted({p[key] for p in pins}):
      subprocess.run(
        ["git", "-C", str(mirror), "fetch", "--quiet", "--depth=1", url, f"{sha}:refs/heads/pin-{sha}"],
        check=True,
        timeout=300,
      )


def benchmark_image(modal):
  # Dependency layers do not depend on frequently changing kernel sources.
  return (
    modal.Image.from_registry(IMAGE)
    .apt_install("git", "build-essential", "ninja-build")
    .run_commands("python -m pip install --no-deps " + " ".join(map(shlex.quote, DEPENDENCIES)))
    .env({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MAX_JOBS": "4", "HF_HUB_OFFLINE": "1"})
  )


def read_bounded(sandbox, remote: str, maximum: int) -> bytes:
  info = sandbox.filesystem.stat(remote)
  if not info.is_file() or info.size > maximum:
    raise ValueError("Benchmark artifact exceeds size limit or is not a regular file")
  value = sandbox.filesystem.read_bytes(remote)
  if len(value) > maximum:
    raise ValueError("Benchmark artifact exceeds size limit")
  return value


def run_benchmark(args) -> int:
  import modal

  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  started = time.monotonic()
  metadata: dict[str, Any] = {"requested_gpu": "H100", "timeout_seconds": TIMEOUT}
  sandbox = None
  status = 1
  try:
    with tempfile.TemporaryDirectory(prefix="llama-ci-") as folder:
      work = Path(folder)
      if args.base_dir and args.head_dir:
        import shutil

        for side, source in (("base", args.base_dir), ("head", args.head_dir)):
          shutil.copytree(source, work / "snapshots" / side)
        metadata.update(base_sha=checked_sha(args.base_sha), head_sha=checked_sha(args.head_sha))
      elif args.base_dir or args.head_dir:
        raise ValueError("Both local snapshot paths are required")
      else:
        metadata.update(fetch_snapshots(args.repo, args.pr, args.base_sha, args.head_sha, work / "snapshots"))
      upstream_cache(work / "snapshots", work / "upstream")
      metadata["source_setup_seconds"] = time.monotonic() - started
      volume = modal.Volume.from_name(VOLUME)
      image = (
        benchmark_image(modal)
        .add_local_dir(STUDY, "/trusted/llama", ignore=[".venv", "__pycache__", ".pytest_cache", ".ruff_cache"])
        .add_local_dir(work / "snapshots", "/snapshots")
        .add_local_dir(work / "upstream", "/upstream")
      )
      app = modal.App("llama-performance-ci")
      with app.run():
        launch = time.monotonic()
        sandbox = modal.Sandbox.create(
          "sleep",
          str(TIMEOUT),
          app=app,
          image=image,
          gpu="H100",
          cpu=8,
          memory=65536,
          timeout=TIMEOUT,
          volumes={"/weights": volume.with_mount_options(read_only=True)},
          block_network=True,
        )
        metadata["sandbox_id"] = sandbox.object_id
        metadata["sandbox_create_seconds"] = time.monotonic() - launch
        print(f"H100 sandbox: {sandbox.object_id}; maximum lifetime {TIMEOUT}s", flush=True)
        process = sandbox.exec(
          "bash",
          "-c",
          "mkdir -p /out; python /trusted/llama/ci/benchmark.py "
          "--base /snapshots/base --head /snapshots/head --model /weights/model "
          "--upstream-cache /upstream --output /out/results --pairs 3 "
          ">/out/benchmark.log 2>&1",
          timeout=TIMEOUT - 60,
        )
        process.wait()
        metadata["exit_code"] = process.returncode
        metadata["sandbox_wall_seconds"] = time.monotonic() - launch
        download = time.monotonic()
        # Do not stream untrusted output as GitHub workflow commands.
        for remote, name, limit in (
          ("/out/results/report.json", "report.json", MAX_REPORT),
          ("/out/benchmark.log", "benchmark.log", 16 * 1024 * 1024),
        ):
          try:
            (output / name).write_bytes(read_bounded(sandbox, remote, limit))
          except Exception:
            metadata.setdefault("missing_artifacts", []).append(name)
        metadata["artifact_download_seconds"] = time.monotonic() - download
        report = json.loads((output / "report.json").read_text())
        status = 0 if process.returncode == 0 and report.get("status") == "complete" else 1
  except Exception as error:
    # Only trusted-host errors reach this log; sandbox text stays in artifacts.
    print(f"Benchmark controller failed: {type(error).__name__}", file=sys.stderr)
    metadata["controller_error_type"] = type(error).__name__
  finally:
    if sandbox is not None:
      try:
        sandbox.terminate()
      except Exception as error:
        metadata["cleanup_error_type"] = type(error).__name__
        status = 1
    metadata["total_seconds"] = time.monotonic() - started
    (output / "controller.json").write_text(json.dumps(metadata, indent=2) + "\n")
  return status


def seed_weights() -> None:
  """Download once on trusted CPU infrastructure; PR jobs mount the result read-only."""
  import modal

  token = os.environ.get("HF_TOKEN")
  if not token:
    raise ValueError("HF_TOKEN is required for the one-time model download")
  code = """import hashlib,json,os,shutil,tempfile
from pathlib import Path
from huggingface_hub import snapshot_download
expected=json.loads(os.environ['EXPECTED_HASHES'])
with tempfile.TemporaryDirectory() as tmp:
 snapshot_download('meta-llama/Llama-3.2-1B-Instruct',allow_patterns=list(expected),local_dir=tmp,token=os.environ['HF_TOKEN'])
 for name,digest in expected.items():
  h=hashlib.sha256()
  with open(Path(tmp)/name,'rb') as f:
   for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
  assert h.hexdigest()==digest, 'Model hash mismatch: '+name
 dst=Path('/weights/model');dst.mkdir(exist_ok=True)
 for name in expected:shutil.copyfile(Path(tmp)/name,dst/name)
 os.sync()
print('Four model files verified and cached')
"""
  volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
  image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.2")
  sandbox = None
  app = modal.App("llama-weights-setup")
  with app.run():
    try:
      sandbox = modal.Sandbox.create(
        "python",
        "-c",
        code,
        app=app,
        image=image,
        cpu=2,
        memory=8192,
        timeout=1200,
        volumes={"/weights": volume},
        secrets=[modal.Secret.from_dict({"HF_TOKEN": token})],
        env={"EXPECTED_HASHES": json.dumps(MODEL_SHA256)},
      )
      sandbox.wait()
      if sandbox.returncode:
        raise RuntimeError("Model download or hash verification failed; inspect the trusted setup sandbox")
      print(sandbox.stdout.read())
    finally:
      if sandbox is not None:
        sandbox.terminate()
  print(f"Weights volume: {VOLUME}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  commands.add_parser("seed-weights")
  run = commands.add_parser("benchmark")
  run.add_argument("--repo", default="msaroufim/megakernels-vs-cuda-graphs")
  run.add_argument("--pr", type=int, default=0)
  run.add_argument("--base-sha", required=True)
  run.add_argument("--head-sha", required=True)
  run.add_argument("--output", type=Path, required=True)
  run.add_argument("--base-dir", type=Path)
  run.add_argument("--head-dir", type=Path)
  args = parser.parse_args()
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
  if args.command == "seed-weights":
    seed_weights()
    return 0
  return run_benchmark(args)


if __name__ == "__main__":
  raise SystemExit(main())
