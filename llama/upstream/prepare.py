"""Build vendored publication-era Megakernels for an explicitly assigned Hopper GPU.

This does not allocate GPUs. Every invocation requires a fresh external directory.
The upstream H100 target is intentional: it selects Hopper SM90a and 132 SMs;
an unrecognized GPU=H200 would instead select Blackwell in the pinned Makefile.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upstream import vendor as vendoring  # noqa: E402

UPSTREAM = "7309cec801537b61fea3b50d7dfe454a6cde578e"
THUNDERKITTENS = "664c108d16f12707a73d3072ab525f26fb2b4f62"
URL = "https://github.com/HazyResearch/Megakernels.git"
MODEL_SHA256 = {
  "config.json": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb",
  "model.safetensors": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
  "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
  "tokenizer_config.json": "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
}
# Versions recorded in the preceding pristine H200 audit. The image supplies torch.
DEPENDENCIES = (
  "transformers==4.48.3",
  "pydra-config==0.0.17.post1",
  "accelerate==1.15.0",
  "tabulate==0.9.0",
  "tqdm==4.67.1",
  "einops==0.8.1",
  "pybind11==2.13.6",
  "ninja==1.11.1.3",
  "triton==3.4.0",
  "huggingface_hub==0.36.2",
  "tokenizers==0.21.4",
  "safetensors==0.5.3",
)


def sha256(path: Path) -> str:
  """Hash model/binary files without reading them all into host memory."""
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def verify_model(model: Path) -> dict[str, str]:
  """Reject silently changed weights, tokenizer or model configuration."""
  actual = {name: sha256(model / name) for name in MODEL_SHA256}
  if actual != MODEL_SHA256:
    raise ValueError(f"Model identity mismatch: {actual}")
  return actual


def checked_source(source: Path, vendor: Path | None = None) -> dict[str, Any]:
  """Verify both revision identities and all tracked working-tree/index bytes."""
  result = {}
  for name, root, pin in (
    ("upstream", source, UPSTREAM),
    ("thunderkittens", source / "ThunderKittens", THUNDERKITTENS),
  ):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != pin:
      raise ValueError(f"{name}: expected {pin}, got {head}")
    exclusions = [":(exclude)demos/low-latency-llama", ":(exclude)include"] if vendor and name == "upstream" else []
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ".", *exclusions], cwd=root, check=True)
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    hashes = {os.fsdecode(p): sha256(root / os.fsdecode(p)) for p in files if p and (root / os.fsdecode(p)).is_file()}
    if vendor is not None and name == "upstream":
      hashes.update(vendoring.verify(source, vendor))
    result[name] = {"commit": head, "tracked_sha256": hashes}
  return result


def main() -> None:
  """Build using the pinned author's Makefile and preserve success or failure."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--gpu-uuid", required=True)
  parser.add_argument("--gpu-type", choices=["H200", "H100"], default="H200")
  args = parser.parse_args()
  if not args.gpu_uuid.startswith("GPU-") or "," in args.gpu_uuid:
    parser.error("An explicit single physical GPU UUID is required")
  if "TORCH_CUDA_ARCH_LIST" in os.environ:
    parser.error("Remove inherited TORCH_CUDA_ARCH_LIST before starting this build")
  out, model = args.output.resolve(), args.model.resolve()
  out.mkdir(parents=True, exist_ok=False)
  source, venv = out / "upstream-authors", out / "venv"
  vendor = Path(__file__).resolve().parents[1] / "megakernel"
  report: dict[str, Any] = {
    "passed": False,
    "lineage": "publication-era-vendored-hopper",
    "upstream_commit": UPSTREAM,
    "thunderkittens_commit": THUNDERKITTENS,
    "source_root": str(source),
    "python": str(venv / "bin/python"),
    "model": str(model),
    "gpu_uuid": args.gpu_uuid,
    "gpu_type": args.gpu_type,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "setup_source_sha256": sha256(Path(__file__)),
    "commands": [],
  }
  env = os.environ.copy()
  env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MAX_JOBS="3")

  def run(argv: list[str], *, cwd: Path | None = None, timeout: int = 600) -> str:
    """Run a bounded subprocess and retain its exact argv and complete log."""
    index = len(report["commands"])
    row = {"argv": argv, "cwd": str(cwd) if cwd else None}
    report["commands"].append(row)
    stdout = out / f"command-{index:02}.log"
    with stdout.open("x") as log:
      proc = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    row.update(returncode=proc.returncode, log=stdout.name)
    if proc.returncode:
      raise RuntimeError(f"Command {index} failed; see {stdout}")
    return stdout.read_text()

  try:
    report["model_sha256"] = verify_model(model)
    report["nvcc"] = run(["nvcc", "--version"])
    if "release 12.8" not in report["nvcc"]:
      raise ValueError("This reproduction setup expects NGC25.03 CUDA12.8")
    report["hardware_csv"] = run(
      [
        "nvidia-smi",
        "-i",
        args.gpu_uuid,
        "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
        "--format=csv",
      ]
    )
    probe = (
      "import json,torch; p=torch.cuda.get_device_properties(0); "
      "assert torch.cuda.device_count()==1; "
      "assert torch.cuda.get_device_capability(0)==(9,0); "
      f"assert p.multi_processor_count==132 and {args.gpu_type!r} in p.name; "
      "print(json.dumps(dict(name=p.name,sm_count=p.multi_processor_count,"
      "capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda=torch.version.cuda)))"
    )
    report["device"] = json.loads(run([sys.executable, "-c", probe]))
    run(["git", "clone", "--no-checkout", URL, str(source)])
    run(["git", "checkout", "--detach", UPSTREAM], cwd=source)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=source)
    checked_source(source)
    report["vendored_sha256"] = vendoring.overlay(source, vendor)
    report["source_before"] = checked_source(source, vendor)
    run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
    python = str(venv / "bin/python")
    # The documented container bootstrap installs uv in system-site-packages.
    # Force this small tool into the private venv so its bin/uv exists as well.
    run([python, "-m", "pip", "install", "--ignore-installed", "--no-deps", "uv==0.8.22"])
    uv = str(venv / "bin/uv")
    # uv does not resolve against this image's inherited system-site torch.
    # Install only explicit audited additions; never fetch replacement Torch/CUDA.
    run([uv, "pip", "install", "--python", python, "--no-deps", *DEPENDENCIES])
    run([uv, "pip", "install", "--python", python, "--no-deps", "-e", str(source)])
    run(
      [
        python,
        "-c",
        "import pydra,accelerate; from transformers import AutoTokenizer,AutoModelForCausalLM; "
        "import megakernels.generators,megakernels.dispatch; print('upstream imports passed')",
      ]
    )
    pdl_probe = (
      "import hashlib,inspect,json,triton; "
      "from triton.backends.nvidia import driver,compiler; "
      "assert triton.__version__=='3.4.0'; "
      "launcher=inspect.getsource(driver.CudaLauncher); "
      "generated=driver.make_launcher({}, {}, None); "
      "assert 'launch_pdl' in compiler.CUDAOptions.__dataclass_fields__; "
      "assert 'self.launch_pdl = metadata.launch_pdl' in launcher; "
      "assert 'CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION' in generated; "
      "assert 'launch_pdl' in generated; "
      "print(json.dumps(dict(triton=triton.__version__,launcher_pdl_source_check=True,"
      "driver_sha256=hashlib.sha256(inspect.getsource(driver).encode()).hexdigest(),"
      "scope='Static launcher capability check; GPU execution validated separately')))"
    )
    report["triton_pdl_preflight"] = json.loads(run([python, "-c", pdl_probe]))
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env.update(THUNDERKITTENS_ROOT=str(source / "ThunderKittens"), MEGAKERNELS_ROOT=str(source))
    env["PYTHON_VERSION"] = run(
      [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"]
    ).strip()
    run(["make", "-C", str(source / "demos/low-latency-llama"), "GPU=H100"], timeout=1800)
    binaries = list((source / "demos/low-latency-llama").glob("mk_llama*.so"))
    if len(binaries) != 1:
      raise ValueError(f"Expected exactly one fresh built binary, got {binaries}")
    binary = binaries[0]
    report.update(binary_path=str(binary), binary_sha256=sha256(binary))
    report["binary_elf"] = run(["cuobjdump", "--list-elf", str(binary)])
    report["binary_resources"] = run(["cuobjdump", "--dump-resource-usage", str(binary)])
    if "sm_90" not in report["binary_elf"]:
      raise ValueError("Built binary did not report a Hopper ELF")
    report["packages"] = run([uv, "pip", "freeze", "--python", python])
    runtime_probe = (
      "import json,platform,torch,triton,transformers; "
      "print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,"
      "cuda=torch.version.cuda,triton=triton.__version__,transformers=transformers.__version__)))"
    )
    expected_runtime = json.loads(run([python, "-c", runtime_probe]))
    if (expected_runtime["torch"], expected_runtime["cuda"]) != (report["device"]["torch"], report["device"]["cuda"]):
      raise ValueError("Private dependencies unexpectedly changed the image's PyTorch/CUDA")
    expected_runtime["nvcc"] = report["nvcc"].strip()
    report["source_after"] = checked_source(source, vendor)
    if report["source_before"] != report["source_after"]:
      raise ValueError("Tracked source changed during build")
    metadata = out / "phase3/upstream"
    metadata.mkdir(parents=True)
    runtime_hashes = {}
    for scope, prefix in (("upstream", ""), ("thunderkittens", "ThunderKittens/")):
      for name, digest in report["source_after"][scope]["tracked_sha256"].items():
        if Path(name).suffix in (".py", ".cu", ".cuh", ".h", ".hpp"):
          runtime_hashes[prefix + name] = digest
    runtime_hashes[str(binary.relative_to(source))] = report["binary_sha256"]
    (metadata / "legacy-runtime-sha256.json").write_text(json.dumps(runtime_hashes, indent=2) + "\n")
    (metadata.parent / f"{args.gpu_type}_RUNTIME.json").write_text(
      json.dumps(
        {
          "schema": f"llama1b-{args.gpu_type.lower()}-runtime-v1",
          "expected_runtime": expected_runtime,
          "hardware": {"capability": [9, 0], "name_contains": args.gpu_type},
        },
        indent=2,
      )
      + "\n"
    )
    report["passed"] = True
    (metadata / "source-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
  except BaseException as error:
    report["error"] = f"{type(error).__name__}: {error}"
    raise
  finally:
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / "setup.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  main()
