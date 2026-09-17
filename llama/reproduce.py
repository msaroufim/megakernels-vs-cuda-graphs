"""Run the standalone frozen Hopper reproduction without private metadata inputs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.hardware_profile import bind_hardware


def sha256(path: Path) -> str:
  """Hash complete file bytes, including large model files."""
  value = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 << 20), b""):
      value.update(chunk)
  return value.hexdigest()


def source_commit(repo: Path) -> str:
  """Require the standalone checkout's own committed, clean tracked source."""
  top = Path(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--show-toplevel"], text=True).strip())
  if top.resolve() != repo.resolve():
    raise ValueError("Use this repository's own Git checkout, not a parent repository")
  dirty = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
  ).strip()
  if dirty:
    raise ValueError("Commit or restore changed tracked source before reproducing")
  commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
  if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise ValueError("Expected a full Git source commit")
  return commit


def validate_paths(repo: Path, workdir: Path, model: Path, gpu_uuid: str) -> None:
  """Reject accidental source-tree artifacts, overwrites, and ambiguous devices."""
  if "TORCH_CUDA_ARCH_LIST" in os.environ:
    raise ValueError("Unset TORCH_CUDA_ARCH_LIST; use PyTorch's visible-device resolver")
  if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid):
    raise ValueError("Specify exactly one physical GPU UUID")
  if workdir.exists() or workdir.is_symlink():
    raise FileExistsError(f"Use a fresh workdir: {workdir}")
  if workdir.resolve().is_relative_to(repo.resolve()):
    raise ValueError("Generated artifacts must be outside the source checkout")
  if not workdir.parent.is_dir() or not model.is_dir():
    raise ValueError("The workdir parent and supplied local model directory must exist")


def validate_build(build: Path, model: Path, gpu_type: str = "H200") -> dict:
  """Verify a locally built vendored control and every declared runtime byte."""
  from upstream import prepare

  report = json.loads((build / "setup.json").read_text())
  if not report.get("passed"):
    raise ValueError("The supplied build did not finish successfully")
  if (report.get("upstream_commit"), report.get("thunderkittens_commit")) != (prepare.UPSTREAM, prepare.THUNDERKITTENS):
    raise ValueError("Build revisions differ from the pinned public upstreams")
  if report.get("model_sha256") != prepare.verify_model(model):
    raise ValueError("Build model identity differs from the supplied model")
  bind_hardware("", gpu_type)
  device = report.get("device", {})
  if gpu_type not in device.get("name", "") or device.get("sm_count") != 132 or device.get("capability") != [9, 0]:
    raise ValueError("Build hardware differs from the requested full 132-SM GPU profile")
  runtime_manifest = json.loads((build / f"phase3/{gpu_type}_RUNTIME.json").read_text())
  if runtime_manifest.get("schema") != f"llama1b-{gpu_type.lower()}-runtime-v1" or runtime_manifest.get("hardware") != {
    "capability": [9, 0],
    "name_contains": gpu_type,
  }:
    raise ValueError("Build runtime manifest differs from the requested GPU profile")
  source = build / "upstream-authors"
  python = build / "venv/bin/python"
  if (
    Path(report["source_root"]).resolve() != source.resolve() or Path(report["python"]).absolute() != python.absolute()
  ):
    raise ValueError("Build source/interpreter paths do not belong to this build")
  if not python.is_file():
    raise ValueError("The build interpreter is missing")
  checked = prepare.checked_source(source, Path(__file__).resolve().parent / "megakernel")
  if checked != report["source_before"] or checked != report["source_after"]:
    raise ValueError("Pinned tracked upstream source changed after building")
  binary = Path(report["binary_path"])
  if not binary.resolve().is_relative_to(source.resolve()) or sha256(binary) != report["binary_sha256"]:
    raise ValueError("Control binary identity differs from the successful build")
  expected = {}
  for scope, prefix in (("upstream", ""), ("thunderkittens", "ThunderKittens/")):
    for name, digest in checked[scope]["tracked_sha256"].items():
      if Path(name).suffix in (".py", ".cu", ".cuh", ".h", ".hpp"):
        expected[prefix + name] = digest
  expected[str(binary.relative_to(source))] = report["binary_sha256"]
  manifest = json.loads((build / "phase3/upstream/legacy-runtime-sha256.json").read_text())
  if manifest != expected:
    raise ValueError("Legacy runtime manifest is incomplete or differs from the build")
  for name, digest in manifest.items():
    path = (source / name).resolve()
    if not path.is_relative_to(source.resolve()) or sha256(path) != digest:
      raise ValueError(f"Legacy runtime byte mismatch: {name}")
  provenance = json.loads((build / "phase3/upstream/source-manifest.json").read_text())
  for key in (
    "passed",
    "upstream_commit",
    "thunderkittens_commit",
    "source_after",
    "binary_path",
    "binary_sha256",
    "device",
  ):
    if provenance.get(key) != report[key]:
      raise ValueError(f"Build provenance mismatch: {key}")
  return report


def write_metadata(build: Path, output: Path, gpu_type: str = "H200") -> dict[str, str]:
  """Generate all staging metadata from bundled inputs and the verified build."""
  import campaign
  from tools.prompt_suite import PROMPTS_JSON
  from upstream.prepare import MODEL_SHA256

  files = {
    "phase3/CAMPAIGN.json": (bind_hardware(json.dumps(campaign.plan(), indent=2), gpu_type) + "\n").encode(),
    "phase3/reference/model-sha256.json": (json.dumps(MODEL_SHA256, indent=2) + "\n").encode(),
    "phase3/reference/prompts-v1.json": PROMPTS_JSON.encode() if isinstance(PROMPTS_JSON, str) else PROMPTS_JSON,
  }
  for name in (
    f"phase3/{gpu_type}_RUNTIME.json",
    "phase3/upstream/legacy-runtime-sha256.json",
    "phase3/upstream/source-manifest.json",
  ):
    files[name] = (build / name).read_bytes()
  output.mkdir()
  for name, data in files.items():
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
  return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def command_plan(workdir: Path, model: Path, build: Path, suite: str) -> list[tuple[str, list[str]]]:
  """Describe the serial one-GPU evaluation; command construction is GPU-free."""
  python = str(build / "venv/bin/python")
  runtime = workdir / "runtime"
  fixtures = workdir / "fixtures"
  paired = workdir / "paired"
  checks = workdir / "checks"
  candidate = runtime / "phase3/fp32_headout_fastmath/candidate.py"
  freeze = workdir / "freeze.json"
  commands = [
    (
      "freeze",
      [
        python,
        str(runtime / "phase3/harness/freeze_final.py"),
        "--candidate",
        f"portable={candidate}",
        "--out",
        str(freeze),
      ],
    )
  ]
  reference = [
    python,
    str(runtime / "phase3/reference/generate.py"),
    "--model",
    str(model),
    "--out",
    str(fixtures),
    "--split",
    "validation" if suite == "full" else "development",
  ]
  if suite == "full":
    reference.extend(["--candidate-freeze", str(freeze)])
  commands.append(("references", reference))
  if suite == "smoke":
    fixture = fixtures / "dev_weather/fixture.pt"
    result = paired / "dev_weather-graph-r0"
    commands.extend(
      [
        (
          "matched-smoke",
          [
            python,
            str(runtime / "phase3/harness/matched_final.py"),
            "--model",
            str(model),
            "--candidate",
            str(candidate),
            "--fixture",
            str(fixture),
            "--out",
            str(result),
            "--control",
            "legacy",
            "--mode",
            "graph",
            "--max-len",
            "16384",
            "--samples",
            "10",
            "--warmup",
            "5",
            "--seed",
            "314159",
            "--replicate",
            "0",
            "--mk-dir",
            str(build / "upstream-authors/demos/low-latency-llama"),
          ],
        ),
        (
          "actual-smoke",
          [
            python,
            str(runtime / "phase3/reference/check_execution.py"),
            "--model",
            str(model),
            "--results",
            str(result),
            "--fixture",
            str(fixture),
            "--out",
            str(checks / "dev_weather-graph-r0"),
          ],
        ),
      ]
    )
  elif suite == "full":
    runner = [python, str(runtime / "phase3/harness/run_campaign.py")]
    shared = ["--runtime", str(runtime), "--model", str(model), "--fixtures", str(fixtures)]
    commands.append(
      (
        "paired-full",
        [
          *runner,
          "run-lane",
          "--lane",
          "0",
          *shared,
          "--mk-dir",
          str(build / "upstream-authors/demos/low-latency-llama"),
          "--out",
          str(paired),
        ],
      )
    )
    for lane in (1, 2, 3):
      commands.append(
        (
          f"actual-lane-{lane}",
          [
            *runner,
            "check-lane",
            "--lane",
            str(lane),
            *shared,
            "--results",
            str(paired),
            "--out",
            str(checks / f"lane{lane}"),
          ],
        )
      )
    commands.append(
      (
        "summary",
        [
          python,
          str(runtime / "phase3/harness/summarize.py"),
          "--results-root",
          str(paired),
          "--validation-root",
          str(checks),
          "--split",
          "validation",
          "--min-processes",
          "3",
          "--out",
          str(workdir / "summary.json"),
        ],
      )
    )
  else:
    raise ValueError("Suite must be smoke or full")
  return commands


def main(argv: list[str] | None = None) -> int:
  """Build, freeze and execute immutable outputs; retain failures without retrying."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--workdir", type=Path, required=True)
  parser.add_argument("--gpu-uuid", required=True)
  parser.add_argument("--gpu-type", choices=["H200", "H100"], default="H200")
  parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
  parser.add_argument("--build", type=Path, help="Reuse a verified compatible successful build")
  args = parser.parse_args(argv)
  study = Path(__file__).resolve().parent
  repo = study.parent
  args.model, args.workdir = args.model.resolve(), args.workdir.absolute()
  validate_paths(repo, args.workdir, args.model, args.gpu_uuid)
  args.workdir = args.workdir.resolve()
  commit = source_commit(repo)
  args.workdir.mkdir()
  report: dict[str, Any] = {
    "schema": "llama1b-standalone-reproduction-v1",
    "status": "running",
    "source_commit": commit,
    "runner_sha256": sha256(Path(__file__)),
    "suite": args.suite,
    "gpu_uuid": args.gpu_uuid,
    "gpu_type": args.gpu_type,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "commands": [],
    "interpretation": "Smoke is development-only. Numerical rejection is a preserved result, not a process failure.",
  }
  env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "MAX_JOBS": "3",
  }
  for name in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
    env[name] = str(args.workdir / "caches" / name.lower())

  def save() -> None:
    temporary = args.workdir / "reproduction.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.workdir / "reproduction.json")

  def run(label: str, command: list[str]) -> None:
    log = args.workdir / f"{len(report['commands']):02}-{label}.log"
    row = {"label": label, "argv": command, "log": log.name}
    report["commands"].append(row)
    save()
    print(f"Starting {label}; complete log: {log}", flush=True)
    with log.open("x") as handle:
      process = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    row["returncode"] = process.returncode
    row["log_sha256"] = sha256(log)
    save()
    if process.returncode:
      raise RuntimeError(f"{label} failed; retained command and log at {log}")
    print(f"Completed {label}", flush=True)

  save()
  try:
    build = args.build.resolve() if args.build else args.workdir / "build"
    if args.build is None:
      run(
        "prepare",
        [
          sys.executable,
          str(study / "upstream/prepare.py"),
          "--output",
          str(build),
          "--model",
          str(args.model),
          "--gpu-uuid",
          args.gpu_uuid,
          "--gpu-type",
          args.gpu_type,
        ],
      )
    setup = validate_build(build, args.model, args.gpu_type)
    env["PATH"] = str(build / "venv/bin") + os.pathsep + env.get("PATH", "")
    report.update(build=str(build), build_setup_sha256=sha256(build / "setup.json"), model_sha256=setup["model_sha256"])
    report["metadata_sha256"] = write_metadata(build, args.workdir / "metadata", args.gpu_type)
    from tools.stage_runtime import stage

    stage(study, args.workdir / "metadata", args.workdir / "runtime", commit, args.gpu_type)
    report["stage_manifest_sha256"] = sha256(args.workdir / "runtime/stage-manifest.json")
    runtime_probe = (
      "import importlib.util,pathlib,sys,torch,transformers; "
      "p=pathlib.Path(sys.argv[1]); s=importlib.util.spec_from_file_location('guard',p); "
      "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.validate_hardware(torch); "
      "print(m.validate_runtime(m.runtime_values(torch,transformers),pathlib.Path(sys.argv[2])))"
    )
    run(
      "runtime-preflight",
      [
        str(build / "venv/bin/python"),
        "-c",
        runtime_probe,
        str(args.workdir / f"runtime/phase3/harness/{args.gpu_type.lower()}_runtime.py"),
        str(build / f"phase3/{args.gpu_type}_RUNTIME.json"),
      ],
    )
    for label, command in command_plan(args.workdir, args.model, build, args.suite):
      run(label, command)
    if source_commit(repo) != commit:
      raise RuntimeError("Source checkout changed during reproduction")
    report["status"] = "complete"
    report["freeze_sha256"] = sha256(args.workdir / "freeze.json")
    if args.suite == "full":
      report["summary_sha256"] = sha256(args.workdir / "summary.json")
    else:
      result = args.workdir / "paired/dev_weather-graph-r0/results.json"
      report["smoke_results_sha256"] = sha256(result)
      measured = json.loads(result.read_text())["mean_us_per_step"]
      report["smoke_mean_us_per_step"] = measured
      print(f"Development smoke mean microseconds/token: {json.dumps(measured, sort_keys=True)}", flush=True)
      print("Development smoke only; full-suite numerical and performance acceptance was not evaluated.", flush=True)
    print(f"Completed {args.suite} reproduction: {args.workdir / 'reproduction.json'}")
    return 0
  except Exception as error:
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    print(report["error"], file=sys.stderr)
    return 1
  finally:
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()


if __name__ == "__main__":
  raise SystemExit(main())
