"""Development regression measurements in an isolated, credential-free H100 sandbox.

The calling service supplies read-only snapshots, model files and local Git mirrors.
Only this runner's harness/reference code is used. Proposed kernels and upstream
builds are executable untrusted code: isolation is the caller's responsibility.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
from tools.hardware_profile import bind_hardware  # noqa: E402
from tools.stage_runtime import stage  # noqa: E402
from upstream.prepare import sha256, verify_model  # noqa: E402
from upstream.vendor import overlay  # noqa: E402

ARMS = {"cuda_graph_pdl": "candidate", "megakernel": "control"}
METRICS = ("kl", "abs_nll_error")
# Two-sided 95% Student-t critical values for the supported 3..10 process pairs.
T975 = (
  4.302652729696142,
  3.182446305284263,
  2.7764451051977987,
  2.570581835636305,
  2.446911848791681,
  2.3646242510102993,
  2.306004135204166,
  2.2621571628540993,
)
INTERVAL_ASSUMPTIONS = (
  "Independent process-pair effects with an approximately normal distribution; "
  "the small sample cannot establish these assumptions. Advisory single-prompt "
  "intervals do not cover GPU/session variation or multiple-comparison selection."
)


def pins(snapshot: Path) -> dict[str, str]:
  """Read literal commit declarations without importing proposed Python code."""
  path = contained(snapshot, "llama/upstream/prepare.py")
  values = {}
  for node in ast.parse(path.read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
      name = node.targets[0].id
      if name in ("UPSTREAM", "THUNDERKITTENS"):
        if name in values or not isinstance(node.value, ast.Constant):
          raise ValueError(f"{name} must have one literal assignment")
        value = node.value.value
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
          raise ValueError(f"{name} must be a full lowercase Git commit")
        values[name] = value
  if set(values) != {"UPSTREAM", "THUNDERKITTENS"}:
    raise ValueError("Both upstream commit declarations are required")
  return values


def contained(root: Path, name: str) -> Path:
  path = root / name
  if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
    raise ValueError(f"Missing, linked or escaping file: {name}")
  return path


def source_files(root: Path) -> dict[str, str]:
  """Hash complete source trees, excluding Git internals and generated caches."""
  result = {}
  for path in sorted(root.rglob("*")):
    if any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts):
      continue
    if path.is_symlink():
      raise ValueError(f"Source symlinks are unsupported: {path}")
    if path.is_file():
      result[str(path.relative_to(root))] = sha256(path)
  return result


def interval(values: list[float], *, ratio: bool = False) -> tuple[float, list[float]]:
  """Student-t interval over paired process effects, optionally on the log scale."""
  if not 3 <= len(values) <= 10 or not all(math.isfinite(x) for x in values):
    raise ValueError("Three to ten finite process pairs are required")
  mean = statistics.mean(values)
  radius = T975[len(values) - 3] * statistics.stdev(values) / math.sqrt(len(values))
  transform = (lambda x: 100 * math.expm1(x)) if ratio else (lambda x: x)
  try:
    estimate, lower, upper = [transform(x) for x in (mean, mean - radius, mean + radius)]
  except OverflowError as error:
    raise ValueError("Nonfinite interval") from error
  if not all(math.isfinite(x) for x in (estimate, lower, upper)):
    raise ValueError("Nonfinite interval")
  return estimate, [lower, upper]


def numerical_summary(matched: dict, checked: dict, arm: str) -> dict:
  """Canonical and actual timed-output errors relative to native BF16 HF."""
  if matched.get("status") != "complete" or checked.get("status") != "complete":
    raise ValueError("Incomplete timing or numerical validation")
  for key in ("all_actual_final_logits_finite", "all_reference_logits_finite", "all_token_boundaries_valid"):
    if checked.get(key) is not True:
      raise ValueError(f"Execution validation failed: {key}")
  reference = matched["bf16_calibration"]
  repetitions = matched["teacher_forced_graph"][arm]
  if len(reference) != 127 or len(repetitions) != 3 or any(len(rows) != 127 for rows in repetitions):
    raise ValueError("Incomplete canonical graph validation")
  values: dict[str, float] = {}
  for metric in METRICS:
    deltas = [
      row[metric] - reference[index][metric]
      for rows in repetitions
      for index, row in enumerate(rows)
      if row.get("finite") and reference[index].get("finite")
    ]
    if len(deltas) != 381 or not all(math.isfinite(x) for x in deltas):
      raise ValueError("Nonfinite canonical metrics")
    values[f"canonical_{metric}_delta"] = statistics.mean(deltas)
  actual: dict[str, list[float]] = {metric: [] for metric in METRICS}
  for entry in checked["unique_executions"].values():
    for observation in entry["observations"]:
      if observation["arm"] != arm or not observation["stage"].startswith("timed-"):
        continue
      row = entry["metrics_by_selected_token"][str(observation["selected_final_token"])]["paired_error_differences"]
      actual["kl"].append(row["kl_error_minus_native_bf16"])
      actual["abs_nll_error"].append(row["absolute_nll_error_minus_native_bf16"])
  for metric, observations in actual.items():
    if len(observations) != 10 or not all(math.isfinite(x) for x in observations):
      raise ValueError("Expected ten finite independently checked timed outputs per arm")
    values[f"actual_{metric}_delta"] = statistics.mean(observations)
  return values


def summarize(runs: list[dict]) -> dict:
  groups = {side: sorted((r for r in runs if r["side"] == side), key=lambda r: r["pair"]) for side in ("base", "head")}
  if len(groups["base"]) < 3 or [r["pair"] for r in groups["base"]] != [r["pair"] for r in groups["head"]]:
    raise ValueError("Missing paired processes")
  if len({r["pair"] for r in groups["base"]}) != len(groups["base"]):
    raise ValueError("Duplicate process pair")
  comparisons = {}
  for name, arm in ARMS.items():
    base = [r["mean_us_per_step"][arm] for r in groups["base"]]
    head = [r["mean_us_per_step"][arm] for r in groups["head"]]
    if any(not math.isfinite(x) or x <= 0 for x in base + head):
      raise ValueError("Latencies must be positive and finite")
    paired_log_ratios = [math.log(h) - math.log(b) for b, h in zip(base, head)]
    change, ci = interval(paired_log_ratios, ratio=True)
    numeric = {side: {} for side in ("base", "head", "head_minus_base")}
    regressions = []
    for metric in groups["base"][0]["numerics"][arm]:
      b = [r["numerics"][arm][metric] for r in groups["base"]]
      h = [r["numerics"][arm][metric] for r in groups["head"]]
      numeric["base"][metric] = statistics.mean(b)
      numeric["head"][metric] = statistics.mean(h)
      delta, bounds = interval([y - x for x, y in zip(b, h)])
      numeric["head_minus_base"][metric] = {"mean": delta, "ci95": bounds}
      if bounds[0] > 0:
        regressions.append(metric)
    comparisons[name] = {
      "base_us": statistics.mean(base),
      "head_us": statistics.mean(head),
      "latency_change_pct": change,
      "latency_change_ci95_pct": ci,
      "paired_latency_change_pct": [100 * math.expm1(x) for x in paired_log_ratios],
      "interval": {
        "method": "paired-student-t",
        "confidence": 0.95,
        "degrees_of_freedom": len(base) - 1,
        "latency_scale": "log(head/base)",
        "numerical_scale": "head-base",
        "assumptions": INTERVAL_ASSUMPTIONS,
      },
      "verdict": "faster" if ci[1] < 0 else "slower" if ci[0] > 0 else "inconclusive",
      "numerics": {**numeric, "regressions": regressions},
    }
  return comparisons


class Runner:
  def __init__(self, output: Path, model: Path, cache: Path):
    self.output, self.model, self.cache = output, model, cache
    self.report: dict[str, Any] = {
      "schema": "llama-performance-ci-v1",
      "status": "running",
      "interpretation": (
        "Development regression check on one prompt; not full-study qualification. "
        "Student-t intervals use paired processes, not tokens, and assume independent, approximately normal pair effects. Numerical deltas compare "
        "against native BF16 HF; existing baseline misses are not new regressions. "
        "Positive head-minus-base numerical intervals are review flags; no automatic merge."
      ),
      "commands": [],
      "snapshots": {},
      "runs": [],
      "comparisons": {},
      "failures": [],
    }
    self.env: dict[str, str] = {
      **os.environ,
      "OMP_NUM_THREADS": "4",
      "MKL_NUM_THREADS": "4",
      "MAX_JOBS": "3",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1",
      "PYTHON_VERSION": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    self.env.pop("TORCH_CUDA_ARCH_LIST", None)

  def save(self) -> None:
    temporary = self.output / "report.tmp"
    temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
    temporary.replace(self.output / "report.json")

  def run(self, label: str, command: list[str], *, env: dict[str, str] | None = None, timeout: int = 1200) -> str:
    log = self.output / f"{len(self.report['commands']):02}-{label}.log"
    row = {"label": label, "argv": command, "log": log.name}
    self.report["commands"].append(row)
    self.save()
    print(label, flush=True)
    started = time.monotonic()
    failure = None
    try:
      with log.open("w") as handle:
        result = subprocess.run(command, env=env or self.env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
      row["returncode"] = result.returncode
      if result.returncode:
        failure = f"exit {result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as error:
      failure = f"{type(error).__name__}: {error}"
      row["error"] = failure
    finally:
      row["duration_seconds"] = time.monotonic() - started
      row["log_sha256"] = sha256(log)
      self.save()
    if failure:
      tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])[-6000:]
      print(f"{label} failed ({failure}); {log.name} tail:\n{tail}", flush=True)
      raise RuntimeError(f"{label} failed ({failure}); see {log.name}")
    return log.read_text(errors="replace")

  def build(self, side: str, snapshot: Path, runtime_values: dict) -> tuple[Path, Path, dict]:
    declarations = pins(snapshot)
    root = self.output / side
    root.mkdir()
    upstream = root / "upstream"
    for repo, destination, revision in (
      ("Megakernels", upstream, declarations["UPSTREAM"]),
      ("ThunderKittens", upstream / "ThunderKittens", declarations["THUNDERKITTENS"]),
    ):
      self.run(
        f"{side}-{repo}-clone",
        ["git", "clone", "--no-hardlinks", "--no-checkout", str(self.cache / f"{repo}.git"), str(destination)],
      )
      self.run(f"{side}-{repo}-checkout", ["git", "-C", str(destination), "checkout", "--detach", revision])
    vendor = snapshot / "llama/megakernel"
    vendored = overlay(upstream, vendor) if vendor.exists() else {}
    identity = {"pins": declarations, "vendored_sha256": vendored, "source_sha256": source_files(upstream)}
    env = {**self.env, "THUNDERKITTENS_ROOT": str(upstream / "ThunderKittens"), "MEGAKERNELS_ROOT": str(upstream)}
    env["PYTHONPATH"] = str(upstream)
    for name in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
      env[name] = str(root / "caches" / name.lower())
    self.run(
      f"{side}-build", ["make", "-C", str(upstream / "demos/low-latency-llama"), "GPU=H100"], env=env, timeout=1800
    )
    binaries = list((upstream / "demos/low-latency-llama").glob("mk_llama*.so"))
    if len(binaries) != 1:
      raise ValueError("Expected one fresh megakernel binary")
    current = source_files(upstream)
    if any(current.get(name) != digest for name, digest in identity["source_sha256"].items()):
      raise ValueError("Source changed during build")
    identity["binary_sha256"] = sha256(binaries[0])
    metadata = root / "metadata"
    import campaign
    from tools.prompt_suite import PROMPTS_JSON
    from upstream.prepare import MODEL_SHA256

    files = {
      "phase3/CAMPAIGN.json": bind_hardware(json.dumps(campaign.plan()), "H100"),
      "phase3/reference/model-sha256.json": json.dumps(MODEL_SHA256),
      "phase3/reference/prompts-v1.json": PROMPTS_JSON,
      "phase3/upstream/legacy-runtime-sha256.json": json.dumps(
        {
          name: digest
          for name, digest in current.items()
          if Path(name).suffix in {".py", ".cu", ".cuh", ".h", ".hpp", ".so"}
        }
      ),
      "phase3/upstream/source-manifest.json": json.dumps(identity),
      "phase3/H100_RUNTIME.json": json.dumps(
        {
          "schema": "llama1b-h100-runtime-v1",
          "expected_runtime": runtime_values,
          "hardware": {"capability": [9, 0], "name_contains": "H100"},
        }
      ),
    }
    for name, content in files.items():
      path = metadata / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(content)
    runtime = root / "runtime"
    stage(STUDY, metadata, runtime, "0" * 40, "H100")
    target = runtime / "phase3/fp32_headout_fastmath"
    shutil.rmtree(target)
    kernel_root = snapshot / "llama/kernels"
    kernel_hashes = source_files(kernel_root)
    if "candidate.py" not in kernel_hashes:
      raise ValueError("Missing candidate.py")
    for name in kernel_hashes:
      path = target / name
      path.parent.mkdir(parents=True, exist_ok=True)
      data = contained(kernel_root, name).read_bytes()
      if name == "candidate.py":
        data = data.replace(b"native_bf16_fma: bool = True", b"native_bf16_fma: bool = False")
      path.write_bytes(data)
    manifest_path = runtime / "stage-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["staged_sha256"] = source_files(runtime)
    manifest["staged_sha256"].pop("stage-manifest.json", None)
    manifest["ci_candidate_sha256"] = kernel_hashes
    manifest_path.write_text(json.dumps(manifest, indent=2))
    identity["kernels_sha256"] = kernel_hashes
    self.report["snapshots"][side] = identity
    self.save()
    return runtime, upstream, env

  def execute(self, base: Path, head: Path, pairs: int) -> None:
    probe = (
      "import json,torch,transformers; from tools.h200_runtime import runtime_values; "
      "p=torch.cuda.get_device_properties(0); "
      "assert torch.cuda.device_count()==1 and 'H100' in p.name and p.multi_processor_count==132; "
      "assert torch.cuda.get_device_capability()==(9,0); "
      "r=runtime_values(torch,transformers); "
      "assert r['torch']=='2.7.0a0+7c8ec84dab.nv25.03' and r['cuda']=='12.8'; "
      "assert r['triton']=='3.4.0' and r['transformers']=='4.48.3'; "
      "assert 'release 12.8' in r['nvcc']; "
      "print(json.dumps(dict(runtime=r,gpu=dict(name=p.name,uuid=str(p.uuid),sms=p.multi_processor_count))))"
    )
    self.report["configuration"] = {
      "prompt": "dev_weather",
      "process_pairs": pairs,
      "warmup_sequences": 5,
      "timed_sequences": 10,
      "steps_per_sequence": 127,
      "kv_capacity": 16384,
      "process_order": "base/head on even pairs; head/base on odd pairs",
      "seed_rule": "314159 + pair",
      "interval_method": "paired-student-t",
    }
    self.report["model_sha256"] = verify_model(self.model)
    self.report["device"] = json.loads(
      self.run("preflight", [sys.executable, "-c", probe], env={**self.env, "PYTHONPATH": str(STUDY)})
    )
    builds = {
      side: self.build(side, snapshot, self.report["device"]["runtime"])
      for side, snapshot in (("base", base), ("head", head))
    }
    runtime, _, env = builds["base"]
    fixtures = self.output / "fixtures"
    self.run(
      "references",
      [
        sys.executable,
        str(runtime / "phase3/reference/generate.py"),
        "--model",
        str(self.model),
        "--out",
        str(fixtures),
        "--split",
        "development",
      ],
      env=self.env,
    )
    fixture = fixtures / "dev_weather/fixture.pt"
    self.report["fixture_sha256"] = sha256(fixture)
    for pair in range(pairs):
      for side in ("base", "head") if pair % 2 == 0 else ("head", "base"):
        runtime, upstream, env = builds[side]
        result = self.output / f"{side}-pair-{pair}"
        checked = self.output / f"{side}-check-{pair}"
        command = [
          sys.executable,
          str(runtime / "phase3/harness/matched_final.py"),
          "--model",
          str(self.model),
          "--candidate",
          str(runtime / "phase3/fp32_headout_fastmath/candidate.py"),
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
          str(314159 + pair),
          "--replicate",
          str(pair),
          "--mk-dir",
          str(upstream / "demos/low-latency-llama"),
        ]
        self.run(f"{side}-pair-{pair}", command, env=env)
        self.run(
          f"{side}-check-{pair}",
          [
            sys.executable,
            str(runtime / "phase3/reference/check_execution.py"),
            "--model",
            str(self.model),
            "--results",
            str(result),
            "--fixture",
            str(fixture),
            "--out",
            str(checked),
          ],
          env=self.env,
        )
        measured = json.loads((result / "results.json").read_text())
        validation = json.loads((checked / "results.json").read_text())
        self.report["runs"].append(
          {
            "side": side,
            "pair": pair,
            "mean_us_per_step": measured["mean_us_per_step"],
            "timing_us": measured["timing_us"],
            "orders": measured["orders"],
            "numerics": {arm: numerical_summary(measured, validation, arm) for arm in ARMS.values()},
            "matched_sha256": sha256(result / "results.json"),
            "checked_sha256": sha256(checked / "results.json"),
            "telemetry_before": measured["telemetry_before"],
            "telemetry_after": measured["telemetry_after"],
          }
        )
        self.save()
    self.report["comparisons"] = summarize(self.report["runs"])
    self.report["status"] = "complete"


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("base", "head", "model", "output", "upstream-cache"):
    parser.add_argument(f"--{name}", type=Path, required=True)
  parser.add_argument("--pairs", type=int, default=3)
  args = parser.parse_args(argv)
  if not 3 <= args.pairs <= 10:
    parser.error("Use 3 to 10 process pairs")
  args.output.mkdir(parents=True, exist_ok=False)
  runner = Runner(args.output.resolve(), args.model.resolve(), args.upstream_cache.resolve())
  try:
    runner.execute(args.base.resolve(), args.head.resolve(), args.pairs)
  except Exception as error:
    runner.report["status"] = "failed"
    runner.report["failures"].append(f"{type(error).__name__}: {error}")
  finally:
    runner.save()
  return 0 if runner.report["status"] == "complete" else 1


if __name__ == "__main__":
  raise SystemExit(main())
