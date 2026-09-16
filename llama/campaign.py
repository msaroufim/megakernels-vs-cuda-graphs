"""Write the predeclared H200 campaign and run one explicitly assigned lane."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = (
  "val_archaeology",
  "val_baking",
  "val_astronomy",
  "val_transit",
  "val_language",
  "val_ecology",
  "val_materials",
  "val_history",
)


def plan() -> dict:
  """Return the legacy-only H200 coverage declared before reserved outcomes."""
  return {
    "study": "llama1b-h200-v1",
    "status": "predeclared-before-h200-reserved-outcomes",
    "candidate": "portable-fp32-headout-fastmath",
    "candidate_path": "phase3/fp32_headout_fastmath/candidate.py",
    "primary_controls": ["legacy"],
    "primary_modes": ["graph"],
    "prompts": list(PROMPTS),
    "fresh_processes_per_prompt": 3,
    "warmup_sequences": 5,
    "measured_sequences": 10,
    "kv_capacity": 16384,
    "seed_rule": "314159 + 100*prompt_index_in_reserved_list + replicate_index",
    "sequence_glue": "same fast two-stage argmax and next embedding for both arms",
    "output_precision": {"candidate": "FP32 logits, BF16 stored activations", "legacy": "released BF16"},
    "native_diagnostic": {
      "prompt": PROMPTS[0],
      "controls": ["legacy"],
      "fresh_processes": 3,
      "warmups": 5,
      "samples": 10,
      "primary_acceptance_eligible": False,
    },
    "physical_lanes": {
      "0": "all eight paired graph prompts and the native diagnostic; one complete same-GPU group",
      "1": "development then independent realized-history validation",
      "2": "independent realized-history validation",
      "3": "independent H200 reference and realized-history validation",
    },
    "excluded_control": "mk-v2-llama requires Blackwell; no unchanged H200 implementation",
    "claim_rule": (
      "Complete coverage and original symmetric numerical/execution criteria plus paired latency CI excluding parity."
    ),
  }


def run_lane(args: argparse.Namespace) -> None:
  """Execute fresh paired processes serially; preserve failed run directories."""
  if json.loads((args.runtime / "phase3/CAMPAIGN.json").read_text()) != plan():
    raise ValueError("Staged campaign differs from the declared H200 plan")
  gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
    raise ValueError("Set CUDA_VISIBLE_DEVICES to the one physical UUID assigned in the ledger")
  args.out.mkdir(parents=True, exist_ok=False)
  for name in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
    os.environ[name] = str(args.out / "caches" / name.lower())
  commands = []
  for index, prompt in enumerate(PROMPTS):
    for mode in ["graph", "native"] if index == 0 else ["graph"]:
      for replicate in range(3):
        result = args.out / f"{prompt}-{mode}-r{replicate}"
        command = [
          sys.executable,
          str(args.runtime / "phase3/harness/matched_final.py"),
          "--model",
          str(args.model),
          "--candidate",
          str(args.runtime / "phase3/fp32_headout_fastmath/candidate.py"),
          "--fixture",
          str(args.fixtures / prompt / "fixture.pt"),
          "--mk-dir",
          str(args.mk_dir),
          "--control",
          "legacy",
          "--mode",
          mode,
          "--max-len",
          "16384",
          "--warmup",
          "5",
          "--samples",
          "10",
          "--seed",
          str(314159 + 100 * index + replicate),
          "--replicate",
          str(replicate),
          "--out",
          str(result),
        ]
        before = idle_boundary(gpu_uuid)
        with (args.out / f"{result.name}.log").open("x") as log:
          process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        commands.append({"argv": command, "returncode": process.returncode, "boundary_before": before})
        (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if process.returncode:
          raise RuntimeError(f"Matched process failed; retained {result} and its log")
        commands[-1]["boundary_after"] = idle_boundary(gpu_uuid)
        (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def idle_boundary(gpu_uuid: str) -> dict:
  """Require an empty assigned GPU at process boundaries and retain telemetry."""
  processes = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True
  )
  assigned = [row for row in csv.reader(processes.splitlines()) if row and row[0].strip() == gpu_uuid]
  if assigned:
    raise RuntimeError(f"Assigned GPU has another live process: {assigned}")
  telemetry = subprocess.check_output(
    [
      "nvidia-smi",
      "-i",
      gpu_uuid,
      "--query-gpu=uuid,name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
      "--format=csv,noheader",
    ],
    text=True,
  )
  return {"gpu_uuid": gpu_uuid, "assigned_processes": assigned, "telemetry": telemetry.strip()}


def check_lane(args: argparse.Namespace) -> None:
  """Validate one third of complete receipts on a distinct assigned H200."""
  if json.loads((args.runtime / "phase3/CAMPAIGN.json").read_text()) != plan():
    raise ValueError("Staged campaign differs from the declared H200 plan")
  gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
    raise ValueError("Expose the single checker GPU UUID assigned in the ledger")
  args.out.mkdir(parents=True, exist_ok=False)
  cases = [
    (prompt, mode, replicate)
    for index, prompt in enumerate(PROMPTS)
    for mode in (["graph", "native"] if index == 0 else ["graph"])
    for replicate in range(3)
  ]
  deadline = time.monotonic() + args.timeout
  commands = []
  for index, (prompt, mode, replicate) in enumerate(cases):
    if index % 3 != args.lane - 1:
      continue
    name = f"{prompt}-{mode}-r{replicate}"
    result = args.results / name
    while True:
      path = result / "results.json"
      if path.is_file() and json.loads(path.read_text()).get("status") == "complete":
        break
      if time.monotonic() >= deadline:
        raise TimeoutError(f"No complete matched receipt before checker deadline: {result}")
      time.sleep(5)
    idle_boundary(gpu_uuid)
    command = [
      sys.executable,
      str(args.runtime / "phase3/reference/check_execution.py"),
      "--model",
      str(args.model),
      "--results",
      str(result),
      "--fixture",
      str(args.fixtures / prompt / "fixture.pt"),
      "--out",
      str(args.out / name),
    ]
    with (args.out / f"{name}.log").open("x") as log:
      process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    commands.append({"argv": command, "returncode": process.returncode, "gpu_uuid": gpu_uuid})
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    if process.returncode:
      raise RuntimeError(f"Independent checker failed; retained {args.out / name}")
    print(f"Validated {name}", flush=True)


def main() -> None:
  """Create external metadata or execute a predeclared measurement lane."""
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  write = subparsers.add_parser("plan")
  write.add_argument("--out", type=Path, required=True)
  run = subparsers.add_parser("run-lane")
  run.add_argument("--lane", type=int, choices=[0], required=True)
  for field in ("runtime", "model", "fixtures", "mk-dir", "out"):
    run.add_argument("--" + field, type=Path, required=True)
  check = subparsers.add_parser("check-lane")
  check.add_argument("--lane", type=int, choices=[1, 2, 3], required=True)
  check.add_argument("--timeout", type=float, default=5400)
  for field in ("runtime", "model", "fixtures", "results", "out"):
    check.add_argument("--" + field, type=Path, required=True)
  args = parser.parse_args()
  if args.command == "plan":
    with args.out.open("x") as output:
      output.write(json.dumps(plan(), indent=2) + "\n")
  elif args.command == "run-lane":
    run_lane(args)
  else:
    check_lane(args)


if __name__ == "__main__":
  main()
