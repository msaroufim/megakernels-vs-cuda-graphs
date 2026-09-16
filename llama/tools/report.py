"""Print existing benchmark receipts without GPU access or reinterpreting acceptance."""

import argparse
import json
import statistics
from pathlib import Path


def render(workdir: Path) -> str:
  """Summarize complete measured processes, keeping graph/native scopes separate."""
  grouped: dict[str, list[dict]] = {}
  incomplete = []
  identities: set[tuple[str, str, int]] = set()
  for path in sorted((workdir / "paired").glob("*/results.json")):
    run = json.loads(path.read_text())
    if run.get("status") != "complete":
      incomplete.append(path.parent.name)
      continue
    gpu = run.get("gpu", {})
    if not gpu.get("name") or not gpu.get("uuid") or not isinstance(gpu.get("sms"), int):
      raise ValueError("Receipt is missing observed GPU identity")
    identities.add((gpu["name"], gpu["uuid"], gpu["sms"]))
    grouped.setdefault(run["configuration"]["mode"], []).append(run)
  if not grouped:
    raise ValueError("No complete matched results found under workdir/paired")
  if len(identities) != 1:
    raise ValueError("Do not pool different GPU hardware or UUIDs in one report")
  name, uuid, sms = next(iter(identities))
  lines = [
    f"Observed GPU: {name}; {sms} SMs; UUID {uuid}.",
    "Descriptive timings; numerical decisions are reported separately.",
    "| Mode | Paired processes | Conventional us/step | Hazy us/step |",
    "|---|---:|---:|---:|",
  ]
  for mode, runs in sorted(grouped.items()):
    means = {
      arm: statistics.mean(statistics.mean(run["timing_us"][arm]) / 127 for run in runs)
      for arm in ("candidate", "control")
    }
    lines.append(f"| {mode} | {len(runs)} | {means['candidate']:.2f} | {means['control']:.2f} |")
  if incomplete:
    lines.append("Incomplete receipts retained: " + ", ".join(incomplete))
  summary = workdir / "summary.json"
  if summary.is_file():
    for group in json.loads(summary.read_text())["groups"]:
      value = group["summary"]
      mode = group["group"]["mode"]
      lines.append(
        f"\n{mode}: complete primary matrix={value['complete_matrix']}; "
        f"actual-output coverage complete={value['realized_output_validation_complete']}"
      )
      for arm, fields in value["numerical"].items():
        canonical = all(row["accepted"] for metrics in fields.values() for row in metrics.values())
        actual = value["realized_output_validation"]["by_arm_stage"].get(arm, {})
        accepted = (
          value["realized_output_validation_complete"]
          and bool(actual)
          and all(row["accepted"] for metrics in actual.values() for row in metrics.values())
        )
        label = "Conventional" if arm == "candidate" else "Hazy"
        lines.append(f"{label}: canonical criteria={canonical}; actual-output criteria={accepted}")
      lines.append(
        f"Conventional/Hazy throughput ratio={value['geometric_mean_speedup']}; "
        f"95% CI={value['hierarchical_prompt_process_bootstrap95']}"
      )
      lines.append(f"Original candidate-win rule satisfied={value['accepted_speed_win']}")
  else:
    lines.append("\nDevelopment smoke: no full-suite statistical or numerical acceptance claim.")
  return "\n".join(lines)


def main() -> None:
  """Read an existing work directory and print a human-readable result table."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workdir", type=Path, required=True)
  args = parser.parse_args()
  print(render(args.workdir))


if __name__ == "__main__":
  main()
