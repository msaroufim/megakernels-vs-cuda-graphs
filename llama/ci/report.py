"""Render numeric results without copying sandbox text into PR comments."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

MARKER = "<!-- llama-performance-ci -->"
SCHEMA = "llama-performance-ci-v1"
ARMS = {"cuda_graph_pdl": "CUDA Graphs + PDL", "megakernel": "Megakernel"}
VERDICTS = {"faster": "Faster", "slower": "Slower", "inconclusive": "Inconclusive"}
MAX_REPORT_BYTES = 8 * 1024 * 1024
METRICS = {
  "canonical_kl_delta": "Canonical KL",
  "canonical_abs_nll_error_delta": "Canonical absolute NLL",
  "actual_kl_delta": "Timed KL",
  "actual_abs_nll_error_delta": "Timed absolute NLL",
}


def _number(value: Any, *, positive: bool = False) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError("Expected a finite number")
  result = float(value)
  if not math.isfinite(result) or (positive and result <= 0):
    raise ValueError("Expected a finite number")
  return result


def _rows(report: Any) -> tuple[list[str], list[str]]:
  if not isinstance(report, dict) or report.get("schema") != SCHEMA or report.get("status") != "complete":
    raise ValueError("Incomplete report")
  comparisons = report.get("comparisons")
  if not isinstance(comparisons, dict):
    raise ValueError("Missing comparisons")
  rows, paired_rows = [], []
  pair_count = None
  for arm, label in ARMS.items():
    comparison = comparisons.get(arm)
    if not isinstance(comparison, dict):
      raise ValueError("Missing arm")
    base = _number(comparison.get("base_us"), positive=True)
    head = _number(comparison.get("head_us"), positive=True)
    delta = _number(comparison.get("latency_change_pct"))
    paired = comparison.get("paired_latency_change_pct")
    if not isinstance(paired, list) or not 3 <= len(paired) <= 10:
      raise ValueError("Expected three to ten paired latency changes")
    pairs = [_number(value) for value in paired]
    if any(value <= -100 for value in pairs) or (pair_count is not None and len(pairs) != pair_count):
      raise ValueError("Invalid paired latency changes")
    pair_count = len(pairs)
    paired_rows.append(f"{label} pairs: " + ", ".join(f"{value:+.2f}%" for value in pairs) + ".")
    interval = comparison.get("latency_change_ci95_pct")
    if not isinstance(interval, list) or len(interval) != 2:
      raise ValueError("Missing confidence interval")
    low, high = (_number(item) for item in interval)
    verdict = comparison.get("verdict")
    if not isinstance(verdict, str) or verdict not in VERDICTS or low > high:
      raise ValueError("Invalid comparison")
    expected_verdict = "faster" if high < 0 else "slower" if low > 0 else "inconclusive"
    if verdict != expected_verdict:
      raise ValueError("Verdict disagrees with confidence interval")
    rows.append(
      f"| {label} | {base:.2f} | {head:.2f} | {delta:+.2f}% | [{low:+.2f}%, {high:+.2f}%] | {VERDICTS[verdict]} |"
    )
  return rows, paired_rows


def _numerics(report: dict) -> tuple[list[str], int]:
  rows, flagged = [], 0
  for arm, label in ARMS.items():
    numeric = report["comparisons"][arm].get("numerics")
    if not isinstance(numeric, dict):
      raise ValueError("Missing numerical checks")
    regressions = numeric.get("regressions")
    if not isinstance(regressions, list) or any(not isinstance(x, str) or x not in METRICS for x in regressions):
      raise ValueError("Invalid numerical flags")
    if len(set(regressions)) != len(regressions):
      raise ValueError("Duplicate numerical flags")
    for metric, metric_label in METRICS.items():
      base = _number(numeric["base"][metric])
      head = _number(numeric["head"][metric])
      difference = numeric["head_minus_base"][metric]
      delta = _number(difference["mean"])
      bounds = difference["ci95"]
      if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Missing numerical interval")
      low, high = (_number(x) for x in bounds)
      if low > high or (low > 0) != (metric in regressions):
        raise ValueError("Inconsistent numerical flag")
      flagged += int(low > 0)
      rows.append(f"| {label} | {metric_label} | {base:+.3g} | {head:+.3g} | {delta:+.3g} [{low:+.3g}, {high:+.3g}] |")
  return rows, flagged


def render(report: Any, *, base_sha: str, head_sha: str, run_url: str) -> str:
  """Fail closed on malformed measurements; emit only fixed prose and numeric cells."""
  if not all(re.fullmatch(r"[a-fA-F0-9]{40}", sha) for sha in (base_sha, head_sha)):
    raise ValueError("Invalid commit SHA")
  if not re.fullmatch(r"https://[A-Za-z0-9.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+", run_url):
    raise ValueError("Invalid Actions URL")
  lines = [
    MARKER,
    "### Llama performance",
    "",
    f"Base `{base_sha[:12]}` → PR `{head_sha[:12]}` · [Run and artifacts]({run_url})",
    "",
  ]
  try:
    rows, paired_rows = _rows(report)
    numerical_rows, flagged = _numerics(report)
  except (ValueError, OverflowError, TypeError, KeyError):
    lines.append(
      "Benchmark failed or did not produce a complete, valid report. No performance conclusion is available."
    )
  else:
    lines.extend(
      [
        "| Implementation | Base µs/step | PR µs/step | Latency change | Paired t 95% CI | Result |",
        "|---|---:|---:|---:|---:|---|",
        *rows,
        "",
        "Negative latency change is faster; positive is slower.",
        "",
        *paired_rows,
        "",
        "Latency intervals use paired Student-t on log(PR/base), with n−1 degrees of freedom. "
        "With only 3–10 pairs, they depend on independent, approximately normal log ratios. "
        "Treat them as advisory, not a published accuracy-qualified result.",
        "",
        f"Numerical review flags: **{flagged}/8** metrics have a positive PR−base 95% interval.",
        "Errors below are relative to native BF16 HF; lower is better. Existing baseline error is not a new regression.",
        "",
        "| Implementation | Error metric | Base | PR | PR−base [95% CI] |",
        "|---|---|---:|---:|---:|",
        *numerical_rows,
        "",
        "This is a development check on one prompt. See artifacts for individual runs; numerical fidelity is separate from latency.",
      ]
    )
  lines.extend(["", "Manual code review and merge required; this workflow never merges a PR.", ""])
  return "\n".join(lines)


def read_report(path: Path) -> Any:
  try:
    if path.stat().st_size > MAX_REPORT_BYTES:
      return None
    return json.loads(path.read_text())
  except (OSError, UnicodeError, ValueError, RecursionError):
    return None


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("report", type=Path)
  parser.add_argument("--base-sha", required=True)
  parser.add_argument("--head-sha", required=True)
  parser.add_argument("--run-url", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  comment = render(read_report(args.report), base_sha=args.base_sha, head_sha=args.head_sha, run_url=args.run_url)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(comment)


if __name__ == "__main__":
  main()
