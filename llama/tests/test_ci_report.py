import json

import pytest

from ci.report import MARKER, MAX_REPORT_BYTES, METRICS, read_report, render

BASE = "a" * 40
HEAD = "b" * 40
RUN = "https://github.com/example/research/actions/runs/123"


def report():
  data = {
    "schema": "llama-performance-ci-v1",
    "status": "complete",
    "comparisons": {
      "cuda_graph_pdl": {
        "base_us": 100,
        "head_us": 90,
        "latency_change_pct": -10,
        "paired_latency_change_pct": [-11, -10, -9],
        "latency_change_ci95_pct": [-11, -9],
        "verdict": "faster",
      },
      "megakernel": {
        "base_us": 100,
        "head_us": 105,
        "latency_change_pct": 5,
        "paired_latency_change_pct": [0, 5, 10],
        "latency_change_ci95_pct": [-1, 9],
        "verdict": "inconclusive",
      },
    },
  }

  for arm in data["comparisons"].values():
    arm["numerics"] = {
      "base": {metric: 0.02 for metric in METRICS},
      "head": {metric: 0.01 for metric in METRICS},
      "head_minus_base": {metric: {"mean": -0.01, "ci95": [-0.015, -0.005]} for metric in METRICS},
      "regressions": [],
    }
  return data


def render_report(value):
  return render(value, base_sha=BASE, head_sha=HEAD, run_url=RUN)


def test_numeric_table_and_no_untrusted_prose():
  value = report()
  value["failures"] = ["@everyone <script>malicious</script>"]
  value["snapshots"] = {"gpu": "@everyone"}
  result = render_report(value)
  assert result.startswith(MARKER)
  assert "| CUDA Graphs + PDL | 100.00 | 90.00 | -10.00% | [-11.00%, -9.00%] | Faster |" in result
  assert "| Megakernel | 100.00 | 105.00 | +5.00% | [-1.00%, +9.00%] | Inconclusive |" in result
  assert "@everyone" not in result
  assert "script" not in result
  assert "Manual code review" in result
  assert "CUDA Graphs + PDL pairs: -11.00%, -10.00%, -9.00%." in result
  assert "Megakernel pairs: +0.00%, +5.00%, +10.00%." in result
  assert "Student-t on log(PR/base)" in result
  assert "advisory" in result


@pytest.mark.parametrize(
  "value", [None, [], {}, {"status": "failed"}, {"schema": "unrecognized", "status": "complete"}]
)
def test_missing_or_failed_report(value):
  assert "No performance conclusion" in render_report(value)


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("base_us", 0),
    ("head_us", -1),
    ("head_us", True),
    ("head_us", "@everyone"),
    ("head_us", float("nan")),
    ("base_us", float("inf")),
    ("latency_change_pct", None),
    ("latency_change_ci95_pct", [1, -1]),
    ("latency_change_ci95_pct", [float("-inf"), 0]),
    ("latency_change_ci95_pct", [-1]),
    ("verdict", "@everyone"),
    ("verdict", []),
    ("verdict", "slower"),
  ],
)
def test_rejects_invalid_comparison(field, value):
  data = report()
  data["comparisons"]["cuda_graph_pdl"][field] = value
  result = render_report(data)
  assert "No performance conclusion" in result
  assert "| CUDA Graphs" not in result
  assert "@everyone" not in result


def test_missing_arm_rejects_entire_table():
  data = report()
  del data["comparisons"]["megakernel"]
  assert "No performance conclusion" in render_report(data)


def test_slower_and_touching_zero_interval():
  data = report()
  arm = data["comparisons"]["megakernel"]
  arm.update(latency_change_ci95_pct=[1, 9], verdict="slower")
  assert "| Slower |" in render_report(data)
  arm.update(latency_change_ci95_pct=[0, 9], verdict="inconclusive")
  assert "| Inconclusive |" in render_report(data)


def test_read_failures_and_size_limit(tmp_path):
  path = tmp_path / "report.json"
  assert read_report(path) is None
  path.write_text("not json")
  assert read_report(path) is None
  path.write_bytes(b"\xff")
  assert read_report(path) is None
  path.write_text(" " * (MAX_REPORT_BYTES + 1))
  assert read_report(path) is None
  data = report()
  path.write_text(json.dumps(data))
  assert read_report(path) == data


@pytest.mark.parametrize(
  "kwargs",
  [
    {"base_sha": "<script>"},
    {"head_sha": "b" * 39},
    {"run_url": "https://github.com/a/b/actions/runs/123)\n@everyone"},
    {"run_url": "javascript:alert(1)"},
  ],
)
def test_rejects_markdown_in_trusted_metadata(kwargs):
  arguments = {"base_sha": BASE, "head_sha": HEAD, "run_url": RUN}
  arguments.update(kwargs)
  with pytest.raises(ValueError):
    render(report(), **arguments)


@pytest.mark.parametrize("mutation", ["missing", "unknown_flag", "duplicate", "nonfinite", "inconsistent", "reversed"])
def test_invalid_numerics_prevent_performance_conclusion(mutation):
  data = report()
  arm = data["comparisons"]["cuda_graph_pdl"]
  numeric = arm["numerics"]
  if mutation == "missing":
    del arm["numerics"]
  elif mutation == "unknown_flag":
    numeric["regressions"] = ["@everyone"]
  elif mutation == "duplicate":
    numeric["regressions"] = ["canonical_kl_delta"] * 2
  elif mutation == "nonfinite":
    numeric["base"]["canonical_kl_delta"] = float("nan")
  elif mutation == "inconsistent":
    numeric["regressions"] = ["canonical_kl_delta"]
  else:
    numeric["head_minus_base"]["canonical_kl_delta"]["ci95"] = [1, -1]
  text = render_report(data)
  assert "No performance conclusion" in text
  assert "@everyone" not in text


def test_numerical_regression_is_visible():
  data = report()
  numeric = data["comparisons"]["cuda_graph_pdl"]["numerics"]
  numeric["head"]["canonical_kl_delta"] = 0.03
  numeric["head_minus_base"]["canonical_kl_delta"] = {"mean": 0.01, "ci95": [0.005, 0.015]}
  numeric["regressions"] = ["canonical_kl_delta"]
  text = render_report(data)
  assert "**1/8**" in text
  assert "Canonical KL | +0.02 | +0.03 | +0.01 [+0.005, +0.015]" in text


@pytest.mark.parametrize(
  "pairs",
  [
    None,
    [],
    [0, 1],
    [0] * 11,
    [0, True, 1],
    [0, float("nan"), 1],
    [0, float("inf"), 1],
    [0, "@everyone", 1],
    [-100, 0, 1],
    [-101, 0, 1],
  ],
)
def test_invalid_pair_effects_fail_closed(pairs):
  data = report()
  data["comparisons"]["cuda_graph_pdl"]["paired_latency_change_pct"] = pairs
  text = render_report(data)
  assert "No performance conclusion" in text
  assert "@everyone" not in text


def test_inconsistent_pair_counts_fail_closed():
  data = report()
  data["comparisons"]["cuda_graph_pdl"]["paired_latency_change_pct"] = [0, 1, 2, 3]
  assert "No performance conclusion" in render_report(data)
