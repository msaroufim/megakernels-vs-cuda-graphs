"""CPU checks for untrusted source selection and paired regression reporting."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ci import benchmark as ci


def snapshot(root, upstream="a" * 40, kittens="b" * 40):
  path = root / "llama/upstream/prepare.py"
  path.parent.mkdir(parents=True)
  path.write_text(f"UPSTREAM = {upstream!r}\nTHUNDERKITTENS = {kittens!r}\nraise RuntimeError('do not execute')\n")
  return root


def test_pin_reader_never_executes_snapshot(tmp_path):
  assert ci.pins(snapshot(tmp_path)) == {"UPSTREAM": "a" * 40, "THUNDERKITTENS": "b" * 40}
  path = tmp_path / "llama/upstream/prepare.py"
  path.write_text("UPSTREAM = dangerous()\nTHUNDERKITTENS = '" + "b" * 40 + "'\n")
  with pytest.raises(ValueError, match="literal"):
    ci.pins(tmp_path)
  snapshot_text = "UPSTREAM = 'main'\nTHUNDERKITTENS = '" + "b" * 40 + "'\n"
  path.write_text(snapshot_text)
  with pytest.raises(ValueError, match="full lowercase"):
    ci.pins(tmp_path)


def test_source_symlinks_rejected(tmp_path):
  (tmp_path / "escape.py").symlink_to(Path(__file__))
  with pytest.raises(ValueError, match="symlinks"):
    ci.source_files(tmp_path)
  with pytest.raises(ValueError, match="linked"):
    ci.contained(tmp_path, "escape.py")


def runs(scale=1.0, error_delta=0.0):
  return [
    {
      "side": side,
      "pair": pair,
      "mean_us_per_step": {arm: (100 + pair) * (scale if side == "head" else 1) for arm in ci.ARMS.values()},
      "numerics": {
        arm: {"actual_kl_delta": 0.008 + (error_delta if side == "head" else 0)} for arm in ci.ARMS.values()
      },
    }
    for pair in range(3)
    for side in ("base", "head")
  ]


def test_existing_numerical_miss_is_not_regression():
  result = ci.summarize(runs())
  for row in result.values():
    assert row["verdict"] == "inconclusive"
    assert row["latency_change_ci95_pct"] == [0, 0]
    assert row["numerics"]["regressions"] == []
    assert row["numerics"]["head"]["actual_kl_delta"] > 0


@pytest.mark.parametrize(("scale", "verdict"), [(0.9, "faster"), (1.1, "slower")])
def test_paired_process_change_direction(scale, verdict):
  row = ci.summarize(runs(scale))["megakernel"]
  assert row["verdict"] == verdict
  assert row["latency_change_pct"] == pytest.approx((scale - 1) * 100)
  assert row["latency_change_ci95_pct"] == pytest.approx([(scale - 1) * 100] * 2)


def test_numerical_worsening_is_flagged_separately():
  row = ci.summarize(runs(0.9, 0.001))["cuda_graph_pdl"]
  assert row["verdict"] == "faster"
  assert row["numerics"]["regressions"] == ["actual_kl_delta"]


def test_missing_duplicate_and_nonfinite_processes_fail():
  with pytest.raises(ValueError, match="Missing"):
    ci.summarize(runs()[:-1])
  with pytest.raises(ValueError, match="Duplicate"):
    ci.summarize(runs() + runs()[:2])
  invalid = runs()
  invalid[0]["mean_us_per_step"]["candidate"] = float("nan")
  with pytest.raises(ValueError, match="finite"):
    ci.summarize(invalid)


def numerical_inputs():
  row = {"finite": True, "kl": 0.01, "abs_nll_error": 0.03}
  matched = {
    "status": "complete",
    "bf16_calibration": [row] * 127,
    "teacher_forced_graph": {arm: [[row] * 127] * 3 for arm in ci.ARMS.values()},
  }
  checked = {
    "status": "complete",
    "all_actual_final_logits_finite": True,
    "all_reference_logits_finite": True,
    "all_token_boundaries_valid": True,
    "unique_executions": {
      "x": {
        "observations": [
          {"arm": arm, "stage": f"timed-{i}", "selected_final_token": 7} for arm in ci.ARMS.values() for i in range(10)
        ],
        "metrics_by_selected_token": {
          "7": {
            "paired_error_differences": {
              "kl_error_minus_native_bf16": 0.004,
              "absolute_nll_error_minus_native_bf16": -0.001,
            }
          }
        },
      }
    },
  }
  return matched, checked


def test_numeric_summary_counts_observations_not_unique_histories():
  matched, checked = numerical_inputs()
  row = ci.numerical_summary(matched, checked, "candidate")
  assert row["canonical_kl_delta"] == 0
  assert row["actual_kl_delta"] == 0.004
  checked["unique_executions"]["x"]["observations"].pop(0)
  with pytest.raises(ValueError, match="ten finite"):
    ci.numerical_summary(matched, checked, "candidate")


def test_invalid_execution_cannot_have_valid_latency():
  matched, checked = numerical_inputs()
  checked["all_token_boundaries_valid"] = False
  with pytest.raises(ValueError, match="boundaries"):
    ci.numerical_summary(matched, checked, "candidate")


def git(root, *args):
  return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.mark.parametrize("vendored", [False, True])
def test_offline_build_uses_trusted_harness_and_proposed_kernels(tmp_path, vendored):
  cache = tmp_path / "cache"
  cache.mkdir()
  revisions = []
  for name in ("Megakernels", "ThunderKittens"):
    repo = cache / f"{name}.git"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "source.txt").write_text("original\n")
    if name == "Megakernels":
      demo = repo / "demos/low-latency-llama"
      demo.mkdir(parents=True)
      (demo / "Makefile").write_text("all:\n\tprintf original > mk_llama.so\n")
      (demo / "llama.cu").write_text("// original kernel\n")
      (demo / "deleted.cuh").write_text("// removed in vendor\n")
      (repo / "include").mkdir()
      (repo / "include/deleted.cuh").write_text("// removed shared header\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "source")
    revisions.append(git(repo, "rev-parse", "HEAD"))
  proposed = snapshot(tmp_path / "snapshot", *revisions)
  kernels = proposed / "llama/kernels"
  kernels.mkdir()
  (kernels / "candidate.py").write_text("native_bf16_fma: bool = True\n")
  (kernels / "new.cuh").write_text("// new source\n")
  if vendored:
    vendor = proposed / "llama/megakernel"
    vendor.mkdir()
    (vendor / "Makefile").write_text("all:\n\tcat llama.cu ../../include/new.cuh > mk_llama.so\n")
    (vendor / "llama.cu").write_text("// proposed kernel\n")
    (vendor / "new.cuh").write_text("// new local header\n")
    (vendor / "include").mkdir()
    (vendor / "include/new.cuh").write_text("// new shared header\n")
  output = tmp_path / "out"
  output.mkdir()
  runner = ci.Runner(output, tmp_path / "model", cache)
  runtime, upstream, _ = runner.build("head", proposed, {})
  assert (upstream / "source.txt").read_text() == "original\n"
  demo = upstream / "demos/low-latency-llama"
  vendor_hashes = runner.report["snapshots"]["head"]["vendored_sha256"]
  if vendored:
    assert (demo / "llama.cu").read_text() == "// proposed kernel\n"
    assert (demo / "new.cuh").is_file()
    assert not (demo / "deleted.cuh").exists()
    assert not (upstream / "include/deleted.cuh").exists()
    assert (demo / "mk_llama.so").read_text() == "// proposed kernel\n// new shared header\n"
    assert "demos/low-latency-llama/llama.cu" in vendor_hashes
    assert "include/new.cuh" in vendor_hashes
  else:
    assert (demo / "llama.cu").read_text() == "// original kernel\n"
    assert (demo / "deleted.cuh").is_file()
    assert (upstream / "include/deleted.cuh").is_file()
    assert (demo / "mk_llama.so").read_text() == "original"
    assert vendor_hashes == {}
  assert (runtime / "phase3/fp32_headout_fastmath/candidate.py").read_text() == "native_bf16_fma: bool = False\n"
  assert (runtime / "phase3/fp32_headout_fastmath/new.cuh").read_text() == "// new source\n"
  harness = runtime / "phase3/harness/matched_final.py"
  assert "def metrics(actual,reference,target):" in harness.read_text()
  manifest = json.loads((runtime / "stage-manifest.json").read_text())
  assert all(ci.sha256(runtime / name) == digest for name, digest in manifest["staged_sha256"].items())


def test_help_does_not_import_gpu_packages():
  result = subprocess.run([sys.executable, str(ci.STUDY / "ci/benchmark.py"), "--help"], capture_output=True, text=True)
  assert result.returncode == 0
  assert "--upstream-cache" in result.stdout


def test_subprocess_failure_retains_duration_and_prints_log_tail(tmp_path, capsys):
  runner = ci.Runner(tmp_path, tmp_path / "model", tmp_path / "cache")
  with pytest.raises(RuntimeError, match="exit 7"):
    runner.run("failure", [sys.executable, "-c", "print('useful compiler diagnostic'); raise SystemExit(7)"])
  command = runner.report["commands"][0]
  assert command["duration_seconds"] >= 0
  assert command["returncode"] == 7
  assert "useful compiler diagnostic" in capsys.readouterr().out
  assert json.loads((tmp_path / "report.json").read_text())["commands"][0]["log_sha256"]


def test_subprocess_timeout_retains_failure_record(tmp_path, monkeypatch):
  def timeout(command, **kwargs):
    kwargs["stdout"].write("last progress before timeout\n")
    raise subprocess.TimeoutExpired(command, 1)

  monkeypatch.setattr(ci.subprocess, "run", timeout)
  runner = ci.Runner(tmp_path, tmp_path / "model", tmp_path / "cache")
  with pytest.raises(RuntimeError, match="TimeoutExpired"):
    runner.run("timeout", ["never-completes"], timeout=1)
  assert runner.report["commands"][0]["duration_seconds"] >= 0
  assert "TimeoutExpired" in runner.report["commands"][0]["error"]


def test_student_t_known_three_pair_interval():
  mean, bounds = ci.interval([1.0, 2.0, 3.0])
  assert mean == 2.0
  assert bounds == pytest.approx([-0.484137711750, 4.484137711750], abs=1e-10)
  # Unlike a three-observation percentile bootstrap, this interval extends
  # beyond observed effects, reflecting uncertainty from only two degrees of freedom.
  assert bounds[0] < 1 and bounds[1] > 3


def test_student_t_log_ratio_and_ten_pair_critical_value():
  import math
  import statistics

  effects = [0.01 * i for i in range(10)]
  mean, bounds = ci.interval(effects, ratio=True)
  radius = 2.2621571628540993 * statistics.stdev(effects) / math.sqrt(10)
  assert mean == pytest.approx(100 * math.expm1(0.045))
  assert bounds == pytest.approx([100 * math.expm1(0.045 - radius), 100 * math.expm1(0.045 + radius)])


def test_report_exposes_pair_effects_and_interval_assumptions():
  row = ci.summarize(runs(1.1))["megakernel"]
  assert row["paired_latency_change_pct"] == pytest.approx([10.0, 10.0, 10.0])
  assert row["interval"]["method"] == "paired-student-t"
  assert row["interval"]["degrees_of_freedom"] == 2
  assert row["interval"]["latency_scale"] == "log(head/base)"
  assert "normal" in row["interval"]["assumptions"]


@pytest.mark.parametrize("values", [[0.0] * 2, [0.0] * 11, [0.0, 1.0, float("nan")]])
def test_student_t_rejects_unsupported_or_nonfinite_pairs(values):
  with pytest.raises(ValueError, match="Three to ten finite"):
    ci.interval(values)


def test_student_t_rejects_overflowing_transformed_interval():
  with pytest.raises(ValueError, match="Nonfinite interval"):
    ci.interval([-1000, 0, 1000], ratio=True)
