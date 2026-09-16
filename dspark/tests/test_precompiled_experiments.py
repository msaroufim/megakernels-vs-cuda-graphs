"""Protect artifact selection and scoped packing state for AOT experiments."""

import hashlib
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from scripts.megakernel import precompiled_experiments as aot


def write_plan(tmp_path: Path, *, record_changes=None, labels=None) -> Path:
    """Create one fake native artifact and a correctly hashed proposal record."""
    library = tmp_path / "retained.so"
    library.write_bytes(b"test native library")
    record = {
        "module": library.name,
        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "environment": {"DSPARK_SPECIALIZE_ALL_PHASES": "1"},
        "kwargs": {
            "instrumented": False,
            "relaxed_dag": True,
            "greedy_tail": True,
            "full_loop_device_epoch": True,
            "compiled_execution_mask": 2047,
            "compiled_draft_layer_mask": 7,
        },
    }
    record.update(record_changes or {})
    build = tmp_path / "build.json"
    build.write_text(json.dumps(record))
    digest = hashlib.sha256(build.read_bytes()).hexdigest()
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema": "dspark.precompiled_plan.v1",
                "arms": [
                    {"label": label, "build_record": build.name, "sha256": digest}
                    for label in labels or ["proposal_megakernel", "identical_control"]
                ],
            }
        )
    )
    return path


def test_identical_binary_shares_native_module(tmp_path, monkeypatch):
    """Load aliases once while preserving their separate benchmark labels."""
    loaded = []
    module = ModuleType("retained")

    def load(library):
        """Record native imports without requiring CUDA in this CPU test."""
        loaded.append(library)
        return module

    monkeypatch.setattr(aot, "import_native", load)
    plan = aot.load_plan(write_plan(tmp_path))
    assert len(loaded) == 1
    assert plan.arms[0].module is plan.arms[1].module
    assert plan.identity()["source_to_binary_mapping_verified"] is False
    assert plan.identity()["full_contract_validated"] is False
    assert all("cluster_dim" not in arm.identity() for arm in plan.arms)


def test_same_binary_captures_distinct_geometry_without_changing_layout(tmp_path, monkeypatch):
    """Reuse one library while restoring capture settings on success and failure."""
    path = write_plan(tmp_path)
    raw = json.loads(path.read_text())
    for entry, dim in zip(raw["arms"], (1, 2)):
        entry["cluster_dim"] = dim
    path.write_text(json.dumps(raw))
    loaded = []

    def load(library):
        """Record imports without requiring a CUDA runtime."""
        loaded.append(library)
        return ModuleType("retained")

    monkeypatch.setattr(aot, "import_native", load)
    monkeypatch.setenv("DSPARK_V4_CLUSTER_DIM", "8")
    before = dict(os.environ)
    plan = aot.load_plan(path)
    assert len(loaded) == 1 and plan.arms[0].module is plan.arms[1].module
    assert aot.host_environment(plan.arms[0]) == aot.host_environment(plan.arms[1])
    for arm, dim in zip(plan.arms, (1, 2)):
        assert arm.identity()["cluster_dim"] == dim
        with aot.packing_environment(arm):
            assert os.environ["DSPARK_V4_CLUSTER_DIM"] == str(dim)
        assert dict(os.environ) == before
        with pytest.raises(RuntimeError, match="capture failed"):
            with aot.packing_environment(arm):
                assert os.environ["DSPARK_V4_CLUSTER_DIM"] == str(dim)
                raise RuntimeError("capture failed")
        assert dict(os.environ) == before


@pytest.mark.parametrize("value", [None, True, False, "2", 2.0, 0, -2, 3, 16])
def test_invalid_geometry_fails_before_import(tmp_path, monkeypatch, value):
    """Do not silently accept a malformed or unsupported capture geometry."""
    path = write_plan(tmp_path)
    raw = json.loads(path.read_text())
    raw["arms"][0]["cluster_dim"] = value
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(aot, "import_native", lambda _: pytest.fail("native import reached"))
    with pytest.raises(ValueError, match="cluster_dim"):
        aot.load_plan(path)


@pytest.mark.parametrize("changed", ["build.json", "retained.so"])
def test_changed_artifact_fails_before_native_import(tmp_path, monkeypatch, changed):
    """A stale build record or library must fail before code can be imported."""
    path = write_plan(tmp_path)
    artifact = tmp_path / changed
    artifact.write_bytes(artifact.read_bytes() + b" ")
    monkeypatch.setattr(aot, "import_native", lambda _: pytest.fail("native import reached"))
    with pytest.raises(ValueError, match="artifact hash changed"):
        aot.load_plan(path)


@pytest.mark.parametrize(
    "changes,labels,match",
    [
        ({"kwargs": {"instrumented": True}}, None, "build configuration"),
        ({"environment": {"CUDA_LAUNCH_BLOCKING": "1"}}, None, "build/packing environment"),
        ({}, ["proposal_megakernel", "proposal_megakernel"], "repeated arm"),
        ({}, ["sglang_cuda_graph"], "invalid or repeated"),
        ({}, ["candidate"], "first arm"),
    ],
)
def test_invalid_plan_fails_before_native_import(tmp_path, monkeypatch, changes, labels, match):
    """Reject unsupported modes, labels and non-packing process settings."""
    path = write_plan(tmp_path, record_changes=changes, labels=labels)
    monkeypatch.setattr(aot, "import_native", lambda _: pytest.fail("native import reached"))
    with pytest.raises(ValueError, match=match):
        aot.load_plan(path)


def test_packing_environment_restored_after_capture_failure(tmp_path, monkeypatch):
    """Prevent one failed arm's flags from contaminating later captures."""
    monkeypatch.setattr(aot, "import_native", lambda _: ModuleType("retained"))
    plan = aot.load_plan(write_plan(tmp_path))
    monkeypatch.setenv("DSPARK_UNRELATED_PRIOR", "preserve")
    monkeypatch.setenv("UNRELATED_PROCESS_SETTING", "preserve")
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="capture failed"):
        with aot.packing_environment(plan.arms[0]):
            assert "DSPARK_UNRELATED_PRIOR" not in os.environ
            assert os.environ["DSPARK_SPECIALIZE_ALL_PHASES"] == "1"
            assert os.environ["UNRELATED_PROCESS_SETTING"] == "preserve"
            os.environ["DSPARK_ADDED_DURING_CAPTURE"] = "discard"
            raise RuntimeError("capture failed")
    assert dict(os.environ) == before
