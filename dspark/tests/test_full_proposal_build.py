"""Preserve retained math when explicitly applying a scheduler-only overlay."""

import importlib.util
import sys
from pathlib import Path

import pytest

H = Path(__file__).parents[1] / "scripts/megakernel/hazy_hybrid"
spec = importlib.util.spec_from_file_location("full_proposal_build", H / "full_proposal_build.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(H))
spec.loader.exec_module(module)


def test_scheduler_overlay_is_explicit_and_bounded(tmp_path):
    retained = tmp_path / "retained.cu"
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    original = "int phase_math() { return 7; }\nwhile (load_acquire_u64(counter) != ready) {\n}\n"
    retained.write_text(original)
    (overlay / Path(module.KERNEL).name).write_text(original)
    (overlay / "dspark_proposal_io.cuh").write_text("// retained IO\n")
    with pytest.raises(ValueError, match="no scheduler changes"):
        module.scheduler_overlay(retained, overlay)

    candidate = original.replace("load_acquire_u64", "wait_counter_exact")
    (overlay / Path(module.KERNEL).name).write_text(candidate)
    proof = module.scheduler_overlay(retained, overlay)
    assert proof["scheduler_only"] is True
    assert retained.read_text() == candidate


def test_scheduler_overlay_rejects_phase_math_change(tmp_path):
    retained = tmp_path / "retained.cu"
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    retained.write_text("int phase_math() { return 7; }\n")
    (overlay / Path(module.KERNEL).name).write_text("int phase_math() { return 8; }\n")
    (overlay / "dspark_proposal_io.cuh").write_text("// retained IO\n")
    with pytest.raises(ValueError, match="non-polling"):
        module.scheduler_overlay(retained, overlay)
    assert retained.read_text() == "int phase_math() { return 7; }\n"


def test_tmem_overlay_does_not_require_scheduler_change(tmp_path):
    retained = tmp_path / "retained.cu"
    retained.write_text("int phase_math() { return 7; }\n")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / Path(module.KERNEL).name).write_text(retained.read_text())
    (overlay / "dspark_tmem.cuh").write_text("relinquish_alloc_permit\n")
    (overlay / "dspark_proposal_io.cuh").write_text("// retained IO\n")
    (tmp_path / "dspark_tmem.cuh").write_text("old allocation\n")
    proof = module.scheduler_overlay(retained, overlay, include_tmem=True, include_scheduler=False)
    assert proof["tmem_only"] is True
    assert (tmp_path / "dspark_tmem.cuh").read_text() == "relinquish_alloc_permit\n"
    assert retained.read_text() == "int phase_math() { return 7; }\n"
