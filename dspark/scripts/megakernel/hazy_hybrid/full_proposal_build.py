"""Build single-launch proposal IO around the exact retained full V4 math image."""

import argparse
import difflib
import json
import os
import shutil
import subprocess
from pathlib import Path

from benchmark_utils import emit, sha

BASE_RECORD_SHA = "b602925cbc290184e88b14e9954d6627ec02063ed463183817f1a756f9bdd3ba"
BASE_BINARY_SHA = "0aa23af95acc88bb3bbaf096174c9af9fb2e7062b16bd8b6f5447d3dba128c55"
FLAG = "-DDSPARK_V4_INTEGRATED_PROPOSAL_IO=1"
KERNEL = "deepspec/megakernel/csrc/dspark_v4_kernel.cu"
HEADER = "deepspec/megakernel/csrc/dspark_proposal_io.cuh"


def guarded_blocks(text):
    """Extract the four new compile-guarded integration blocks, including nested guards."""
    lines = text.splitlines(keepends=True)
    blocks, active, depth = [], [], 0
    for line in lines:
        if line == "#ifdef DSPARK_V4_INTEGRATED_PROPOSAL_IO\n" and not active:
            active, depth = [line], 1
            continue
        if active:
            active.append(line)
            if line.startswith(("#if ", "#ifdef ", "#ifndef ")):
                depth += 1
            elif line.startswith("#endif"):
                depth -= 1
            if depth == 0:
                blocks.append("".join(active))
                active = []
    if active or len(blocks) != 4:
        raise ValueError("Expected exactly four closed integrated-IO guards")
    return blocks


def integrate(source, overlay):
    """Insert only the reviewed IO blocks; preserve all retained phase and scheduler bodies."""
    path = source / KERNEL
    original = path.read_text()
    blocks = guarded_blocks((overlay / "dspark_v4_kernel.cu").read_text())
    anchors = (
        "// Outlining isolates phase register lifetimes from the persistent scheduler.",
        "#ifdef DSPARK_V4_TMA_WEIGHTS\n"
        "  if (w13_weight_tma != nullptr && w2_weight_tma != nullptr) {",
        "#if DSPARK_ROUTED_REUSE_TMEM\n  dspark_tmem::release_routed_reservation();",
        "  const int device = workspace.get_device();",
    )
    updated = original
    for index, (block, anchor) in enumerate(zip(blocks, anchors, strict=True)):
        if index == 3:
            begin = updated.index("void launch_v4_scheduler(")
            before, section = updated[:begin], updated[begin:]
        else:
            before, section = "", updated
        if section.count(anchor) != 1:
            raise ValueError(f"Retained IO insertion anchor changed: {anchor}")
        updated = before + section.replace(anchor, block + anchor)
    reverted = updated
    for block in blocks:
        if reverted.count(block) != 1:
            raise ValueError("An IO block is ambiguous")
        reverted = reverted.replace(block, "")
    if reverted != original:
        raise ValueError("IO integration altered retained math or scheduling")
    path.write_text(updated)
    shutil.copyfile(overlay / "dspark_proposal_io.cuh", source / HEADER)
    return {"retained_kernel_sha256": sha_bytes(original.encode()), "inserted_blocks": blocks}


def sha_bytes(raw):
    """Hash in-memory source for the exact inverse record."""
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def scheduler_overlay(retained, overlay, *, include_tmem=False, include_scheduler=True):
    """Apply only reviewed scheduler edits; keep all retained phase bodies exact."""
    original = retained.read_text()
    candidate = (overlay / Path(KERNEL).name).read_text()
    if not include_scheduler:
        candidate = original
    if include_tmem:
        tmem_path = retained.parent / "dspark_tmem.cuh"
        tmem_original = tmem_path.read_text()
        tmem_candidate = (overlay / "dspark_tmem.cuh").read_text()
        if tmem_original == tmem_candidate:
            raise ValueError("TMEM overlay requested but no TMEM changes found")
        tmem_path.write_text(tmem_candidate)
    else:
        tmem_original = tmem_candidate = None
    diff = list(difflib.unified_diff(original.splitlines(), candidate.splitlines(), n=0))
    changed = [
        line[1:]
        for line in diff
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    forbidden = [
        line
        for line in changed
        if line.strip()
        not in (
            "",
            "}",
            "do {",
            "} while (true);",
            "break;",
            "return false;",
            "const uint64_t* address, uint64_t target) {",
        )
        and not any(
            token in line
            for token in (
                "load_acquire_u64",
                "load_relaxed_u64",
                "wait_counter_",
                "observed",
                "nanosleep",
                "published",
                "publish_epoch",
                "ready",
                "coherence",
                "acquire",
                "counter",
                "dependent data",
            )
        )
    ]
    if forbidden:
        raise ValueError(f"Scheduler overlay changes non-polling code: {forbidden[:4]}")
    if original == candidate and not include_tmem:
        raise ValueError("Scheduler overlay requested but no scheduler changes found")
    if include_scheduler and not changed:
        raise ValueError("Scheduler overlay produced no reviewable diff")
    retained.write_text(candidate)
    shutil.copyfile(overlay / "dspark_proposal_io.cuh", retained.parent / Path(HEADER).name)
    return {
        "mode": "reviewed scheduler overlay",
        "retained_kernel_sha256": sha_bytes(original.encode()),
        "candidate_kernel_sha256": sha_bytes(candidate.encode()),
        "tmem_original_sha256": sha_bytes(tmem_original.encode()) if tmem_original else None,
        "tmem_candidate_sha256": sha_bytes(tmem_candidate.encode()) if tmem_candidate else None,
        "changed_lines": len(changed),
        "scheduler_only": include_scheduler,
        "tmem_only": include_tmem and not include_scheduler,
    }


def build(
    artifact_root, output, overlay, scheduler_overlay_enabled=False, tmem_overlay_enabled=False
):
    """Rebuild retained inputs with IO and, optionally, reviewed scheduler edits."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Build in a subprocess with CUDA devices hidden")
    import torch
    from torch.utils.cpp_extension import load

    if torch.cuda.is_initialized():
        raise ValueError("The build process initialized CUDA")
    base_path = artifact_root.parent / "extensions-resumed-completion-red-0908/build-record.json"
    if sha(base_path) != BASE_RECORD_SHA:
        raise ValueError("Retained full-proposal build record changed")
    base = json.loads(base_path.read_text())
    if base["binary_sha256"] != BASE_BINARY_SHA or sha(base["module"]) != BASE_BINARY_SHA:
        raise ValueError("Retained full-proposal native image changed")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    for name, digest in base["source_sha256"].items():
        donor, target = Path(base["source"]) / name, source / name
        if sha(donor) != digest:
            raise ValueError(f"Retained source changed: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(donor, target)
    io_proof = integrate(source, overlay)
    if scheduler_overlay_enabled or tmem_overlay_enabled:
        proof = scheduler_overlay(
            source / KERNEL,
            overlay,
            include_tmem=tmem_overlay_enabled,
            include_scheduler=scheduler_overlay_enabled,
        )
        proof["io_integration"] = io_proof
    else:
        proof = io_proof
    inventory = {name: sha(source / name) for name in (*base["source_sha256"], HEADER)}
    emit(output / "source-manifest.json", inventory)
    emit(output / "integration-proof.json", proof)
    args = dict(base["observed_load_inputs"])
    args["name"] = "dspark_full_proposal_io_0913"
    if scheduler_overlay_enabled:
        args["name"] += "_scheduler_poll"
    if tmem_overlay_enabled:
        args["name"] += "_tmem_permit"
    for key in ("sources", "extra_include_paths"):
        if key in args:
            args[key] = [str(value).replace(base["source"], str(source)) for value in args[key]]
    args["extra_cuda_cflags"] = [*args["extra_cuda_cflags"], FLAG]
    args["build_directory"] = str(output / "native")
    Path(args["build_directory"]).mkdir()
    emit(output / "build-inputs.json", args)
    module = load(**args)
    if torch.cuda.is_initialized():
        raise ValueError("The compiler initialized CUDA")
    record = {
        "source": str(source),
        "source_sha256": inventory,
        "module": module.__file__,
        "binary_sha256": sha(module.__file__),
        "kwargs": base["kwargs"],
        "environment": base["environment"],
        "observed_load_inputs": args,
        "private_algorithm_flags": [*base["private_algorithm_flags"], FLAG[2:]],
        "base_binary_sha256": BASE_BINARY_SHA,
        "base_record_sha256": BASE_RECORD_SHA,
        "torch_version": torch.__version__,
        "cuda_initialized": False,
        "scope": "One full proposal launch; numerical/runtime qualification pending",
        "scheduler_overlay": scheduler_overlay_enabled,
        "tmem_overlay": tmem_overlay_enabled,
    }
    emit(output / "build-record.json", record)
    for option, name in (("--dump-resource-usage", "resources.txt"), ("--dump-sass", "sass.txt")):
        with (output / name).open("w") as log:
            subprocess.run(
                ["/usr/local/cuda/bin/cuobjdump", option, module.__file__],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
    print("BUILD_COMPLETE", record["binary_sha256"], flush=True)


def main():
    """Run one explicit bounded build; this script never executes model kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument(
        "--scheduler-overlay",
        action="store_true",
        help="Apply reviewed scheduler-only changes instead of requiring the historical kernel",
    )
    parser.add_argument(
        "--tmem-overlay",
        action="store_true",
        help="Apply the independently reviewed TMEM lifetime change from the overlay",
    )
    args = parser.parse_args()
    build(
        args.artifact_root,
        args.output,
        args.overlay,
        args.scheduler_overlay,
        args.tmem_overlay,
    )


if __name__ == "__main__":
    main()
