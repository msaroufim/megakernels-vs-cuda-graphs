"""Build single-launch proposal IO around the exact retained full V4 math image."""

import argparse
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


def build(artifact_root, output, overlay):
    """Reuse the retained compiler inputs with one explicit feature flag and a new build name."""
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
    proof = integrate(source, overlay)
    inventory = {name: sha(source / name) for name in (*base["source_sha256"], HEADER)}
    emit(output / "source-manifest.json", inventory)
    emit(output / "integration-proof.json", proof)
    args = dict(base["observed_load_inputs"])
    args["name"] = "dspark_full_proposal_io_0913"
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
    args = parser.parse_args()
    build(args.artifact_root, args.output, args.overlay)


if __name__ == "__main__":
    main()
