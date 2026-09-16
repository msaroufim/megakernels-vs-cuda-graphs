"""Load explicit AOT experiment arms without replacing the production loader.

Build records identify the library, its compiler inputs and packing environment.
This adapter verifies artifact bytes and the proposal entry point configuration;
it does not certify the recorded source-to-binary relationship or numerical math.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class PrecompiledArm:
    """A named, hash-checked scheduler library and its existing build record."""

    label: str
    record_path: Path
    record_sha256: str
    library: Path
    binary_sha256: str
    environment: tuple[tuple[str, str], ...]
    module: ModuleType
    cluster_dim: int | None = None

    def identity(self) -> dict:
        """Return serializable artifact provenance for the benchmark receipt."""
        identity = {
            "label": self.label,
            "build_record": str(self.record_path),
            "build_record_sha256": self.record_sha256,
            "library": str(self.library),
            "binary_sha256": self.binary_sha256,
            "environment": dict(self.environment),
        }
        if self.cluster_dim is not None:
            identity["cluster_dim"] = self.cluster_dim
        return identity


@dataclass(frozen=True)
class PrecompiledPlan:
    """A control scheduler followed by candidates sharing one model capture."""

    path: Path
    sha256: str
    arms: tuple[PrecompiledArm, ...]

    def identity(self) -> dict:
        """Describe what was loaded without implying full contract validation."""
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "arms": [arm.identity() for arm in self.arms],
            "validation_scope": "artifact hashes and proposal build configuration",
            "source_to_binary_mapping_verified": False,
            "full_contract_validated": False,
        }


def checked_bytes(path: Path, expected: str) -> bytes:
    """Read a local artifact and reject a malformed or mismatching SHA256."""
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"invalid SHA256 for {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"artifact hash changed: {path}")
    return raw


def import_native(library: Path) -> ModuleType:
    """Import the extension under its compiled Python initialization name."""
    spec = importlib.util.spec_from_file_location(library.stem, library)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load extension: {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "front_kv_stage", None)):
        raise ValueError(f"missing proposal entry point: {library}")
    return module


def load_plan(path: Path) -> PrecompiledPlan:
    """Validate the whole plan before loading any native candidate code.

    A plan references existing build records rather than copying compiler flags
    into another configuration. Relative record paths resolve beside the plan;
    relative library paths resolve beside their build record. Repeated library
    identities share one module, allowing an identical-binary timing control.
    An optional cluster_dim selects launch geometry during capture only. The
    caller must verify the captured attributes; a library can override the env.
    """
    path = path.resolve()
    raw = path.read_bytes()
    plan = json.loads(raw)
    if set(plan) != {"schema", "arms"} or plan["schema"] != "dspark.precompiled_plan.v1":
        raise ValueError("expected a dspark.precompiled_plan.v1 plan with schema and arms")
    entries = plan["arms"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("provide the retained proposal arm first")
    expected_kwargs = {
        "instrumented": False,
        "relaxed_dag": True,
        "greedy_tail": True,
        "full_loop_device_epoch": True,
        "compiled_execution_mask": 2047,
        "compiled_draft_layer_mask": 7,
    }
    labels, names, pending = set(), {}, []
    for entry in entries:
        required = {"label", "build_record", "sha256"}
        if not required <= set(entry) or set(entry) - required - {"cluster_dim"}:
            raise ValueError("each arm needs label, build_record, sha256 and optional cluster_dim")
        cluster_dim = entry.get("cluster_dim")
        if "cluster_dim" in entry and (
            type(cluster_dim) is not int or cluster_dim not in (1, 2, 4, 8)
        ):
            raise ValueError("cluster_dim must be an integer in (1, 2, 4, 8)")
        label = entry["label"]
        if (
            not isinstance(label, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", label) is None
            or label in labels
            or label == "sglang_cuda_graph"
        ):
            raise ValueError(f"invalid or repeated arm label: {label}")
        labels.add(label)
        record_path = (path.parent / entry["build_record"]).resolve()
        record = json.loads(checked_bytes(record_path, entry["sha256"]))
        if record["kwargs"] != expected_kwargs:
            raise ValueError(f"{label}: build configuration does not match this proposal driver")
        environment = record["environment"]
        if not isinstance(environment, dict) or not all(
            isinstance(k, str) and k.startswith("DSPARK_") and isinstance(v, str)
            for k, v in environment.items()
        ):
            raise ValueError(f"{label}: expected DSPARK build/packing environment")
        library = (record_path.parent / record["module"]).resolve()
        digest = record["binary_sha256"]
        checked_bytes(library, digest)
        if library.suffix != ".so":
            raise ValueError(f"{label}: expected a native .so library")
        if library.stem in names and names[library.stem] != digest:
            raise ValueError("different candidate binaries must have distinct module names")
        names[library.stem] = digest
        pending.append(
            (label, record_path, entry["sha256"], library, digest, environment, cluster_dim)
        )
    if pending[0][0] != "proposal_megakernel":
        raise ValueError("the first arm must be named proposal_megakernel")
    modules, arms = {}, []
    for label, record_path, record_sha, library, digest, environment, cluster_dim in pending:
        key = (library.stem, digest)
        if key not in modules:
            checked_bytes(library, digest)
            modules[key] = import_native(library)
        arms.append(
            PrecompiledArm(
                label,
                record_path,
                record_sha,
                library,
                digest,
                tuple(sorted(environment.items())),
                modules[key],
                cluster_dim,
            )
        )
    return PrecompiledPlan(path, hashlib.sha256(raw).hexdigest(), tuple(arms))


def host_environment(arm: PrecompiledArm) -> dict[str, str]:
    """Select the existing host layout flags needed by a precompiled arm.

    Source directories, compiler-only options and old report destinations in a
    build record are provenance, not runtime settings. Only the existing host
    packing/preparation flags are needed when the scheduler is already built.
    """
    host_keys = {
        "DSPARK_LM_BULK_STAGING",
        "DSPARK_WOA_BULK_STAGING",
        "DSPARK_QB_BLOCK_SCALED",
        "DSPARK_DENSE_BLOCK_SCALED",
        "DSPARK_MARKOV_BULK_K",
        "DSPARK_ROUTED_PACKED_SFA",
        "DSPARK_ROUTED_BULK_WEIGHTS",
        "DSPARK_ROUTED_INTERLEAVE",
        "DSPARK_SHARED_BULK_STAGING",
        "DSPARK_SPECIALIZE_ALL_PHASES",
        "DSPARK_PREPARE_VECTOR_COPY",
        "DSPARK_V4_IDLE_SLEEP_NS",
        "DSPARK_V4_IDLE_SLEEP_LONG_NS",
        "DSPARK_V4_IDLE_SLEEP_AFTER",
        "DSPARK_V4_SCHED_FLAGS",
        "DSPARK_V4_DEEP_PERIOD",
        "DSPARK_V4_EXPRESS_WORKERS",
        "DSPARK_V4_PREFETCH_PERIOD",
    }
    return {key: value for key, value in arm.environment if key in host_keys}


@contextmanager
def packing_environment(arm: PrecompiledArm) -> Iterator[None]:
    """Scope layout and explicit launch geometry to capture, restoring on failure.

    Capture each arm inside its selected configuration. Replay the resulting
    graphs after this context exits; never set geometry per sample.
    Existing exact-function residency checks remain responsible for safe launch.
    """
    previous = {k: v for k, v in os.environ.items() if k.startswith("DSPARK_")}
    try:
        for key in previous:
            os.environ.pop(key)
        os.environ.update(host_environment(arm))
        if arm.cluster_dim is not None:
            os.environ["DSPARK_V4_CLUSTER_DIM"] = str(arm.cluster_dim)
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith("DSPARK_"):
                os.environ.pop(key)
        os.environ.update(previous)
