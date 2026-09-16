"""Artifact, capture and graph utilities for the retained full-proposal benchmark."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from graph_metadata import identity


def load(name, path):
    """Import a named retained helper from its explicit artifact path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def emit(path, value):
    """Atomically retain progress or a completed measurement."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha(path):
    """Hash files without loading large banks into host memory at once."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def capture_factory(factory, snapshot, port, proposal_output=None):
    """Retain the existing recapture closure without changing the original factory."""
    capture = []
    if sys.getprofile() is not None:
        raise RuntimeError("Unexpected active Python profiler")

    def observe(frame, event, arg):
        """Observe the factory's first capture entry, then disable observation."""
        if (
            event == "call"
            and frame.f_code.co_name == "capture_current_tail"
            and frame.f_code.co_filename == factory.__code__.co_filename
        ):
            capture.append(frame.f_back.f_locals["capture_current_tail"])
            sys.setprofile(None)

    sys.setprofile(observe)
    try:
        graft = factory(
            snapshot=snapshot,
            dist_port=port,
            fused_markov_argmax=False,
            capture_markov_control=False,
            proposal_output=proposal_output,
        )
    finally:
        sys.setprofile(None)
    if len(capture) != 1:
        raise RuntimeError("Original recapture closure not found")
    return graft, capture[0]


def tensor_fields(owner, torch):
    """Enumerate direct runner buffer tensors used by the captured proposal."""
    return {name: value for name, value in vars(owner).items() if isinstance(value, torch.Tensor)}


def graph_record(graph, nc, tg):
    """Record actual native nodes and all dependency edge types, including PDL."""
    handle = nc.cu.CUgraph(graph.raw_cuda_graph())
    nodes = nc.graph_nodes(handle)
    edges = tg.edges(nodes, nc.graph_edges(handle))
    return {
        "nodes": [identity(node, nc, tg.identity) for node in nodes],
        "edges": edges,
        "node_count": len(nodes),
        "pdl_edges": sum(e["type"] == 1 for e in edges),
    }


def save_tensors(torch, path, values):
    """Save comparison tensors and their byte identities outside measurement."""
    path.mkdir(parents=True, exist_ok=False)
    result = {}
    for name, value in values.items():
        data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        file = path / (name + ".bin")
        file.write_bytes(data)
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    emit(path / "tensors.json", result)
    return result
