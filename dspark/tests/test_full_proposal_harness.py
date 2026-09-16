"""Protect retained replay ordering, argument storage and artifact serialization on CPU."""

import ast
import ctypes as ct
import importlib.util
import json
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

H = Path(__file__).parents[1] / "scripts/megakernel/hazy_hybrid"


@pytest.fixture
def harness(monkeypatch):
    """Import the standalone entry point without its retired experiment dependencies."""
    monkeypatch.syspath_prepend(str(H))
    spec = importlib.util.spec_from_file_location(
        "retained_full_proposal_lab", H / "full_proposal_lab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nodes():
    """Read the real nested functions and measurement loop without executing CUDA setup."""
    return ast.parse((H / "full_proposal_lab.py").read_text())


@pytest.mark.parametrize("commit, selected", [(0, 0), (1, 0), (6, 5), (7, 5)])
def test_target_refresh_projects_only_committed_row(harness, commit, selected):
    """Never charge six target rows when native selects one clamped committed row."""
    writes = []
    target = torch.arange(6, dtype=torch.bfloat16)[:, None].expand(6, 12288).contiguous()

    def write(**values):
        writes.append({key: value.clone() for key, value in values.items() if key != "pool"})

    graft = SimpleNamespace(
        bundle=SimpleNamespace(
            draft_model=SimpleNamespace(write_target_hidden_kv=write),
            draft_model_runner=SimpleNamespace(token_to_kv_pool=object()),
        )
    )
    cpu = SimpleNamespace(
        **{
            name: getattr(torch, name)
            for name in ("empty", "empty_like", "bfloat16", "sub", "index_select", "remainder")
        }
    )
    cpu.cuda = SimpleNamespace(
        synchronize=lambda: None, CUDAGraph=lambda **_: object(), graph=lambda _: nullcontext()
    )
    harness.capture_target_work(
        cpu, graft, target, torch.tensor([commit], dtype=torch.int32), torch.tensor([131])
    )
    assert len(writes) == 4
    for values in writes:
        assert values["main_hidden"].shape == (1, 12288)
        assert torch.equal(values["main_hidden"], target[selected : selected + 1])
        assert values["positions"].tolist() == [130]
        assert values["swa_loc"].tolist() == [2]


def test_packed_and_array_kernel_arguments(harness):
    """Preserve coverage formerly in the mixed historical Hazy test file."""
    from graph_metadata import parameter_bytes

    assert parameter_bytes(0, 0, []) == []
    a, b = ct.c_uint32(7), ct.c_uint64(123)
    pointers = (ct.c_void_p * 2)(ct.addressof(a), ct.addressof(b))
    layout = [(0, 4), (8, 8)]
    expected = parameter_bytes(ct.addressof(pointers), 0, layout)
    packed = ct.create_string_buffer(16)
    ct.memmove(ct.addressof(packed), ct.addressof(a), 4)
    ct.memmove(ct.addressof(packed) + 8, ct.addressof(b), 8)
    size = ct.c_size_t(16)
    extra = (ct.c_void_p * 5)(1, ct.addressof(packed), 2, ct.addressof(size), 0)
    assert parameter_bytes(0, ct.addressof(extra), layout) == expected
    size.value = 12
    with pytest.raises(ValueError, match="exceeds"):
        parameter_bytes(0, ct.addressof(extra), layout)


def test_scalar_serialization_keeps_shape_and_bytes(harness, tmp_path):
    """Scalar and BF16 data retain actual shapes and bytes in the normal artifact writer."""
    values = {"epoch": torch.tensor(17), "head": torch.tensor([1.5, -2.0], dtype=torch.bfloat16)}
    result = harness.save_tensors(torch, tmp_path / "outputs", values)
    assert result["epoch"]["shape"] == []
    assert result["head"]["shape"] == [2]
    assert result == json.loads((tmp_path / "outputs/tensors.json").read_text())
    for name, value in values.items():
        expected = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        assert (tmp_path / "outputs" / (name + ".bin")).read_bytes() == expected
    with pytest.raises(FileExistsError):
        harness.save_tensors(torch, tmp_path / "outputs", values)


def test_factory_keeps_original_options_and_restores_profiler(harness):
    """The wrapper delegates the same original capture and clears observation on failure."""
    calls = []

    def factory(**kwargs):
        """Provide the retained closure name used by the actual factory."""

        def capture_current_tail():
            """Mark the captured closure's entry."""
            calls.append(kwargs)

        capture_current_tail()
        return "graft"

    graft, recapture = harness.capture_factory(factory, "snapshot", 123, "output")
    assert graft == "graft" and callable(recapture)
    assert calls == [
        dict(
            snapshot="snapshot",
            dist_port=123,
            fused_markov_argmax=False,
            capture_markov_control=False,
            proposal_output="output",
        )
    ]
    assert sys.getprofile() is None

    def failure(**kwargs):
        """Fail before any closure becomes available."""
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        harness.capture_factory(failure, "snapshot", 123)
    assert sys.getprofile() is None


def test_aliasing_fixture_cannot_reach_loading(harness):
    """The known aliased baseline cannot be selected by omitting its correction."""
    with pytest.raises(ValueError, match="disjoint-draft-kv-fixture"):
        harness.run(SimpleNamespace(disjoint_draft_kv_fixture=False))


@pytest.mark.parametrize("query_status", [600, 0])
def test_real_pair_loop_keeps_submission_and_timing_scope(harness, query_status):
    """Execute the actual pair loop against an inert driver and reject an already-started event."""
    tree = nodes()
    loop = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == "comparisons"
    )
    events = []
    current = [None]

    def record(kind, *args):
        """Retain every operation in the simulated submission stream."""
        events.append((kind, *args))

    def launch(name):
        """Track the selected graph without executing a device operation."""
        current[0] = name
        record("launch", name)

    cu = SimpleNamespace(
        CUstream=lambda value: value,
        CUresult=SimpleNamespace(CUDA_ERROR_NOT_READY=600),
        cuEventRecord=lambda event, stream: record("event", event),
        cuEventQuery=lambda event: (query_status,),
        cuEventSynchronize=lambda event: record("wait", event),
        cuEventElapsedTime=lambda a, b: {"original": 1, "retained": 2, "integrated": 3}[current[0]],
    )
    binding = SimpleNamespace(verify_resident=lambda: record("resident"), record=lambda: {})
    state = {"comparisons": []}
    env = dict(
        args=SimpleNamespace(equal_work=False),
        comparisons=[
            ("original", "retained"),
            ("retained", "integrated"),
            ("original", "integrated"),
        ],
        reset=lambda name: record("reset", name),
        launch=launch,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: record("sync"),
                current_stream=lambda: SimpleNamespace(cuda_stream=0),
                _sleep=lambda n: record("sleep", n),
            )
        ),
        cu=cu,
        nc=SimpleNamespace(checked=lambda value: value),
        start=1,
        end=2,
        statistics=statistics,
        state=state,
        disjoint_fixture=None,
        bands={"integrated": SimpleNamespace(io_binding=binding)},
        emit=lambda *args: None,
        out=Path("unused"),
        save=lambda: None,
    )
    code = compile(
        ast.Module(body=[loop], type_ignores=[]), str(H / "full_proposal_lab.py"), "exec"
    )
    if query_status == 0:
        with pytest.raises(ValueError, match="complete host submission"):
            exec(code, env)
        assert not state["comparisons"]
        return
    exec(code, env)
    assert len(state["comparisons"]) == 3
    for comparison in state["comparisons"]:
        assert len(comparison["pairs"]) == 40
        assert [row["round"] for row in comparison["pairs"]] == list(range(-8, 32))
        assert comparison["candidate_numerically_qualified"] is None
    assert sum(event[0] == "launch" for event in events) == 480
    for i, event in enumerate(events):
        if event == ("event", 1):
            assert events[i - 2][0] == "reset" and events[i - 1] == ("sleep", 8_000_000)
            assert events[i + 1][0] == "launch" and events[i + 2] == ("event", 2)
            assert events[i + 3] == ("wait", 2)


def test_validation_replays_before_snapshot_and_keeps_hard_gates(harness, tmp_path):
    """Stale outputs, changed native math and changed reference IDs must not reach timing."""
    definition = next(
        n for n in ast.walk(nodes()) if isinstance(n, ast.FunctionDef) and n.name == "validate"
    )
    events = []
    fields = (
        "ids",
        "output_ids",
        "corrected_logits",
        "confidence_logits",
        "calibrated_confidences",
        "kv_cache",
        "head_normalized",
        "base_logits",
        "router_indices",
        "routed_output",
        "shared_output",
    )
    values = {key: torch.ones(2) for key in fields}
    copies = {
        name: {key: value.clone() for key, value in values.items()}
        for name in ("retained", "integrated")
    }
    copies["original"] = {
        key: values[key].clone() for key in ("ids", "head_normalized", "base_logits")
    }

    def evidence(name):
        """Require a completed same-arm replay immediately before reading results."""
        assert events[-2:] == [("launch", name), ("sync",)]
        return copies[name]

    band = SimpleNamespace(
        proposal_outputs=SimpleNamespace(output_ids=torch.ones(6)),
        keepalive=(None, SimpleNamespace(out=torch.ones(2), status=torch.zeros(1))),
        io_binding=SimpleNamespace(verify_resident=lambda: None, record=lambda: {}),
    )
    env = dict(
        executables=["original", "retained", "integrated"],
        reference_names={"original"},
        args=SimpleNamespace(equal_work=False),
        reset=lambda n: events.append(("reset", n)),
        launch=lambda n: events.append(("launch", n)),
        evidence=evidence,
        bands={"retained": band, "integrated": band},
        torch=SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: events.append(("sync",))),
            count_nonzero=torch.count_nonzero,
        ),
        error=lambda ignored, *a, **k: harness.error(torch, *a, **k),
        save_tensors=lambda *args: None,
        emit=lambda *args: None,
        state={"correctness": {}},
        save=lambda: None,
        out=tmp_path,
        disjoint_fixture=None,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "validation", "exec"), env)
    env["validate"]("before")
    assert len(env["state"]["correctness"]["before"]["retained_vs_integrated"]) == 11
    copies["integrated"]["routed_output"][0] = 2
    with pytest.raises(ValueError, match="retained proposal math"):
        env["validate"]("changed-math")
    copies["integrated"]["routed_output"][0] = 1
    copies["original"]["ids"][0] = 2
    with pytest.raises(ValueError, match="reference IDs"):
        env["validate"]("changed-ids")
