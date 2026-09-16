"""CPU-only capture wiring and loader guards; these do not certify CUDA math."""

import contextlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from deepspec.megakernel import extension as extension_api
from deepspec.megakernel import full_loop_proposal as proposal


@pytest.fixture
def loader(monkeypatch):
    """Exercise the real option validation without compiling or loading CUDA."""
    for name in tuple(os.environ):
        if name.startswith("DSPARK_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(extension_api, "_cutlass_include_path", lambda: Path("/mock/cutlass"))
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(extension_api, "load", load)
    extension_api.load_dspark_v4_scheduler_extension.cache_clear()
    yield calls
    extension_api.load_dspark_v4_scheduler_extension.cache_clear()


def test_device_epoch_normalized_and_greedy_builds_remain_distinct(loader):
    """Both existing DAGs receive device epochs; the greedy define stays opt-in."""
    common = {"relaxed_dag": True, "full_loop_device_epoch": True}
    normal = extension_api.load_dspark_v4_scheduler_extension(**common)
    greedy = extension_api.load_dspark_v4_scheduler_extension(**common, greedy_tail=True)
    assert normal.name != greedy.name
    assert len(loader) == 2
    for module in (normal, greedy):
        assert "-DDSPARK_V4_RELAXED_DAG=1" in module.extra_cuda_cflags
        assert "-DDSPARK_V4_FULL_LOOP_DEVICE_EPOCH=1" in module.extra_cuda_cflags
    assert "-DDSPARK_V4_GREEDY_TAIL=1" not in normal.extra_cuda_cflags
    assert "-DDSPARK_V4_GREEDY_TAIL=1" in greedy.extra_cuda_cflags
    assert "-DDSPARK_CONFIDENCE_HEAD_PREFIX=1" not in normal.extra_cuda_cflags
    assert extension_api.load_dspark_v4_scheduler_extension(**common) is normal


@pytest.mark.parametrize(
    "kwargs,environment,message",
    [
        ({"relaxed_dag": False}, {}, "batch-1 relaxed scheduler"),
        ({"batch_size": 2}, {}, "batch_size > 1"),
        ({}, {"DSPARK_V4_STATIC_QUEUES": "1"}, "relaxed greedy GB300"),
        (
            {"compiled_execution_mask": 2047, "compiled_draft_layer_mask": 7},
            {},
            "compiled phase masks require",
        ),
        ({}, {"DSPARK_CONFIDENCE_HEAD_PREFIX": "1"}, "greedy configuration"),
    ],
)
def test_normalized_mode_keeps_other_loader_restrictions(
    loader, monkeypatch, kwargs, environment, message
):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    options = {"relaxed_dag": True, "greedy_tail": False, "full_loop_device_epoch": True}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        extension_api.load_dspark_v4_scheduler_extension(**options)
    assert not loader


@pytest.fixture
def capture(monkeypatch):
    """Run the Python capture boundary with CPU tensors and inert CUDA stand-ins."""
    for name in tuple(os.environ):
        if name.startswith("DSPARK_"):
            monkeypatch.delenv(name)
    workspace = SimpleNamespace(scratch=torch.zeros(32, dtype=torch.uint8))
    outputs = SimpleNamespace(
        output_ids=torch.zeros((1, 6), dtype=torch.int32),
        draft_probabilities=object(),
        scheduled_prefix_lengths=object(),
        scheduler_read_mask=object(),
    )
    gathered = []
    accessor = SimpleNamespace(
        out=torch.empty((3, 128, 512), dtype=torch.bfloat16),
        status=torch.zeros(1, dtype=torch.int32),
        gather=lambda *args, **kw: gathered.append((args, kw)),
    )
    packed = SimpleNamespace(
        arena=object(), offsets=object(), freqs_cis=torch.zeros((8, 32), dtype=torch.complex64)
    )
    for name, attrs in (
        ("boltin_kv_accessor", {"bind_full_loop_graft_kv_accessor": lambda _: accessor}),
        ("boltin_pack", {"pack_boltin_arena": lambda *a, **k: packed}),
    ):
        module = ModuleType(f"scripts.megakernel.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(proposal, "allocate_v4_scheduler_workspace", lambda **kw: workspace)
    monkeypatch.setattr(proposal, "allocate_v4_proposal_outputs", lambda **kw: outputs)
    monkeypatch.setattr(proposal, "_require_device_launchable", lambda *a, **kw: (0,))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=10, minor=3, multi_processor_count=152),
    )
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda **kw: SimpleNamespace(replay=lambda: None))
    monkeypatch.setattr(torch.cuda, "graph", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    prepared = []
    monkeypatch.setattr(proposal, "prepare_proposal_megakernel", lambda **kw: prepared.append(kw))
    launches = []
    monkeypatch.setattr(
        proposal, "run_v4_front_kv_stage", lambda *a, **kw: launches.append((a, kw))
    )
    monkeypatch.setattr(proposal, "publish_proposal_candidates", lambda src, dst: dst.copy_(src[0]))
    arguments = dict(
        snapshot=Path("/mock/checkpoint"),
        graft=object(),
        target_hidden=torch.empty((6, 12288), dtype=torch.bfloat16),
        bonus=torch.ones(1, dtype=torch.int64),
        commit_len=torch.ones(1, dtype=torch.int64),
        new_seq_len=torch.ones(1, dtype=torch.int64),
        candidates=torch.empty(6, dtype=torch.int64),
        lowering=SimpleNamespace(executor="proposal_megakernel"),
    )
    return arguments, launches, prepared, gathered, outputs, workspace, accessor


@pytest.mark.parametrize("normalized", [False, True])
def test_capture_preserves_owners_epoch_and_tail_arguments(capture, normalized):
    arguments, launches, prepared, gathered, outputs, workspace, accessor = capture
    controlled = torch.tensor([[0.137, 0.731, 0.419, 0.887, 0.263]])
    sts = torch.tensor([0.9, 1.1, 1.3, 0.8, 1.2])
    sps = torch.tensor([0.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2])
    if normalized:
        arguments.update(
            greedy_tail=False,
            sampling_temperature=1.0,
            uniforms=controlled,
            sts_temperatures=sts,
            steps_per_second=sps,
        )
    band = proposal.capture_proposal_megakernel_band(**arguments)
    assert len(launches) == 3  # Two warmups and one captured call; mocked replay is inert.
    assert len(prepared) == len(gathered) == 3
    assert band.proposal_outputs is outputs is band.keepalive[3]
    for args, kwargs in launches:
        assert args[5] is accessor.out
        assert kwargs["workspace"] is workspace
        assert kwargs["proposal_outputs"] is outputs
        assert kwargs["start_pos"] == kwargs["proposal_epoch"] == -1
        assert kwargs["full_loop_device_epoch"] and kwargs["relaxed_dag"]
        assert kwargs["greedy_tail"] is (not normalized)
        assert kwargs["sampling_temperature"] == float(normalized)
        if normalized:
            assert kwargs["uniforms"] is controlled is band.keepalive[13]
            assert kwargs["sts_temperatures"] is sts is band.keepalive[14]
            assert kwargs["steps_per_second"] is sps is band.keepalive[15]
        else:
            assert torch.equal(kwargs["uniforms"], torch.zeros((1, 5)))
            assert torch.equal(kwargs["sts_temperatures"], torch.ones(5))
    assert prepared[0]["epoch"].data_ptr() == workspace.scratch.data_ptr() + 24
    assert gathered[0][1]["expected_pos"].data_ptr() == workspace.scratch.data_ptr() + 16


@pytest.mark.parametrize(
    "options,message",
    [
        ({"sampling_temperature": -1.0}, "finite and non-negative"),
        ({"sampling_temperature": float("nan")}, "finite and non-negative"),
        ({"sampling_temperature": float("inf")}, "finite and non-negative"),
        ({"sampling_temperature": 1.0}, "requires greedy_tail=False"),
        ({"greedy_tail": False, "sampling_temperature": 1.0}, "controlled uniforms"),
        (
            {"greedy_tail": False, "extension_module": ModuleType("unknown")},
            "normal extension loader",
        ),
        ({"uniforms": torch.zeros(5)}, "uniforms must be contiguous float32"),
        ({"sts_temperatures": torch.ones(5, dtype=torch.float64)}, "sts_temperatures must"),
        ({"steps_per_second": torch.ones(14)[::2]}, "steps_per_second must"),
    ],
)
def test_invalid_tail_inputs_fail_before_prepare_or_launch(capture, options, message):
    arguments, launches, prepared, *_ = capture
    with pytest.raises(ValueError, match=message):
        proposal.capture_proposal_megakernel_band(**arguments, **options)
    assert not launches and not prepared


@pytest.mark.parametrize(
    "marker",
    [
        "dspark_full_proposal_probe_abi",
        "dspark_full_proposal_group32_abi",
        "dspark_full_proposal_group32_probe_abi",
        "dspark_full_proposal_stream_boundaries_abi",
        "dspark_full_proposal_layer0_boundaries_abi",
        "dspark_full_proposal_attention_boundaries_abi",
    ],
)
def test_archived_images_fail_before_allocation_or_launch(capture, monkeypatch, marker):
    """Removing an observer API must not admit its image with an undersized workspace."""
    arguments, launches, prepared, gathered, *_ = capture
    module = ModuleType("archived")
    module.__file__ = "/mock/archived.so"
    monkeypatch.setattr(proposal.ctypes, "CDLL", lambda _: SimpleNamespace(**{marker: object()}))
    with pytest.raises(ValueError, match="original host bindings"):
        proposal.capture_proposal_megakernel_band(**arguments, extension_module=module)
    assert not launches and not prepared and not gathered
