"""CPU-only confidence-prefix loader guards and build-cache behavior."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepspec.megakernel import extension as extension_api

# Minimum legal configuration for the certified full proposal. These values
# include the dependencies enforced by the ordinary loader for width128,
# stage3, shared continuation, and the existing HC/WO_A schedule.
CERTIFIED_ENV = {
    "DSPARK_V4_STATIC_QUEUES": "1",
    "DSPARK_V4_ROUTED_DYNAMIC_CLAIMS": "1",
    "DSPARK_V4_SHARED_STAGE_CONTEXT": "1",
    "DSPARK_V4_SHARED_FIRST": "1",
    "DSPARK_SHARED_CONTINUE": "1",
    "DSPARK_ROUTED_W2_WORKERS": "128",
    "DSPARK_ROUTED_WEIGHT_WARP": "1",
    "DSPARK_ROUTED_W13_STAGE3": "1",
    "DSPARK_V4_LAUNCH_SMEM_BYTES": "227328",
    "DSPARK_V4_DYNAMIC_SMEM_BYTES": "153600",
    "DSPARK_ROUTED_PACKED_SFA": "2",
    "DSPARK_ROUTED_GROUP_MASKS": "1",
    "DSPARK_ROUTED_COMPACT_TILES": "1",
    "DSPARK_ROUTED_COMPACT_WORKERS": "120",
    "DSPARK_ROUTED_CLAIM_CHUNK": "8",
    "DSPARK_ROUTED_FUSED_SWIGLU": "1",
    "DSPARK_W13_DG_CHUNKS": "16",
    "DSPARK_W13_DG_STAGES": "2",
    "DSPARK_W13_DG_TILE_PAIRS": "1",
    "DSPARK_SWIGLU_WARP_QUANT": "1",
    "DSPARK_HC_SPLIT": "1",
    "DSPARK_HC_PREFETCH": "1",
    "DSPARK_HC_ELEMENT_LANES": "1",
    "DSPARK_WOA_SPLIT_K": "2",
    "DSPARK_WOA_BULK_STAGING": "1",
    "DSPARK_WOA_STREAMED_K": "256",
    "DSPARK_LM_BULK_STAGING": "1",
    "DSPARK_LM_STREAMED_K": "256",
}


@pytest.fixture
def api(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("DSPARK_"):
            monkeypatch.delenv(key)
    for key, value in CERTIFIED_ENV.items():
        monkeypatch.setenv(key, value)
    kwargs = {
        "instrumented": False,
        "relaxed_dag": True,
        "greedy_tail": True,
        "batch_size": 1,
        "full_loop_device_epoch": True,
        "compiled_execution_mask": extension_api.V4_EXECUTE_ALL,
        "compiled_draft_layer_mask": 7,
    }
    monkeypatch.setattr(extension_api, "_cutlass_include_path", lambda: Path("/mock/cutlass"))
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(extension_api, "load", load)
    extension_api.load_dspark_v4_scheduler_extension.cache_clear()
    try:
        yield extension_api, kwargs, calls
    finally:
        extension_api.load_dspark_v4_scheduler_extension.cache_clear()


def test_default_and_opt_out_are_distinct_cached_builds(api, monkeypatch):
    a, kw, calls = api
    enabled = a.load_dspark_v4_scheduler_extension(**kw)
    assert enabled.extra_cuda_cflags.count("-DDSPARK_CONFIDENCE_HEAD_PREFIX=1") == 1
    assert "-DDSPARK_CONFIDENCE_HEAD_PREFIX=1" not in enabled.extra_cflags
    monkeypatch.setenv("DSPARK_CONFIDENCE_HEAD_PREFIX", "0")
    disabled = a.load_dspark_v4_scheduler_extension(**kw)
    assert enabled is not disabled and enabled.name != disabled.name
    assert len(enabled.name) <= 190 and len(disabled.name) <= 190
    assert enabled.extra_cuda_cflags == disabled.extra_cuda_cflags + [
        "-DDSPARK_CONFIDENCE_HEAD_PREFIX=1"
    ]
    assert enabled.extra_cflags == disabled.extra_cflags
    assert enabled.extra_ldflags == disabled.extra_ldflags
    monkeypatch.setenv("DSPARK_CONFIDENCE_HEAD_PREFIX", "1")
    assert a.load_dspark_v4_scheduler_extension(**kw) is enabled
    monkeypatch.delenv("DSPARK_CONFIDENCE_HEAD_PREFIX")
    assert a.load_dspark_v4_scheduler_extension(**kw) is enabled
    assert len(calls) == 2 and a.load_dspark_v4_scheduler_extension.cache_info().hits == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("instrumented", True),
        ("relaxed_dag", False),
        ("greedy_tail", False),
        ("static_tp2_tail", True),
        ("batch_size", 2),
        ("tp2_ablate", True),
        ("full_loop_device_epoch", False),
        ("compiled_execution_mask", 1023),
        ("compiled_execution_mask", None),
        ("compiled_draft_layer_mask", 3),
    ],
)
def test_argument_guards(api, monkeypatch, key, value):
    a, kw, _ = api
    kw[key] = value
    assert not a._confidence_head_prefix_enabled(**kw)
    monkeypatch.setenv("DSPARK_CONFIDENCE_HEAD_PREFIX", "1")
    with pytest.raises(ValueError, match="certified"):
        a._confidence_head_prefix_enabled(**kw)


@pytest.mark.parametrize(
    "key",
    [
        "DSPARK_V4_STATIC_QUEUES",
        "DSPARK_V4_ROUTED_DYNAMIC_CLAIMS",
        "DSPARK_V4_SHARED_STAGE_CONTEXT",
        "DSPARK_SHARED_CONTINUE",
        "DSPARK_ROUTED_W2_WORKERS",
        "DSPARK_LM_BULK_STAGING",
        "DSPARK_LM_STREAMED_K",
    ],
)
def test_required_environment(api, monkeypatch, key):
    a, kw, _ = api
    monkeypatch.delenv(key)
    assert not a._confidence_head_prefix_enabled(**kw)


@pytest.mark.parametrize(
    "key",
    [
        "DSPARK_V4_STATIC_TP2_TAIL",
        "DSPARK_V4_TP2_ABLATE",
        "DSPARK_V4_DIRECT_DEPENDENCIES",
        "DSPARK_V4_COMPLETION_RELEASE",
        "DSPARK_V4_QUEUE_LOOKAHEAD",
        "DSPARK_ROUTED_GROUP_READY",
        "DSPARK_MARKOV_GATHER_CONTINUE",
        "DSPARK_V4_FINE_GRAINED_OVERLAP",
        "DSPARK_V4_STATIC_GAPS",
        "DSPARK_HEAD_SPLIT",
        "DSPARK_HEAD_PREFETCH",
        "DSPARK_V4_PERSISTENT_TMEM",
        "DSPARK_PHASE_OUTLINE",
        "DSPARK_SCHEDULER_HANDOFF",
        "DSPARK_FRONT_EMBED_FIRST",
        "DSPARK_ROUTED_INTERLEAVE",
        "DSPARK_SHARED_SPLIT_K",
        "DSPARK_V4_ENABLE_DEVICE_TRACE",
        "DSPARK_ENABLE_DEVICE_TRACE",
        "DSPARK_V4_PHASE_TIMESTAMPS",
        "DSPARK_V4_HOST_PROGRESS",
        "DSPARK_V4_BOUNDARY_TIMESTAMPS",
        "DSPARK_HEAD_CLOCK",
        "DSPARK_ATTN_CLOCK",
        "DSPARK_HEAD_CLOCK_PROBE",
        "DSPARK_ATTN_CLOCK_PROBE",
    ],
)
def test_excluded_environment(api, monkeypatch, key):
    a, kw, _ = api
    monkeypatch.setenv(key, "1")
    assert not a._confidence_head_prefix_enabled(**kw)
    monkeypatch.setenv("DSPARK_CONFIDENCE_HEAD_PREFIX", "1")
    with pytest.raises(ValueError, match="certified"):
        a._confidence_head_prefix_enabled(**kw)


def test_trace_and_optout_legal_original_loader(api, monkeypatch):
    a, kw, _ = api
    kw["instrumented"] = True
    r = a.load_dspark_v4_scheduler_extension(**kw)
    assert "-DDSPARK_V4_ENABLE_DEVICE_TRACE=1" in r.extra_cuda_cflags
    assert "-DDSPARK_CONFIDENCE_HEAD_PREFIX=1" not in r.extra_cuda_cflags


@pytest.mark.parametrize("value", ["yes", "2", ""])
def test_bad_switch(api, monkeypatch, value):
    a, kw, _ = api
    monkeypatch.setenv("DSPARK_CONFIDENCE_HEAD_PREFIX", value)
    with pytest.raises(ValueError, match="0 or 1"):
        a.load_dspark_v4_scheduler_extension(**kw)
