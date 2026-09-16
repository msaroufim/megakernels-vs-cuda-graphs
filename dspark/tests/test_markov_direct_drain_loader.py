"""CPU checks for selecting and caching the one-column Markov program."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepspec.megakernel import extension as extension_api


@pytest.fixture
def loader(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("DSPARK_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(extension_api, "_cutlass_include_path", lambda: Path("/mock/cutlass"))
    monkeypatch.setattr(extension_api, "load", lambda **kw: SimpleNamespace(**kw))
    extension_api.load_dspark_v4_scheduler_extension.cache_clear()
    try:
        yield extension_api.load_dspark_v4_scheduler_extension
    finally:
        extension_api.load_dspark_v4_scheduler_extension.cache_clear()


def test_bulk_programs_select_distinct_native_modules(loader, monkeypatch):
    kwargs = {"relaxed_dag": True, "greedy_tail": True, "batch_size": 1}
    original = loader(**kwargs)
    # Bulk K is an existing process-level build choice, so changing it in a
    # diagnostic requires clearing the Python loader cache.
    loader.cache_clear()
    monkeypatch.setenv("DSPARK_MARKOV_BULK_K", "128")
    direct = loader(**kwargs)
    assert loader(**kwargs) is direct
    loader.cache_clear()
    monkeypatch.setenv("DSPARK_MARKOV_BULK_K", "256")
    monkeypatch.setenv("DSPARK_V4_DYNAMIC_SMEM_BYTES", "153600")
    wider = loader(**kwargs)
    flag = "-DDSPARK_MARKOV_DIRECT_DRAIN=1"
    assert direct.extra_cuda_cflags.count(flag) == 1
    assert flag not in original.extra_cuda_cflags
    assert flag not in wider.extra_cuda_cflags
    assert flag not in direct.extra_cflags
    assert len({original.name, direct.name, wider.name}) == 3
    assert "_mkdrain1" in direct.name
    assert all(len(x.name) <= 190 for x in (original, direct, wider))
    assert loader(**kwargs) is wider


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relaxed_dag": False, "greedy_tail": True, "batch_size": 1},
        {"relaxed_dag": True, "greedy_tail": False, "batch_size": 1},
        {"relaxed_dag": True, "greedy_tail": True, "batch_size": 2},
    ],
)
def test_bulk_drain_keeps_existing_shape_and_mode_guards(loader, monkeypatch, kwargs):
    monkeypatch.setenv("DSPARK_MARKOV_BULK_K", "128")
    with pytest.raises(ValueError):
        loader(**kwargs)
