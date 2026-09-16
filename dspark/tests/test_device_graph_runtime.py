from types import SimpleNamespace

import pytest

from deepspec.megakernel import device_graph_runtime as runtime
from deepspec.megakernel.device_graph_runtime import DeviceGraphExecutable


def test_composed_device_graph_rejects_empty_program_before_loading_cuda():
    with pytest.raises(ValueError, match="at least one child"):
        DeviceGraphExecutable.compose(())


def graph(handle):
    return SimpleNamespace(raw_cuda_graph=lambda: handle)


@pytest.mark.parametrize("device_launch", [True, False])
def test_compose_forwards_ordered_children_to_selected_binding(monkeypatch, device_launch):
    calls = []

    def instantiate(label, handles):
        calls.append((label, handles))
        return 91

    extension = SimpleNamespace(
        instantiate_composed=lambda handles: instantiate("device", handles),
        instantiate_composed_host=lambda handles: instantiate("host", handles),
    )
    monkeypatch.setattr(runtime, "load_device_graph_runtime_extension", lambda: extension)
    result = DeviceGraphExecutable.compose((graph(20), graph(10)), device_launch=device_launch)
    assert result.handle == 91
    assert calls == [("device" if device_launch else "host", [20, 10])]


def test_compose_default_uses_unchanged_legacy_binding(monkeypatch):
    calls = []
    # Default callers do not access the newly added host-only export.
    extension = SimpleNamespace(instantiate_composed=lambda handles: calls.append(handles) or 92)
    monkeypatch.setattr(runtime, "load_device_graph_runtime_extension", lambda: extension)
    assert DeviceGraphExecutable.compose((graph(7),)).handle == 92
    assert calls == [[7]]


@pytest.mark.parametrize("invalid", [None, 0, 1, "False"])
def test_compose_requires_boolean_before_loading_cuda(monkeypatch, invalid):
    def unexpected_load():
        pytest.fail("invalid policy must not load CUDA")

    monkeypatch.setattr(runtime, "load_device_graph_runtime_extension", unexpected_load)
    with pytest.raises(TypeError, match="must be a bool"):
        DeviceGraphExecutable.compose((graph(7),), device_launch=invalid)


def test_compose_policy_is_keyword_only():
    with pytest.raises(TypeError):
        DeviceGraphExecutable.compose((graph(7),), False)


@pytest.mark.parametrize("device_launch", [True, False])
def test_compose_propagates_native_failure_without_an_executable(monkeypatch, device_launch):
    def fail(handles):
        assert handles == [7]
        raise RuntimeError("instantiate failed")

    extension = SimpleNamespace(instantiate_composed=fail, instantiate_composed_host=fail)
    monkeypatch.setattr(runtime, "load_device_graph_runtime_extension", lambda: extension)
    with pytest.raises(RuntimeError, match="instantiate failed"):
        DeviceGraphExecutable.compose((graph(7),), device_launch=device_launch)


@pytest.mark.parametrize("device_launch", [True, False])
def test_composed_owner_context_error_closes_once_and_rejects_later_launch(
    monkeypatch, device_launch
):
    calls = []
    extension = SimpleNamespace(
        instantiate_composed=lambda handles: 99,
        instantiate_composed_host=lambda handles: 99,
        launch=lambda handle: calls.append(("launch", handle)),
        destroy=lambda handle: calls.append(("destroy", handle)),
    )
    monkeypatch.setattr(runtime, "load_device_graph_runtime_extension", lambda: extension)
    owner = DeviceGraphExecutable.compose((graph(7),), device_launch=device_launch)
    with pytest.raises(RuntimeError, match="body failed"):
        with owner:
            owner.launch()
            raise RuntimeError("body failed")
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.launch()
    assert calls == [("launch", 99), ("destroy", 99)]


def test_single_graph_instantiation_remains_unchanged(monkeypatch):
    calls = []
    extension = SimpleNamespace(instantiate=lambda handle: calls.append(handle) or 93)
    monkeypatch.setattr(runtime, "load_device_graph_runtime_extension", lambda: extension)
    assert DeviceGraphExecutable.instantiate(graph(8)).handle == 93
    assert calls == [8]
