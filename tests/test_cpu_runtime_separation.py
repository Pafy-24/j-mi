from __future__ import annotations

import sys
import threading
from types import SimpleNamespace


class _CudaMustNotBeProbed:
    def is_available(self):
        raise AssertionError("CPU-bound paths must not query CUDA availability")


class _CudaProbe:
    def __init__(self):
        self.availability_calls = 0

    def is_available(self):
        self.availability_calls += 1
        return True

    def device_count(self):
        return 2

    def empty_cache(self):
        self.empty_cache_calls = getattr(self, "empty_cache_calls", 0) + 1


class _Registry:
    root = "facets"

    def list(self):
        return []

    def loaded(self):
        return []

    def unload(self, *args, **kwargs):
        return []

    def cache_keys_for(self, *args, **kwargs):
        return []


class _Residency:
    def snapshot(self):
        return {}

    def reset(self):
        return None

    def discard(self, keys):
        return None


def _cpu_runtime(monkeypatch):
    from mixed_inference.generate import runtime as runtime_module

    fake_torch = SimpleNamespace(cuda=_CudaMustNotBeProbed())
    monkeypatch.setattr(runtime_module, "require_torch", lambda: fake_torch)
    runtime = runtime_module.MixedInferenceRuntime.__new__(runtime_module.MixedInferenceRuntime)
    runtime._lock = threading.RLock()
    runtime._started_at = 0.0
    runtime._generation_count = 0
    runtime._cache_telemetry = {}
    runtime.registry = _Registry()
    runtime.residency = _Residency()
    runtime.base = SimpleNamespace(
        device="cpu",
        binding=SimpleNamespace(model_id="cpu-base"),
        summary=lambda: {"device": "cpu"},
    )
    return runtime


def test_cpu_runtime_health_does_not_probe_cuda(monkeypatch) -> None:
    runtime = _cpu_runtime(monkeypatch)

    assert runtime.health()["memory"] == {"status": "external", "authority": "RMA"}


def test_cpu_runtime_unload_does_not_probe_cuda(monkeypatch) -> None:
    runtime = _cpu_runtime(monkeypatch)

    assert runtime.unload_base() == {"unloaded": "cpu-base"}


def test_cpu_runtime_facet_unload_does_not_probe_cuda(monkeypatch) -> None:
    runtime = _cpu_runtime(monkeypatch)

    assert runtime.unload_facets(all_facets=True) == {"unloadedFacets": [], "loadedFacets": []}


def test_cpu_seed_uses_only_default_generator() -> None:
    from mixed_inference.generate.runtime import _seed_for_base

    seeded = []
    torch = SimpleNamespace(
        default_generator=SimpleNamespace(manual_seed=lambda value: seeded.append(value)),
        manual_seed=lambda value: (_ for _ in ()).throw(AssertionError("global seed must not run")),
        cuda=_CudaMustNotBeProbed(),
    )

    _seed_for_base(torch, SimpleNamespace(device="cpu"), 17)

    assert seeded == [17]


def test_cuda_unload_keeps_cuda_cleanup(monkeypatch) -> None:
    from mixed_inference.generate import runtime as runtime_module

    cuda = _CudaProbe()
    monkeypatch.setattr(runtime_module, "require_torch", lambda: SimpleNamespace(cuda=cuda))
    runtime = _cpu_runtime(monkeypatch)
    monkeypatch.setattr(runtime_module, "require_torch", lambda: SimpleNamespace(cuda=cuda))
    runtime.base.device = "cuda:0"

    runtime.unload_base()

    assert cuda.availability_calls == 1
    assert cuda.empty_cache_calls == 1


def test_cpu_bound_worker_capabilities_do_not_probe_cuda(monkeypatch) -> None:
    from mixed_inference.worker.service import MIWorkerService

    monkeypatch.setenv("JESTER_RM_DEVICE_BACKEND", "cpu")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=_CudaMustNotBeProbed(), version=SimpleNamespace(hip=None)),
    )

    result = MIWorkerService.__new__(MIWorkerService).capabilities()

    assert [item["backend"] for item in result["backends"]] == ["cpu"]


def test_cuda_bound_worker_capabilities_keep_cuda_probe(monkeypatch) -> None:
    from mixed_inference.worker.service import MIWorkerService

    cuda = _CudaProbe()
    monkeypatch.setenv("JESTER_RM_DEVICE_BACKEND", "torch.cuda")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip=None)),
    )

    result = MIWorkerService.__new__(MIWorkerService).capabilities()

    assert cuda.availability_calls == 1
    assert result["backends"][1]["backend"] == "torch.cuda"
    assert result["backends"][1]["deviceCount"] == 2
