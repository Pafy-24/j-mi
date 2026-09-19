from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mixed_inference.core.errors import CompositionError
from mixed_inference.core.graph import CompositionPlan
from mixed_inference.core.modeling import LayerStackHandle, LoadedBaseModel
from mixed_inference.core.residency import Footprint, ResidencyBudget, ResidencyManager
from mixed_inference.core.specs import BaseBinding, BaseModelRequest
from mixed_inference.generate.runtime import MixedInferenceRuntime


def _base() -> LoadedBaseModel:
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(num_hidden_layers=1, hidden_size=1)
    model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
    return LoadedBaseModel(
        request=BaseModelRequest("base"),
        model=model,
        tokenizer=object(),
        stack=LayerStackHandle("layers", model, "layers", model.layers),
        binding=BaseBinding("base", None, "config", "layers", 1),
        device=torch.device("cpu"),
    )


class _Facet:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(facet_id="facet", version="1")
        self._parameter = torch.nn.Parameter(torch.ones(2))

    def parameters(self):
        return iter((self._parameter,))


def test_generate_releases_admission_when_generation_validation_fails(tmp_path, monkeypatch) -> None:
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=10_000))
    )
    runtime.base = _base()
    runtime.residency.reset(base=Footprint(ram_bytes=8))
    plan = CompositionPlan(requested=("facet",), resolved=("facet@1",), blocks=())
    facet = _Facet()
    monkeypatch.setattr(runtime.registry, "resolve", lambda _requested: (plan, {"facet": object()}))
    monkeypatch.setattr(runtime, "_preflight_facet_staging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime.registry, "cache_key", lambda *_args: "facet-cache")
    monkeypatch.setattr(runtime.registry, "evict_cache_keys", lambda _keys: None)

    def load_resolved(*_args, before_commit, **_kwargs):
        loaded = {"facet": facet}
        before_commit(loaded)
        return plan, loaded

    monkeypatch.setattr(runtime.registry, "load_resolved", load_resolved)
    monkeypatch.setattr(
        runtime,
        "_prepare_inputs",
        lambda _request: {"input_ids": torch.ones((1, 1), dtype=torch.long)},
    )
    with pytest.raises(ValueError, match="invalid literal"):
        runtime.generate(
            {"prompt": "x", "facets": ["facet"], "generation": {}, "seed": "invalid"}
        )
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    assert runtime.residency.snapshot()["entries"] == 1


def test_bounded_base_replacement_rejects_unknown_or_oversized_staging_before_load(tmp_path, monkeypatch) -> None:
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=8))
    )
    old = _base()
    runtime.base = old
    runtime.residency.reset(base=Footprint(ram_bytes=8))
    candidate_path = tmp_path / "candidate.bin"
    candidate_path.write_bytes(b"0123456789")
    monkeypatch.setattr(
        "mixed_inference.generate.runtime.load_base_model",
        lambda *_args, **_kwargs: pytest.fail("candidate must not be constructed"),
    )
    with pytest.raises(CompositionError, match="base_residency_denied"):
        runtime.load_base(BaseModelRequest(str(candidate_path)))
    assert runtime.base is old


def test_residency_snapshot_preserves_unbounded_as_null() -> None:
    snapshot = ResidencyManager().snapshot()
    assert snapshot["ramBudgetBytes"] is None
    assert snapshot["acceleratorBudgetBytes"] is None


def test_runtime_drops_residency_entries_when_registry_invalidates_or_unloads(
    tmp_path, monkeypatch
) -> None:
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=100))
    )
    runtime.residency.reset(base=Footprint(ram_bytes=1))
    key = ((), "facet", "1", "digest")
    admission = runtime.residency.admit({key: Footprint(ram_bytes=10)})
    runtime.residency.release(admission)
    monkeypatch.setattr(runtime.registry, "refresh", lambda: (key,))
    monkeypatch.setattr(runtime.registry, "list", lambda: [])
    runtime.refresh_registry()
    assert runtime.residency.snapshot()["entries"] == 0

    admission = runtime.residency.admit({key: Footprint(ram_bytes=10)})
    runtime.residency.release(admission)
    monkeypatch.setattr(runtime.registry, "cache_keys_for", lambda *_args, **_kwargs: (key,))
    monkeypatch.setattr(runtime.registry, "unload", lambda *_args, **_kwargs: ["facet@1"])
    runtime.unload_facets(all_facets=True)
    assert runtime.residency.snapshot()["entries"] == 0


def test_load_facets_admits_cache_then_releases_the_preload_pin(tmp_path, monkeypatch) -> None:
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=100))
    )
    runtime.base = _base()
    runtime.residency.reset(base=Footprint(ram_bytes=8))
    plan = CompositionPlan(requested=("facet",), resolved=("facet@1",), blocks=())
    facet = _Facet()
    monkeypatch.setattr(runtime.registry, "resolve", lambda _requested: (plan, {"facet": object()}))
    monkeypatch.setattr(runtime, "_preflight_facet_staging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime.registry, "cache_key", lambda *_args: "facet-cache")
    monkeypatch.setattr(runtime.registry, "evict_cache_keys", lambda _keys: None)
    monkeypatch.setattr(runtime.registry, "loaded", lambda: ["facet@1"])

    def load_resolved(*_args, before_commit, **_kwargs):
        loaded = {"facet": facet}
        before_commit(loaded)
        return plan, loaded

    monkeypatch.setattr(runtime.registry, "load_resolved", load_resolved)
    result = runtime.load_facets(["facet"])
    assert result["loadedFacets"] == ["facet@1"]
    assert runtime.residency.snapshot()["entries"] == 1
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
