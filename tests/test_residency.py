from __future__ import annotations

import pytest
import torch

from mixed_inference.core.errors import CompatibilityError, CompositionError
from mixed_inference.core.modeling import LayerStackHandle, LoadedBaseModel, validate_base_binding
from mixed_inference.core.residency import Footprint, ResidencyBudget, ResidencyManager
from mixed_inference.core.registry import FacetRegistry
from mixed_inference.core.specs import BaseBinding, BaseModelRequest, BlockSpec, FacetManifest, SlotSpec


def test_pinned_entries_are_not_evicted_and_inactive_entries_use_lru() -> None:
    manager = ResidencyManager(ResidencyBudget(ram_bytes=100, operational_margin_bytes=10))
    manager.reset(base=Footprint(ram_bytes=20))
    first = manager.admit({"old": Footprint(ram_bytes=30)})
    manager.release(first)
    active = manager.admit({"active": Footprint(ram_bytes=30)})
    replacement = manager.admit({"new": Footprint(ram_bytes=40)})
    assert replacement.evicted == ("old",)
    assert manager.snapshot()["pinnedEntries"] == 2
    manager.release(replacement)
    manager.release(active)


def test_admission_failure_is_transactional_and_does_not_drop_old_cache() -> None:
    manager = ResidencyManager(ResidencyBudget(ram_bytes=80))
    manager.reset(base=Footprint(ram_bytes=20))
    admitted = manager.admit({"retained": Footprint(ram_bytes=30)})
    manager.release(admitted)
    before = manager.snapshot()
    with pytest.raises(CompositionError, match="composition_residency_denied"):
        manager.admit({"too-large": Footprint(ram_bytes=70)})
    assert manager.snapshot() == before


def test_kv_and_margin_are_part_of_admission() -> None:
    manager = ResidencyManager(ResidencyBudget(ram_bytes=100, operational_margin_bytes=10))
    manager.reset(base=Footprint(ram_bytes=30))
    with pytest.raises(CompositionError, match="composition_residency_denied"):
        manager.admit({"facet": Footprint(ram_bytes=40)}, kv_cache=Footprint(ram_bytes=25))


def _manifest(name: str) -> FacetManifest:
    return FacetManifest(
        facet_id=name,
        version="1",
        base=BaseBinding("base", None, "config", "layers", 1),
        slots=(SlotSpec("slot", 0, 0, (BlockSpec(0),)),),
    )


def test_registry_stages_all_loads_before_cache_mutation(tmp_path, monkeypatch) -> None:
    registry = FacetRegistry(tmp_path)
    one, two = _manifest("one"), _manifest("two")
    registry._manifests = {(one.facet_id, one.version): (one, tmp_path), (two.facet_id, two.version): (two, tmp_path)}
    plan = type("Plan", (), {})()
    monkeypatch.setattr(registry, "resolve", lambda _requested: (plan, {"one": one, "two": two}))
    monkeypatch.setattr("mixed_inference.core.registry.validate_base_binding", lambda *_args, **_kwargs: None)

    def load(path, _layers):
        if path is tmp_path:
            # use manifest identity to fail deterministically on the second call
            load.calls += 1
            if load.calls == 2:
                raise CompositionError("second facet failed")
            return type("Loaded", (), {"manifest": one})()
        raise AssertionError(path)

    load.calls = 0
    monkeypatch.setattr("mixed_inference.core.registry.load_facet", load)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
    base = LoadedBaseModel(
        request=BaseModelRequest("base"), model=model, tokenizer=None,
        stack=LayerStackHandle("layers", model, "layers", model.layers),
        binding=BaseBinding("base", None, "config", "layers", 1), device=torch.device("cpu"),
    )
    with pytest.raises(CompositionError, match="second facet failed"):
        registry.load_resolved(["one", "two"], base)
    assert registry._loaded == {}


def test_base_binding_includes_artifact_and_tokenizer_identity() -> None:
    actual = BaseBinding("base", "r1", "config", "layers", 1, artifact_sha256="artifact-a", tokenizer_sha256="tokenizer-a")
    expected = BaseBinding("base", "r1", "config", "layers", 1, artifact_sha256="artifact-b", tokenizer_sha256="tokenizer-b")
    with pytest.raises(CompatibilityError) as error:
        validate_base_binding(actual, expected)
    assert "base artifact hash differs" in str(error.value)
    assert "tokenizer hash differs" in str(error.value)
