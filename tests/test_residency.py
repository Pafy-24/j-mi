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
    monkeypatch.setattr(registry, "validate_resolved", lambda *_args, **_kwargs: None)

    def load(path, _layers, **_kwargs):
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


def test_staging_plan_evicts_inactive_before_allocation_and_is_version_fenced() -> None:
    manager = ResidencyManager(ResidencyBudget(ram_bytes=70, operational_margin_bytes=10))
    manager.reset(base=Footprint(ram_bytes=20))
    old = manager.admit({"old": Footprint(ram_bytes=30)})
    manager.release(old)

    plan = manager.prepare_admission(
        {"new": Footprint(ram_bytes=30)},
        staging={"new": Footprint(ram_bytes=40)},
    )
    assert plan.evicted == ("old",)
    applied = manager.apply_admission(plan)
    admission = manager.commit_admission(applied, {"new": Footprint(ram_bytes=30)})
    assert admission.evicted == ("old",)
    manager.release(admission)
    assert manager.snapshot()["entries"] == 1

    stale = manager.prepare_admission({"next": Footprint(ram_bytes=20)})
    touched = manager.admit({"new": Footprint(ram_bytes=30)})
    manager.release(touched)
    before = manager.snapshot()
    with pytest.raises(CompositionError, match="composition_residency_stale"):
        manager.apply_admission(stale)
    assert manager.snapshot() == before

    commit_manager = ResidencyManager(ResidencyBudget(ram_bytes=100))
    commit_manager.reset(base=Footprint(ram_bytes=10))
    held = commit_manager.admit({"held": Footprint(ram_bytes=20)})
    commit_plan = commit_manager.prepare_admission({"request": Footprint(ram_bytes=20)})
    commit_applied = commit_manager.apply_admission(commit_plan)
    commit_manager.release(held)
    before_commit = commit_manager.snapshot()
    with pytest.raises(CompositionError, match="composition_residency_stale"):
        commit_manager.commit_admission(
            commit_applied, {"request": Footprint(ram_bytes=20)}
        )
    assert commit_manager.snapshot() == before_commit


def test_registry_commit_rollback_preserves_preexisting_hit() -> None:
    manager = ResidencyManager(ResidencyBudget(ram_bytes=100))
    manager.reset(base=Footprint(ram_bytes=10))
    hit = manager.admit({"hit": Footprint(ram_bytes=20)})
    manager.release(hit)
    prepared = manager.prepare_admission(
        {"hit": Footprint(ram_bytes=20), "miss": Footprint(ram_bytes=30)},
        staging={"miss": Footprint(ram_bytes=40)},
    )
    applied = manager.apply_admission(prepared)
    request = manager.commit_admission(
        applied,
        {"hit": Footprint(ram_bytes=20), "miss": Footprint(ram_bytes=30)},
    )
    manager.rollback(request)
    snapshot = manager.snapshot()
    assert snapshot["entries"] == 1
    assert snapshot["pinnedEntries"] == 0
    assert snapshot["ramBytes"] == 30


def test_admission_plan_cannot_cross_residency_manager_boundary() -> None:
    first = ResidencyManager(ResidencyBudget(ram_bytes=100))
    second = ResidencyManager(ResidencyBudget(ram_bytes=100))
    first.reset(base=Footprint(ram_bytes=10))
    second.reset(base=Footprint(ram_bytes=10))
    foreign = first.prepare_admission({"facet": Footprint(ram_bytes=20)})
    before = second.snapshot()
    with pytest.raises(CompositionError, match="another residency manager"):
        second.apply_admission(foreign)
    assert second.snapshot() == before

    active_plan = first.prepare_admission({"first": Footprint(ram_bytes=20)})
    active = first.apply_admission(active_plan)
    overlapping = first.prepare_admission({"second": Footprint(ram_bytes=20)})
    with pytest.raises(CompositionError, match="composition_residency_busy"):
        first.apply_admission(overlapping)
    first.cancel_admission(active)


def test_base_binding_includes_artifact_and_tokenizer_identity() -> None:
    actual = BaseBinding("base", "r1", "config", "layers", 1, artifact_sha256="artifact-a", tokenizer_sha256="tokenizer-a")
    expected = BaseBinding("base", "r1", "config", "layers", 1, artifact_sha256="artifact-b", tokenizer_sha256="tokenizer-b")
    with pytest.raises(CompatibilityError) as error:
        validate_base_binding(actual, expected)
    assert "base artifact hash differs" in str(error.value)
    assert "tokenizer hash differs" in str(error.value)
