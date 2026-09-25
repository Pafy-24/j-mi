from __future__ import annotations

from types import SimpleNamespace

import pytest

from mixed_inference.core.errors import ConfigurationError
from mixed_inference.core.residency import Footprint, ResidencyBudget, ResidencyManager
from mixed_inference.generate.runtime import MixedInferenceRuntime
from mixed_inference.rm_integration import (
    facet_staging_allocation,
    residency_budget,
)


def _lease(root, run, *, ram_bytes=4096, quota_bytes=4096):
    return SimpleNamespace(
        lease={
            "state": "active",
            "paths": {"root": str(root), "run": str(run)},
        },
        bindings=[
            {
                "resourceId": "memory",
                "runtime": {"ramBytes": ram_bytes, "swapBytes": 0},
                "enforcement": {"memory": "hard"},
            },
            {
                "resourceId": "storage",
                "runtime": {"rootPath": str(root.parent), "quotaBytes": quota_bytes},
                "enforcement": {"quota": "reservation"},
            },
        ],
        accelerators=[],
    )


def _activate(monkeypatch, lease) -> None:
    import jester_rm_client

    monkeypatch.setattr(
        jester_rm_client,
        "current_lease",
        lambda *, required=None, refresh=False: lease,
    )


def test_managed_staging_counts_whole_instance_root_and_cleans_only_own_request(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    logs = root / "logs"
    run.mkdir(parents=True)
    logs.mkdir()
    (logs / "existing.log").write_bytes(b"x" * 80)
    _activate(monkeypatch, _lease(root, run, quota_bytes=100))

    allocation = facet_staging_allocation()
    assert allocation.usage_root == root.resolve()
    assert allocation.staging_root == run.resolve() / "mi-runtime" / "facet-staging"
    assert allocation.managed is True
    assert allocation.quota_enforcement == "reservation"
    with pytest.raises(ConfigurationError, match=r"80\+21>100"):
        allocation.create_request(21)

    request = allocation.create_request(20)
    owned = request.directory / "owned.bin"
    owned.write_bytes(b"y" * 20)
    owned.unlink()
    request.close()
    assert not request.directory.exists()
    assert (logs / "existing.log").read_bytes() == b"x" * 80


@pytest.mark.parametrize(
    "paths",
    (
        {"run": "present"},
        {"root": "present"},
        {},
    ),
)
def test_managed_staging_refuses_missing_explicit_root_or_run(
    tmp_path, monkeypatch, paths
) -> None:
    expanded = {
        name: str(tmp_path / value)
        for name, value in paths.items()
    }
    lease = _lease(tmp_path, tmp_path)
    lease.lease = {"state": "active", "paths": expanded}
    _activate(monkeypatch, lease)
    with pytest.raises(ConfigurationError, match="requires explicit paths.root and paths.run"):
        facet_staging_allocation()


def test_managed_staging_refuses_run_outside_instance_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "instance"
    run = tmp_path / "outside-run"
    root.mkdir()
    run.mkdir()
    _activate(monkeypatch, _lease(root, run))
    with pytest.raises(ConfigurationError, match="escapes owned instance root"):
        facet_staging_allocation()


def test_managed_staging_refuses_relative_lease_paths(tmp_path, monkeypatch) -> None:
    lease = _lease(tmp_path, tmp_path)
    lease.lease["paths"] = {"root": "relative-root", "run": "relative-root/run"}
    _activate(monkeypatch, lease)
    with pytest.raises(ConfigurationError, match="must be absolute"):
        facet_staging_allocation()


def test_managed_operations_require_active_lease(tmp_path, monkeypatch) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    lease = _lease(root, run)
    lease.lease["state"] = "reserved"
    _activate(monkeypatch, lease)
    with pytest.raises(ConfigurationError, match="active RM lease"):
        facet_staging_allocation()
    with pytest.raises(ConfigurationError, match="active RM lease"):
        residency_budget()


def test_staging_usage_scan_fails_closed_on_unreadable_subtree(
    tmp_path, monkeypatch
) -> None:
    from mixed_inference import rm_integration

    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    _activate(monkeypatch, _lease(root, run))

    def denied_walk(_root, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError("denied subtree"))
        yield ()

    monkeypatch.setattr(rm_integration.os, "walk", denied_walk)
    with pytest.raises(ConfigurationError, match="cannot account instance storage"):
        facet_staging_allocation().create_request(1)


def test_managed_ram_budget_comes_from_lease_and_env_can_only_narrow(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    _activate(monkeypatch, _lease(root, run, ram_bytes=4096))

    assert residency_budget()["ramBytes"] == 4096
    monkeypatch.setenv("JESTER_MI_RAM_BUDGET_BYTES", "3072")
    assert residency_budget()["ramBytes"] == 3072
    monkeypatch.setenv("JESTER_MI_RAM_BUDGET_BYTES", "4097")
    with pytest.raises(ConfigurationError, match="cannot exceed"):
        residency_budget()


@pytest.mark.parametrize("ram_bytes", (None, True, 0, -1, "4096"))
def test_managed_ram_budget_refuses_missing_or_malformed_binding(
    tmp_path, monkeypatch, ram_bytes
) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    lease = _lease(root, run)
    if ram_bytes is None:
        lease.bindings = [lease.bindings[1]]
    else:
        lease.bindings[0]["runtime"]["ramBytes"] = ram_bytes
    _activate(monkeypatch, lease)
    with pytest.raises(ConfigurationError, match="memory binding|positive integer"):
        residency_budget()


@pytest.mark.parametrize("ram_bytes", (None, 4097))
def test_managed_runtime_refuses_injected_manager_that_bypasses_ram_lease(
    tmp_path, monkeypatch, ram_bytes
) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    _activate(monkeypatch, _lease(root, run, ram_bytes=4096))
    manager = ResidencyManager(ResidencyBudget(ram_bytes=ram_bytes))

    with pytest.raises(ConfigurationError, match="injected residency manager"):
        MixedInferenceRuntime(tmp_path / "facets", residency=manager)


def test_managed_runtime_preserves_narrow_populated_residency_manager(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    _activate(monkeypatch, _lease(root, run, ram_bytes=4096))
    manager = ResidencyManager(ResidencyBudget(ram_bytes=2048))
    manager.reset(base=Footprint(ram_bytes=128))
    admission = manager.admit({"existing": Footprint(ram_bytes=64)})
    manager.release(admission)
    before = manager.snapshot()

    runtime = MixedInferenceRuntime(tmp_path / "facets", residency=manager)

    assert runtime.residency is manager
    assert manager.snapshot() == before


def test_managed_runtime_refuses_unbounded_accelerator_override(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "instance"
    run = root / "run"
    run.mkdir(parents=True)
    lease = _lease(root, run, ram_bytes=4096)
    lease.accelerators = [
        {"runtime": {"backend": "torch.cuda", "memoryBudgetBytes": 1024}}
    ]
    _activate(monkeypatch, lease)
    manager = ResidencyManager(ResidencyBudget(ram_bytes=2048))

    with pytest.raises(ConfigurationError, match="accelerator_bytes"):
        MixedInferenceRuntime(tmp_path / "facets", residency=manager)


def test_unmanaged_staging_requires_explicit_path_and_positive_limit(
    tmp_path, monkeypatch
) -> None:
    _activate(monkeypatch, None)
    with pytest.raises(ConfigurationError, match="explicit staging directory"):
        facet_staging_allocation()
    with pytest.raises(ConfigurationError, match="positive integer"):
        facet_staging_allocation(local_root=tmp_path / "stage", local_limit_bytes=True)

    allocation = facet_staging_allocation(
        local_root=tmp_path / "stage", local_limit_bytes=10
    )
    request = allocation.create_request(10)
    request.close()
    request.close()
    assert allocation.managed is False
