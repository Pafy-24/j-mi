"""Deterministic, CPU-testable residency admission for MI facet caches.

This module accounts for declared bytes.  It deliberately does not infer VRAM
availability or launch devices; RM remains the authority for resource leases.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Hashable, Iterable

from .errors import CompositionError


@dataclass(frozen=True)
class ResidencyBudget:
    ram_bytes: int | None = None
    accelerator_bytes: int | None = None
    operational_margin_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("ram_bytes", self.ram_bytes),
            ("accelerator_bytes", self.accelerator_bytes),
            ("operational_margin_bytes", self.operational_margin_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Footprint:
    ram_bytes: int = 0
    accelerator_bytes: int = 0

    def __post_init__(self) -> None:
        if self.ram_bytes < 0 or self.accelerator_bytes < 0:
            raise ValueError("residency footprint bytes must be non-negative")

    def __add__(self, other: "Footprint") -> "Footprint":
        return Footprint(
            ram_bytes=self.ram_bytes + other.ram_bytes,
            accelerator_bytes=self.accelerator_bytes + other.accelerator_bytes,
        )


@dataclass(frozen=True)
class CacheIdentity:
    """Semantic cache identity; do not reduce this to a facet name/version."""

    base_artifact_sha256: str
    base_config_sha256: str
    tokenizer_sha256: str
    dtype: str
    device: str
    request_config_sha256: str

    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.base_artifact_sha256,
            self.base_config_sha256,
            self.tokenizer_sha256,
            self.dtype,
            self.device,
            self.request_config_sha256,
        )


@dataclass
class _Entry:
    footprint: Footprint
    last_used: float
    pins: int = 0


@dataclass(frozen=True)
class Admission:
    evicted: tuple[Hashable, ...]
    pinned: tuple[Hashable, ...]
    created: tuple[Hashable, ...] = ()


@dataclass(frozen=True)
class AdmissionPlan:
    """Immutable admission decision tied to one residency snapshot."""

    owner: object
    snapshot_version: int
    requested: tuple[tuple[Hashable, Footprint], ...]
    staging: tuple[tuple[Hashable, Footprint], ...]
    kv_cache: Footprint
    evicted: tuple[Hashable, ...]


@dataclass(frozen=True)
class AppliedAdmissionPlan:
    plan: AdmissionPlan
    applied_version: int
    reservation: object


class ResidencyManager:
    """Pinned LRU cache admission with all-or-nothing state transitions."""

    def __init__(self, budget: ResidencyBudget | None = None) -> None:
        self.budget = budget or ResidencyBudget()
        self._base = Footprint()
        self._entries: dict[Hashable, _Entry] = {}
        self._version = 0
        self._owner = object()
        self._active_reservation: object | None = None
        self._lock = RLock()

    def reset(self, *, base: Footprint = Footprint()) -> None:
        with self._lock:
            proposed = base + self._margin()
            self._ensure_within_budget(proposed)
            self._base = base
            self._entries.clear()
            self._active_reservation = None
            self._version += 1

    def preflight_staging(self, staged: Iterable[Footprint], *, kv_cache: Footprint = Footprint()) -> None:
        """Fail closed if loading new bytes could exceed the declared peak.

        Existing cache entries stay resident until all facet files load, so
        they are deliberately included here rather than assumed evictable.
        """
        with self._lock:
            total = self._total(self._entries) + kv_cache + self._margin()
            for footprint in staged:
                total = total + footprint
            self._ensure_within_budget(total)

    def prepare_admission(
        self,
        requested: dict[Hashable, Footprint],
        *,
        staging: dict[Hashable, Footprint] | None = None,
        kv_cache: Footprint = Footprint(),
    ) -> AdmissionPlan:
        """Plan final residency and pre-allocation peak without mutating state.

        ``staging`` contains the source-plus-target peak for missing requested
        entries.  Existing request hits remain resident and are protected.
        Inactive entries may be selected for eviction before staging begins.
        """
        with self._lock:
            staged = dict(staging or {})
            unknown = set(staged) - set(requested)
            if unknown:
                raise ValueError("staging entries must be part of the requested set")
            projected = {
                key: _Entry(entry.footprint, entry.last_used, entry.pins)
                for key, entry in self._entries.items()
            }
            missing: dict[Hashable, Footprint] = {}
            for key, footprint in requested.items():
                entry = projected.get(key)
                if entry is None:
                    missing[key] = footprint
                elif entry.footprint != footprint:
                    raise CompositionError(
                        "composition_residency_denied: resident facet footprint changed under the same cache identity"
                    )
            for key in staged:
                if key in projected:
                    raise ValueError("staging footprint supplied for an existing cache entry")
                final = requested[key]
                peak = staged[key]
                if (
                    peak.ram_bytes < final.ram_bytes
                    or peak.accelerator_bytes < final.accelerator_bytes
                ):
                    raise ValueError(
                        "staging footprint must include the requested resident footprint"
                    )

            protected = set(requested)
            evicted: list[Hashable] = []
            while not self._plan_fits(projected, missing, staged, kv_cache):
                candidates = [
                    (entry.last_used, str(key), key)
                    for key, entry in projected.items()
                    if key not in protected and entry.pins == 0
                ]
                if not candidates:
                    raise CompositionError(
                        "composition_residency_denied: base, requested or pinned facets, staging peak, KV cache and operational margin exceed budget"
                    )
                _, _, key = min(candidates)
                projected.pop(key)
                evicted.append(key)
            return AdmissionPlan(
                owner=self._owner,
                snapshot_version=self._version,
                requested=tuple(requested.items()),
                staging=tuple(staged.items()),
                kv_cache=kv_cache,
                evicted=tuple(evicted),
            )

    def apply_admission(self, plan: AdmissionPlan) -> AppliedAdmissionPlan:
        """Apply only the planned inactive evictions after a version check."""
        with self._lock:
            self._require_owner(plan)
            self._require_version(plan.snapshot_version)
            if self._active_reservation is not None:
                raise CompositionError(
                    "composition_residency_busy: another admission is being staged"
                )
            for key in plan.evicted:
                entry = self._entries.get(key)
                if entry is None or entry.pins or key in dict(plan.requested):
                    raise CompositionError(
                        "composition_residency_stale: planned eviction no longer matches residency state"
                    )
            for key in plan.evicted:
                self._entries.pop(key)
            reservation = object()
            self._active_reservation = reservation
            # Applying a plan is itself a state transition even without an
            # eviction: it exclusively reserves the staging window.
            self._version += 1
            return AppliedAdmissionPlan(
                plan=plan,
                applied_version=self._version,
                reservation=reservation,
            )

    def commit_admission(
        self,
        applied: AppliedAdmissionPlan,
        actual: dict[Hashable, Footprint],
    ) -> Admission:
        """Pin a fully staged request after reconciling actual tensor bytes."""
        with self._lock:
            self._require_owner(applied.plan)
            if self._active_reservation is not applied.reservation:
                raise CompositionError(
                    "composition_residency_stale: applied admission is no longer reserved"
                )
            try:
                self._require_version(applied.applied_version)
                expected = dict(applied.plan.requested)
                if set(actual) != set(expected):
                    raise CompositionError(
                        "composition_residency_denied: staged facets differ from the prepared request"
                    )
                for key, footprint in actual.items():
                    estimate = expected[key]
                    if (
                        footprint.ram_bytes > estimate.ram_bytes
                        or footprint.accelerator_bytes > estimate.accelerator_bytes
                    ):
                        raise CompositionError(
                            "composition_residency_denied: actual facet footprint exceeds the pre-allocation bound"
                        )

                projected = {
                    key: _Entry(entry.footprint, entry.last_used, entry.pins)
                    for key, entry in self._entries.items()
                }
                now = monotonic()
                created: list[Hashable] = []
                for key, footprint in actual.items():
                    entry = projected.get(key)
                    if entry is None:
                        projected[key] = _Entry(footprint=footprint, last_used=now)
                        created.append(key)
                    elif entry.footprint != footprint:
                        raise CompositionError(
                            "composition_residency_denied: resident facet footprint changed during staging"
                        )
                if not self._fits(
                    self._total(projected) + applied.plan.kv_cache + self._margin()
                ):
                    raise CompositionError(
                        "composition_residency_denied: reconciled facets, KV cache and operational margin exceed budget"
                    )
                self._entries = projected
                for key in actual:
                    entry = self._entries[key]
                    entry.last_used = now
                    entry.pins += 1
                return Admission(
                    evicted=applied.plan.evicted,
                    pinned=tuple(actual),
                    created=tuple(created),
                )
            finally:
                self._active_reservation = None
                self._version += 1

    def cancel_admission(self, applied: AppliedAdmissionPlan) -> None:
        """Release a staging reservation after load/reconciliation failure."""
        with self._lock:
            self._require_owner(applied.plan)
            if self._active_reservation is applied.reservation:
                self._active_reservation = None
                self._version += 1
                return
            if self._active_reservation is None:
                return
            raise CompositionError(
                "composition_residency_stale: another admission owns the staging reservation"
            )

    def admit(
        self,
        requested: dict[Hashable, Footprint],
        *,
        kv_cache: Footprint = Footprint(),
    ) -> Admission:
        """Compatibility helper for callers without a separate staging phase."""
        plan = self.prepare_admission(requested, kv_cache=kv_cache)
        applied = self.apply_admission(plan)
        return self.commit_admission(applied, requested)

    def release(self, admission: Admission) -> None:
        with self._lock:
            now = monotonic()
            changed = False
            for key in admission.pinned:
                entry = self._entries.get(key)
                if entry is not None:
                    entry.pins = max(0, entry.pins - 1)
                    entry.last_used = now
                    changed = True
            if changed:
                self._version += 1

    def rollback(self, admission: Admission) -> None:
        """Undo a failed registry commit without restoring prior evictions."""
        with self._lock:
            created = set(admission.created)
            now = monotonic()
            for key in admission.pinned:
                if key in created:
                    self._entries.pop(key, None)
                    continue
                entry = self._entries.get(key)
                if entry is not None:
                    entry.pins = max(0, entry.pins - 1)
                    entry.last_used = now
            self._version += 1

    def discard(self, keys: Iterable[Hashable]) -> None:
        """Remove cache entries whose backing facet was invalidated or unloaded.

        Runtime calls are serialized, so an invalidated entry cannot still be
        pinned by an in-flight generation when this is invoked.
        """
        with self._lock:
            changed = False
            for key in keys:
                changed = self._entries.pop(key, None) is not None or changed
            if changed:
                self._version += 1

    def snapshot(self) -> dict[str, int | None]:
        with self._lock:
            total = self._total(self._entries)
            return {
                "ramBytes": total.ram_bytes,
                "acceleratorBytes": total.accelerator_bytes,
                "entries": len(self._entries),
                "pinnedEntries": sum(1 for entry in self._entries.values() if entry.pins),
                "ramBudgetBytes": self.budget.ram_bytes,
                "acceleratorBudgetBytes": self.budget.accelerator_bytes,
            }

    def _plan_fits(
        self,
        retained: dict[Hashable, _Entry],
        missing: dict[Hashable, Footprint],
        staged: dict[Hashable, Footprint],
        kv_cache: Footprint,
    ) -> bool:
        common = self._total(retained) + kv_cache + self._margin()
        final = common
        peak = common
        for key, footprint in missing.items():
            final = final + footprint
            peak = peak + staged.get(key, footprint)
        return self._fits(final) and self._fits(peak)

    def _require_version(self, expected: int) -> None:
        if self._version != expected:
            raise CompositionError(
                "composition_residency_stale: admission plan no longer matches residency state"
            )

    def _require_owner(self, plan: AdmissionPlan) -> None:
        if plan.owner is not self._owner:
            raise CompositionError(
                "composition_residency_stale: admission plan belongs to another residency manager"
            )

    def _total(self, entries: dict[Hashable, _Entry]) -> Footprint:
        total = self._base
        for entry in entries.values():
            total = total + entry.footprint
        return total

    def _margin(self) -> Footprint:
        return Footprint(
            ram_bytes=self.budget.operational_margin_bytes,
            accelerator_bytes=self.budget.operational_margin_bytes,
        )

    def _fits(self, footprint: Footprint) -> bool:
        return (
            (self.budget.ram_bytes is None or footprint.ram_bytes <= self.budget.ram_bytes)
            and (
                self.budget.accelerator_bytes is None
                or footprint.accelerator_bytes <= self.budget.accelerator_bytes
            )
        )

    def _ensure_within_budget(self, footprint: Footprint) -> None:
        if not self._fits(footprint):
            raise CompositionError("base_residency_denied: base and operational margin exceed budget")
