"""Deterministic, CPU-testable residency admission for MI facet caches.

This module accounts for declared bytes.  It deliberately does not infer VRAM
availability or launch devices; RM remains the authority for resource leases.
"""
from __future__ import annotations

from dataclasses import dataclass
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


class ResidencyManager:
    """Pinned LRU cache admission with all-or-nothing state transitions."""

    def __init__(self, budget: ResidencyBudget | None = None) -> None:
        self.budget = budget or ResidencyBudget()
        self._base = Footprint()
        self._entries: dict[Hashable, _Entry] = {}

    def reset(self, *, base: Footprint = Footprint()) -> None:
        proposed = base + Footprint(
            ram_bytes=self.budget.operational_margin_bytes,
            accelerator_bytes=self.budget.operational_margin_bytes,
        )
        self._ensure_within_budget(proposed)
        self._base = base
        self._entries.clear()

    def preflight_staging(self, staged: Iterable[Footprint], *, kv_cache: Footprint = Footprint()) -> None:
        """Fail closed if loading new bytes could exceed the declared peak.

        Existing cache entries stay resident until all facet files load, so
        they are deliberately included here rather than assumed evictable.
        """
        total = self._total(self._entries) + kv_cache + self._margin()
        for footprint in staged:
            total = total + footprint
        self._ensure_within_budget(total)

    def admit(
        self,
        requested: dict[Hashable, Footprint],
        *,
        kv_cache: Footprint = Footprint(),
    ) -> Admission:
        """Pin requested entries or fail without changing cache state."""
        requested_keys = tuple(requested)
        projected = dict(self._entries)
        now = monotonic()
        for key, footprint in requested.items():
            entry = projected.get(key)
            if entry is None:
                projected[key] = _Entry(footprint=footprint, last_used=now)
            elif entry.footprint != footprint:
                # A manifest/dtype change under an old key is never reusable.
                projected[key] = _Entry(footprint=footprint, last_used=now, pins=entry.pins)

        protected = set(requested_keys)
        evicted: list[Hashable] = []
        while not self._fits(self._total(projected) + kv_cache + self._margin()):
            candidates = [
                (entry.last_used, str(key), key)
                for key, entry in projected.items()
                if key not in protected and entry.pins == 0
            ]
            if not candidates:
                raise CompositionError(
                    "composition_residency_denied: base, pinned facets, KV cache and operational margin exceed budget"
                )
            _, _, key = min(candidates)
            projected.pop(key)
            evicted.append(key)

        # Commit only after full admission succeeded.
        self._entries = projected
        for key in requested_keys:
            entry = self._entries[key]
            entry.last_used = now
            entry.pins += 1
        return Admission(evicted=tuple(evicted), pinned=requested_keys)

    def release(self, admission: Admission) -> None:
        now = monotonic()
        for key in admission.pinned:
            entry = self._entries.get(key)
            if entry is not None:
                entry.pins = max(0, entry.pins - 1)
                entry.last_used = now

    def discard(self, keys: Iterable[Hashable]) -> None:
        """Remove cache entries whose backing facet was invalidated or unloaded.

        Runtime calls are serialized, so an invalidated entry cannot still be
        pinned by an in-flight generation when this is invoked.
        """
        for key in keys:
            self._entries.pop(key, None)

    def snapshot(self) -> dict[str, int | None]:
        total = self._total(self._entries)
        return {
            "ramBytes": total.ram_bytes,
            "acceleratorBytes": total.accelerator_bytes,
            "entries": len(self._entries),
            "pinnedEntries": sum(1 for entry in self._entries.values() if entry.pins),
            "ramBudgetBytes": self.budget.ram_bytes,
            "acceleratorBudgetBytes": self.budget.accelerator_bytes,
        }

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
