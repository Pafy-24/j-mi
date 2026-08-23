from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterable

from .errors import CompositionError, MissingDependencyError
from .specs import FacetManifest, FacetReference


@dataclass(frozen=True)
class PlannedBlock:
    facet: str
    facet_version: str
    slot_id: str
    after_layer: int
    slot_order: int
    block_index: int
    source_layer: int


@dataclass(frozen=True)
class CompositionPlan:
    requested: tuple[str, ...]
    resolved: tuple[str, ...]
    blocks: tuple[PlannedBlock, ...]

    def to_dict(self) -> dict[str, object]:
        grouped: dict[str, dict[str, object]] = {}
        for execution_index, block in enumerate(self.blocks):
            key = f"{block.after_layer}:{block.slot_order}:{block.slot_id}"
            slot = grouped.setdefault(
                key,
                {
                    "id": block.slot_id,
                    "afterLayer": block.after_layer,
                    "order": block.slot_order,
                    "executionIndex": execution_index,
                    "blocks": [],
                },
            )
            slot["blocks"].append(
                {
                    "facet": block.facet,
                    "version": block.facet_version,
                    "blockIndex": block.block_index,
                    "sourceLayer": block.source_layer,
                    "executionIndex": execution_index,
                }
            )
        slots = sorted(
            grouped.values(),
            key=lambda item: int(item["executionIndex"]),
        )
        return {
            "schemaVersion": "mi-composition-v1",
            "requestedFacets": list(self.requested),
            "resolvedFacets": list(self.resolved),
            "slots": slots,
            "expandedBlockCount": len(self.blocks),
        }


class CompositionResolver:
    """Resolve recursive dependencies and produce a deterministic facet/slot order."""

    def __init__(self, manifest_loader: Callable[[FacetReference], FacetManifest]):
        self._load = manifest_loader

    def resolve(self, requested: Iterable[str]) -> tuple[CompositionPlan, dict[str, FacetManifest]]:
        requested_refs = tuple(FacetReference.parse(item) for item in requested)
        manifests: dict[str, FacetManifest] = {}
        requested_order: dict[str, int] = {}

        def visit(reference: FacetReference, chain: tuple[str, ...]) -> FacetManifest:
            manifest = self._load(reference)
            key = manifest.reference()
            previous = next(
                (item for item in manifests.values() if item.facet_id == manifest.facet_id),
                None,
            )
            if previous is not None and previous.version != manifest.version:
                raise CompositionError(
                    f"Composition requests two versions of facet {manifest.facet_id}: "
                    f"{previous.version} and {manifest.version}"
                )
            if key in chain:
                cycle = " -> ".join((*chain, key))
                raise CompositionError(f"Facet dependency cycle: {cycle}")
            if key in manifests:
                return manifests[key]
            manifests[key] = manifest
            for dependency in manifest.dependencies.requires:
                visit(FacetReference.parse(dependency), (*chain, key))
            return manifest

        for index, reference in enumerate(requested_refs):
            manifest = visit(reference, ())
            requested_order.setdefault(manifest.reference(), index)

        if not manifests and requested_refs:
            raise MissingDependencyError("No requested facet manifests were resolved")

        by_id = {manifest.facet_id: manifest for manifest in manifests.values()}
        self._validate_conflicts(by_id)
        ordered = self._topological_order(manifests, by_id, requested_order)
        self._validate_slot_coordinates(ordered)

        planned_blocks: list[PlannedBlock] = []
        for manifest in ordered:
            for slot in sorted(manifest.slots, key=lambda value: (value.after_layer, value.order, value.slot_id)):
                for block_index, block in enumerate(slot.blocks):
                    planned_blocks.append(
                        PlannedBlock(
                            facet=manifest.facet_id,
                            facet_version=manifest.version,
                            slot_id=slot.slot_id,
                            after_layer=slot.after_layer,
                            slot_order=slot.order,
                            block_index=block_index,
                            source_layer=block.source_layer,
                        )
                    )
        order_index = {manifest.facet_id: index for index, manifest in enumerate(ordered)}
        planned_blocks.sort(
            key=lambda block: (
                block.after_layer,
                order_index[block.facet],
                block.slot_order,
                block.slot_id,
                block.block_index,
            )
        )
        plan = CompositionPlan(
            requested=tuple(reference.key() for reference in requested_refs),
            resolved=tuple(manifest.reference() for manifest in ordered),
            blocks=tuple(planned_blocks),
        )
        return plan, manifests

    @staticmethod
    def _validate_conflicts(by_id: dict[str, FacetManifest]) -> None:
        active = set(by_id)
        for manifest in by_id.values():
            conflicts = {FacetReference.parse(item).facet_id for item in manifest.dependencies.conflicts}
            overlap = sorted(active & conflicts)
            if overlap:
                raise CompositionError(
                    f"Facet {manifest.facet_id} conflicts with active facet(s): {', '.join(overlap)}"
                )

    @staticmethod
    def _topological_order(
        manifests: dict[str, FacetManifest],
        by_id: dict[str, FacetManifest],
        requested_order: dict[str, int],
    ) -> list[FacetManifest]:
        edges: dict[str, set[str]] = {manifest.facet_id: set() for manifest in manifests.values()}
        indegree = {facet_id: 0 for facet_id in edges}

        def add_edge(before: str, after: str) -> None:
            if before == after:
                raise CompositionError(f"Facet ordering contains a self-edge: {before}")
            if before not in edges or after not in edges:
                return
            if after not in edges[before]:
                edges[before].add(after)
                indegree[after] += 1

        for manifest in manifests.values():
            for item in manifest.dependencies.requires:
                add_edge(FacetReference.parse(item).facet_id, manifest.facet_id)
            for item in manifest.dependencies.after:
                add_edge(FacetReference.parse(item).facet_id, manifest.facet_id)
            for item in manifest.dependencies.before:
                add_edge(manifest.facet_id, FacetReference.parse(item).facet_id)

        default_request_rank = len(requested_order) + len(manifests)

        def rank(facet_id: str) -> tuple[int, int, str]:
            manifest = by_id[facet_id]
            return (
                manifest.dependencies.priority,
                requested_order.get(manifest.reference(), default_request_rank),
                facet_id,
            )

        heap: list[tuple[tuple[int, int, str], str]] = []
        for facet_id, degree in indegree.items():
            if degree == 0:
                heapq.heappush(heap, (rank(facet_id), facet_id))
        ordered: list[FacetManifest] = []
        while heap:
            _, facet_id = heapq.heappop(heap)
            ordered.append(by_id[facet_id])
            for child in sorted(edges[facet_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(heap, (rank(child), child))
        if len(ordered) != len(manifests):
            remaining = sorted(facet_id for facet_id, degree in indegree.items() if degree > 0)
            raise CompositionError(f"Facet ordering cycle among: {', '.join(remaining)}")
        return ordered

    @staticmethod
    def _validate_slot_coordinates(manifests: Iterable[FacetManifest]) -> None:
        coordinates: dict[str, tuple[int, int]] = {}
        for manifest in manifests:
            for slot in manifest.slots:
                coordinate = (slot.after_layer, slot.order)
                previous = coordinates.setdefault(slot.slot_id, coordinate)
                if previous != coordinate:
                    raise CompositionError(
                        f"Slot {slot.slot_id!r} has incompatible coordinates: "
                        f"{previous} versus {coordinate} in facet {manifest.facet_id}"
                    )
