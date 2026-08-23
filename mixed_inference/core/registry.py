from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import MANIFEST_NAME, LoadedFacet, load_facet
from .errors import CompatibilityError, MissingDependencyError
from .graph import CompositionPlan, CompositionResolver
from .modeling import LoadedBaseModel, validate_base_binding
from .specs import FacetManifest, FacetReference
from .util import load_json


class FacetRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._manifests: dict[tuple[str, str], tuple[FacetManifest, Path]] = {}
        self._loaded: dict[tuple[str, str], LoadedFacet] = {}
        self.refresh()

    def refresh(self) -> None:
        previous = self._manifests
        manifests: dict[tuple[str, str], tuple[FacetManifest, Path]] = {}
        if self.root.exists():
            paths = [self.root] if self.root.is_file() else self.root.rglob(MANIFEST_NAME)
            for path in paths:
                manifest_path = path if isinstance(path, Path) else Path(path)
                if manifest_path.name != MANIFEST_NAME:
                    continue
                manifest = FacetManifest.from_dict(load_json(manifest_path))
                key = (manifest.facet_id, manifest.version)
                if key in manifests:
                    raise CompatibilityError(
                        f"Duplicate facet {manifest.reference()} in {manifests[key][1]} and {manifest_path}"
                    )
                manifests[key] = (manifest, manifest_path)
        self._manifests = manifests
        for key in list(self._loaded):
            if key not in manifests or key not in previous:
                self._loaded.pop(key, None)
                continue
            old_manifest, old_path = previous[key]
            new_manifest, new_path = manifests[key]
            if old_path != new_path or old_manifest.to_dict() != new_manifest.to_dict():
                self._loaded.pop(key, None)

    def list(self) -> list[dict[str, Any]]:
        result = []
        for key in sorted(self._manifests):
            manifest, path = self._manifests[key]
            result.append(
                {
                    "id": manifest.facet_id,
                    "version": manifest.version,
                    "reference": manifest.reference(),
                    "status": manifest.status,
                    "path": str(path.parent),
                    "loaded": key in self._loaded,
                    "requires": list(manifest.dependencies.requires),
                    "slots": len(manifest.slots),
                    "blocks": sum(len(slot.blocks) for slot in manifest.slots),
                }
            )
        return result

    def get_manifest(self, reference: FacetReference | str) -> FacetManifest:
        ref = FacetReference.parse(reference) if isinstance(reference, str) else reference
        if ref.version is not None:
            item = self._manifests.get((ref.facet_id, ref.version))
            if item is None:
                raise MissingDependencyError(f"Facet not found: {ref.key()} in {self.root}")
            return item[0]
        candidates = [
            item for (facet_id, _), item in self._manifests.items() if facet_id == ref.facet_id
        ]
        if not candidates:
            raise MissingDependencyError(f"Facet not found: {ref.facet_id} in {self.root}")
        candidates.sort(key=lambda item: _version_key(item[0].version), reverse=True)
        return candidates[0][0]

    def manifest_path(self, manifest: FacetManifest) -> Path:
        try:
            return self._manifests[(manifest.facet_id, manifest.version)][1]
        except KeyError as exc:
            raise MissingDependencyError(f"Unregistered facet: {manifest.reference()}") from exc

    def resolve(self, requested: Iterable[str]) -> tuple[CompositionPlan, dict[str, FacetManifest]]:
        return CompositionResolver(self.get_manifest).resolve(requested)

    def load_resolved(
        self,
        requested: Iterable[str],
        base: LoadedBaseModel,
        *,
        allow_model_id_mismatch: bool = False,
    ) -> tuple[CompositionPlan, dict[str, LoadedFacet]]:
        plan, manifests = self.resolve(requested)
        loaded: dict[str, LoadedFacet] = {}
        for manifest in manifests.values():
            validate_base_binding(
                base.binding,
                manifest.base,
                allow_model_id_mismatch=allow_model_id_mismatch,
            )
            key = (manifest.facet_id, manifest.version)
            facet = self._loaded.get(key)
            if facet is None:
                facet = load_facet(self.manifest_path(manifest), base.stack.layers)
                self._loaded[key] = facet
            loaded[manifest.facet_id] = facet
        return plan, loaded

    def unload(self, facets: Iterable[str] | None = None, *, all_facets: bool = False) -> list[str]:
        if all_facets:
            removed = [facet.manifest.reference() for facet in self._loaded.values()]
            self._loaded.clear()
            return sorted(removed)
        requested = {FacetReference.parse(item).facet_id for item in (facets or [])}
        removed: list[str] = []
        for key, facet in list(self._loaded.items()):
            if facet.manifest.facet_id in requested:
                removed.append(facet.manifest.reference())
                self._loaded.pop(key, None)
        return sorted(removed)

    def loaded(self) -> list[str]:
        return sorted(facet.manifest.reference() for facet in self._loaded.values())


def _version_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"([0-9]+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)
