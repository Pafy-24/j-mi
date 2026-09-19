from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable

from .checkpoint import MANIFEST_NAME, LoadedFacet, load_facet
from .errors import CompatibilityError, MissingDependencyError
from .graph import CompositionPlan, CompositionResolver
from .modeling import LoadedBaseModel, validate_base_binding
from .specs import FacetManifest, FacetReference
from .util import load_json, sha256_json


class FacetRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._manifests: dict[tuple[str, str], tuple[FacetManifest, Path]] = {}
        # A facet's bytes are initialized against a concrete base stack.  Never
        # reuse by name/version alone across base artifact/config/tokenizer,
        # dtype, device, request configuration, or manifest changes.
        self._loaded: dict[tuple[tuple[str, ...], str, str, str], LoadedFacet] = {}
        self.refresh()

    def refresh(self) -> tuple[tuple[tuple[str, ...], str, str, str], ...]:
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
        invalidated: list[tuple[tuple[str, ...], str, str, str]] = []
        for key in list(self._loaded):
            _, facet_id, version, manifest_digest = key
            manifest_key = (facet_id, version)
            if manifest_key not in manifests or manifest_key not in previous:
                self._loaded.pop(key, None)
                invalidated.append(key)
                continue
            old_manifest, old_path = previous[manifest_key]
            new_manifest, new_path = manifests[manifest_key]
            if (
                old_path != new_path
                or old_manifest.to_dict() != new_manifest.to_dict()
                or manifest_digest != _manifest_digest(new_manifest)
            ):
                self._loaded.pop(key, None)
                invalidated.append(key)
        return tuple(invalidated)

    def list(self) -> list[dict[str, Any]]:
        result = []
        for key in sorted(self._manifests):
            manifest, path = self._manifests[key]
            item = {
                "id": manifest.facet_id,
                "version": manifest.version,
                "reference": manifest.reference(),
                "status": manifest.status,
                "path": str(path.parent),
                "loaded": any(cache_key[1:3] == key for cache_key in self._loaded),
                "requires": list(manifest.dependencies.requires),
                "slots": len(manifest.slots),
                "blocks": sum(len(slot.blocks) for slot in manifest.slots),
            }
            if manifest.artifact is not None:
                item["artifact"] = manifest.artifact.to_dict()
            result.append(item)
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
        allow_unsafe_legacy_base_binding: bool = False,
        before_commit: Callable[[dict[str, LoadedFacet]], None] | None = None,
        resolved: tuple[CompositionPlan, dict[str, FacetManifest]] | None = None,
    ) -> tuple[CompositionPlan, dict[str, LoadedFacet]]:
        if allow_model_id_mismatch and not allow_unsafe_legacy_base_binding:
            raise CompatibilityError(
                "allow_model_id_mismatch is an unsafe legacy compatibility bypass; "
                "set allow_unsafe_legacy_base_binding explicitly"
            )
        plan, manifests = resolved or self.resolve(requested)
        loaded: dict[str, LoadedFacet] = {}
        staged: dict[tuple[tuple[str, ...], str, str, str], LoadedFacet] = {}
        # Validate and load every missing dependency before mutating the cache.
        # If a later artifact fails, a previous facet cannot leak into residency.
        for manifest in manifests.values():
            validate_base_binding(
                base.binding,
                manifest.base,
                allow_model_id_mismatch=allow_model_id_mismatch,
            )
            key = _cache_key(base, manifest)
            facet = self._loaded.get(key)
            if facet is None:
                facet = load_facet(self.manifest_path(manifest), base.stack.layers)
                staged[key] = facet
            loaded[manifest.facet_id] = facet
        if before_commit is not None:
            before_commit(loaded)
        self._loaded.update(staged)
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

    def cache_keys_for(
        self, facets: Iterable[str] | None = None, *, all_facets: bool = False
    ) -> tuple[tuple[tuple[str, ...], str, str, str], ...]:
        if all_facets:
            return tuple(self._loaded)
        requested = {FacetReference.parse(item).facet_id for item in (facets or [])}
        return tuple(
            key for key, facet in self._loaded.items()
            if facet.manifest.facet_id in requested
        )

    def loaded(self) -> list[str]:
        return sorted({facet.manifest.reference() for facet in self._loaded.values()})

    def cache_key(self, base: LoadedBaseModel, facet: LoadedFacet) -> tuple[tuple[str, ...], str, str, str]:
        return _cache_key(base, facet.manifest)

    def is_cached(self, base: LoadedBaseModel, manifest: FacetManifest) -> bool:
        return _cache_key(base, manifest) in self._loaded

    def evict_cache_keys(self, keys: Iterable[object]) -> None:
        for key in keys:
            if isinstance(key, tuple):
                self._loaded.pop(key, None)


def _version_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"([0-9]+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _manifest_digest(manifest: FacetManifest) -> str:
    return sha256_json(manifest.to_dict())


def _cache_key(base: LoadedBaseModel, manifest: FacetManifest) -> tuple[tuple[str, ...], str, str, str]:
    binding = base.binding
    identity = (
        binding.artifact_sha256,
        binding.config_sha256,
        binding.tokenizer_sha256,
        base.request.dtype,
        str(base.device),
        sha256_json({
            "modelKwargs": base.request.model_kwargs,
            "tokenizerKwargs": base.request.tokenizer_kwargs,
            "revision": base.request.revision,
            "layerPath": binding.layer_path,
        }),
    )
    return (identity, manifest.facet_id, manifest.version, _manifest_digest(manifest))
