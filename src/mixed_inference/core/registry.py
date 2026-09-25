from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable

from .checkpoint import (
    MANIFEST_NAME,
    ArtifactFileSnapshot,
    LoadedFacet,
    load_facet,
    snapshot_artifact_file,
    verify_artifact_file,
)
from .errors import CompatibilityError, MissingDependencyError
from .graph import CompositionPlan, CompositionResolver
from .modeling import LoadedBaseModel, validate_base_binding
from .specs import FacetManifest, FacetReference
from .util import load_json, sha256_json


@dataclass(frozen=True)
class StagedFacetLoad:
    plan: CompositionPlan
    loaded: dict[str, LoadedFacet]
    staged: dict[tuple[tuple[str, ...], str, str, str], LoadedFacet]
    registry_version: int


class FacetRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._manifests: dict[tuple[str, str], tuple[FacetManifest, Path]] = {}
        # A facet's bytes are initialized against a concrete base stack.  Never
        # reuse by name/version alone across base artifact/config/tokenizer,
        # dtype, device, request configuration, or manifest changes.
        self._loaded: dict[tuple[tuple[str, ...], str, str, str], LoadedFacet] = {}
        self._version = 0
        self._lock = RLock()
        self.refresh()

    def refresh(self) -> tuple[tuple[tuple[str, ...], str, str, str], ...]:
        with self._lock:
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
            if previous != manifests or invalidated:
                self._version += 1
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
        staged = self.stage_resolved(
            requested,
            base,
            allow_model_id_mismatch=allow_model_id_mismatch,
            allow_unsafe_legacy_base_binding=allow_unsafe_legacy_base_binding,
            resolved=resolved,
        )
        if before_commit is not None:
            before_commit(staged.loaded)
        self.commit_staged(staged)
        return staged.plan, staged.loaded

    def validate_resolved(
        self,
        base: LoadedBaseModel,
        manifests: dict[str, FacetManifest],
        *,
        allow_model_id_mismatch: bool = False,
        allow_unsafe_legacy_base_binding: bool = False,
    ) -> dict[str, int]:
        """Validate the complete identity closure before any facet allocation."""
        if allow_model_id_mismatch and not allow_unsafe_legacy_base_binding:
            raise CompatibilityError(
                "allow_model_id_mismatch is an unsafe legacy compatibility bypass; "
                "set allow_unsafe_legacy_base_binding explicitly"
            )
        validated_sizes: dict[str, int] = {}
        for manifest in manifests.values():
            validate_base_binding(
                base.binding,
                manifest.base,
                allow_model_id_mismatch=allow_model_id_mismatch,
            )
            manifest_path = self.manifest_path(manifest)
            current = FacetManifest.from_dict(load_json(manifest_path))
            if current.to_dict() != manifest.to_dict():
                raise CompatibilityError(
                    f"Facet manifest changed after resolution: {manifest_path}"
                )
            artifact = manifest.artifact
            if artifact is None:
                raise CompatibilityError(
                    f"Facet manifest has no artifact: {manifest_path}"
                )
            validated_sizes[manifest.facet_id] = verify_artifact_file(
                manifest_path.parent / artifact.weights,
                expected_sha256=artifact.sha256,
                expected_size=artifact.size_bytes,
            )
        return validated_sizes

    def snapshot_artifacts(
        self,
        manifests: Iterable[FacetManifest],
        *,
        validated_sizes: dict[str, int],
        destination_directory: Path,
    ) -> dict[str, ArtifactFileSnapshot]:
        """Bind missing facets to verified files before sizing or allocation."""
        snapshots: dict[str, ArtifactFileSnapshot] = {}
        try:
            for manifest in manifests:
                manifest_path = self.manifest_path(manifest)
                current = FacetManifest.from_dict(load_json(manifest_path))
                if current.to_dict() != manifest.to_dict():
                    raise CompatibilityError(
                        f"Facet manifest changed after resolution: {manifest_path}"
                    )
                artifact = manifest.artifact
                if artifact is None:
                    raise CompatibilityError(
                        f"Facet manifest has no artifact: {manifest_path}"
                    )
                snapshots[manifest.facet_id] = snapshot_artifact_file(
                    manifest_path.parent / artifact.weights,
                    expected_sha256=artifact.sha256,
                    expected_size=validated_sizes[manifest.facet_id],
                    destination_directory=destination_directory,
                )
            return snapshots
        except BaseException:
            for snapshot in snapshots.values():
                snapshot.close()
            raise

    def stage_resolved(
        self,
        requested: Iterable[str],
        base: LoadedBaseModel,
        *,
        allow_model_id_mismatch: bool = False,
        allow_unsafe_legacy_base_binding: bool = False,
        resolved: tuple[CompositionPlan, dict[str, FacetManifest]] | None = None,
        identities_validated: bool = False,
        artifact_snapshots: dict[str, ArtifactFileSnapshot] | None = None,
    ) -> StagedFacetLoad:
        plan, manifests = resolved or self.resolve(requested)
        if not identities_validated:
            self.validate_resolved(
                base,
                manifests,
                allow_model_id_mismatch=allow_model_id_mismatch,
                allow_unsafe_legacy_base_binding=allow_unsafe_legacy_base_binding,
            )
        with self._lock:
            registry_version = self._version
            cached = {
                _cache_key(base, manifest): self._loaded.get(_cache_key(base, manifest))
                for manifest in manifests.values()
            }
        loaded: dict[str, LoadedFacet] = {}
        staged: dict[tuple[tuple[str, ...], str, str, str], LoadedFacet] = {}
        for manifest in manifests.values():
            key = _cache_key(base, manifest)
            facet = cached[key]
            if facet is None:
                facet = load_facet(
                    self.manifest_path(manifest),
                    base.stack.layers,
                    expected_manifest=manifest,
                    artifact_snapshot=(artifact_snapshots or {}).get(manifest.facet_id),
                )
                staged[key] = facet
            loaded[manifest.facet_id] = facet
        return StagedFacetLoad(
            plan=plan,
            loaded=loaded,
            staged=staged,
            registry_version=registry_version,
        )

    def commit_staged(self, staged: StagedFacetLoad) -> None:
        """Publish only staged misses; cache hits are never rollback targets."""
        with self._lock:
            if self._version != staged.registry_version:
                raise CompatibilityError(
                    "Facet registry changed while facet tensors were staged"
                )
            inserted: list[tuple[tuple[str, ...], str, str, str]] = []
            try:
                for key, facet in staged.staged.items():
                    if key in self._loaded:
                        raise CompatibilityError(
                            "Facet cache changed while facet tensors were staged"
                        )
                    self._loaded[key] = facet
                    inserted.append(key)
            except BaseException:
                for key in inserted:
                    self._loaded.pop(key, None)
                raise
            if inserted:
                self._version += 1

    def unload(self, facets: Iterable[str] | None = None, *, all_facets: bool = False) -> list[str]:
        with self._lock:
            if all_facets:
                removed = [facet.manifest.reference() for facet in self._loaded.values()]
                self._loaded.clear()
                if removed:
                    self._version += 1
                return sorted(removed)
            requested = {FacetReference.parse(item).facet_id for item in (facets or [])}
            removed: list[str] = []
            for key, facet in list(self._loaded.items()):
                if facet.manifest.facet_id in requested:
                    removed.append(facet.manifest.reference())
                    self._loaded.pop(key, None)
            if removed:
                self._version += 1
            return sorted(removed)

    def cache_keys_for(
        self, facets: Iterable[str] | None = None, *, all_facets: bool = False
    ) -> tuple[tuple[tuple[str, ...], str, str, str], ...]:
        with self._lock:
            if all_facets:
                return tuple(self._loaded)
            requested = {FacetReference.parse(item).facet_id for item in (facets or [])}
            return tuple(
                key for key, facet in self._loaded.items()
                if facet.manifest.facet_id in requested
            )

    def loaded(self) -> list[str]:
        with self._lock:
            return sorted({facet.manifest.reference() for facet in self._loaded.values()})

    def cache_key(self, base: LoadedBaseModel, facet: LoadedFacet) -> tuple[tuple[str, ...], str, str, str]:
        return _cache_key(base, facet.manifest)

    def cache_key_for_manifest(
        self, base: LoadedBaseModel, manifest: FacetManifest
    ) -> tuple[tuple[str, ...], str, str, str]:
        return _cache_key(base, manifest)

    def is_cached(self, base: LoadedBaseModel, manifest: FacetManifest) -> bool:
        with self._lock:
            return _cache_key(base, manifest) in self._loaded

    def evict_cache_keys(self, keys: Iterable[object]) -> None:
        with self._lock:
            changed = False
            for key in keys:
                if isinstance(key, tuple):
                    changed = self._loaded.pop(key, None) is not None or changed
            if changed:
                self._version += 1


def _version_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"([0-9]+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _manifest_digest(manifest: FacetManifest) -> str:
    return sha256_json(manifest.to_dict())


def _cache_key(base: LoadedBaseModel, manifest: FacetManifest) -> tuple[tuple[str, ...], str, str, str]:
    binding = base.binding
    identity = (
        binding.artifact_sha256,
        binding.artifact_identity_status,
        binding.config_sha256,
        binding.tokenizer_sha256,
        binding.tokenizer_identity_status,
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
