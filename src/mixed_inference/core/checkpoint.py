from __future__ import annotations

import os
import hashlib
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .errors import CompatibilityError, ConfigurationError
from .initialization import InitializationReport, initialize_expansion_block
from .specs import ArtifactSpec, FacetManifest, SlotSpec
from .util import atomic_write_json, load_json, sha256_file


MANIFEST_NAME = "facet.mi.json"
DEFAULT_WEIGHTS_NAME = "facet.safetensors"


@dataclass
class LoadedFacet:
    manifest: FacetManifest
    directory: Path | None
    blocks: dict[str, list[Any]]
    initialization: dict[str, list[InitializationReport]]

    def block(self, slot_id: str, index: int) -> Any:
        try:
            return self.blocks[slot_id][index]
        except (KeyError, IndexError) as exc:
            raise CompatibilityError(
                f"Facet {self.manifest.reference()} has no block {slot_id}[{index}]"
            ) from exc

    def parameters(self) -> Iterable[Any]:
        for blocks in self.blocks.values():
            for block in blocks:
                yield from block.parameters()

    def buffers(self) -> Iterable[Any]:
        for blocks in self.blocks.values():
            for block in blocks:
                yield from block.buffers()

    def named_parameters(self) -> Iterable[tuple[str, Any]]:
        for slot_id, blocks in self.blocks.items():
            for index, block in enumerate(blocks):
                prefix = f"{slot_id}.{index}"
                for name, parameter in block.named_parameters():
                    yield f"{prefix}.{name}", parameter


@dataclass(frozen=True)
class ArtifactFileSnapshot:
    path: Path
    size_bytes: int

    def close(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def materialize_facet(manifest: FacetManifest, base_layers: Any) -> LoadedFacet:
    blocks: dict[str, list[Any]] = {}
    reports: dict[str, list[InitializationReport]] = {}
    for slot in manifest.slots:
        slot_blocks: list[Any] = []
        slot_reports: list[InitializationReport] = []
        for spec in slot.blocks:
            if spec.source_layer >= len(base_layers):
                raise CompatibilityError(
                    f"Facet {manifest.facet_id} sourceLayer={spec.source_layer} exceeds base layer count"
                )
            block, report = initialize_expansion_block(base_layers[spec.source_layer], spec)
            slot_blocks.append(block)
            slot_reports.append(report)
        blocks[slot.slot_id] = slot_blocks
        reports[slot.slot_id] = slot_reports
    return LoadedFacet(
        manifest=manifest,
        directory=None,
        blocks=blocks,
        initialization=reports,
    )


def load_facet(
    directory_or_manifest: str | Path,
    base_layers: Any,
    *,
    expected_manifest: FacetManifest | None = None,
    artifact_snapshot: ArtifactFileSnapshot | None = None,
) -> LoadedFacet:
    source = Path(directory_or_manifest)
    manifest_path = source if source.is_file() else source / MANIFEST_NAME
    manifest = FacetManifest.from_dict(load_json(manifest_path))
    if expected_manifest is not None and manifest.to_dict() != expected_manifest.to_dict():
        raise CompatibilityError(
            f"Facet manifest changed after resolution: {manifest_path}"
        )
    if manifest.artifact is None:
        raise CompatibilityError(f"Facet manifest has no artifact: {manifest_path}")
    artifact_path = manifest_path.parent / manifest.artifact.weights
    if artifact_snapshot is not None:
        snapshot = artifact_snapshot
        verify_artifact_file(
            snapshot.path,
            expected_sha256=manifest.artifact.sha256,
            expected_size=snapshot.size_bytes,
        )
        load_path = snapshot.path
    else:
        # Compatibility path for direct registry callers. The bounded runtime
        # always supplies an owned snapshot; this path only verifies then reads
        # the source in place and does not claim filesystem snapshot isolation.
        verify_artifact_file(
            artifact_path,
            expected_sha256=manifest.artifact.sha256,
            expected_size=manifest.artifact.size_bytes,
        )
        load_path = artifact_path
    # Bounded callers verify manifest identity and the exact owned artifact
    # snapshot before a decoder block is materialized.
    loaded = materialize_facet(manifest, base_layers)
    state = _load_state(load_path, manifest.artifact.format)
    consumed: set[str] = set()
    for slot in manifest.slots:
        for index, spec in enumerate(slot.blocks):
            prefix = spec.checkpoint_prefix or checkpoint_prefix(slot, index)
            prefix_dot = prefix + "."
            block_state = {
                name[len(prefix_dot) :]: tensor
                for name, tensor in state.items()
                if name.startswith(prefix_dot)
            }
            consumed.update(name for name in state if name.startswith(prefix_dot))
            if not block_state:
                raise CompatibilityError(
                    f"No checkpoint tensors found for {manifest.reference()} prefix {prefix!r}"
                )
            missing, unexpected = loaded.block(slot.slot_id, index).load_state_dict(block_state, strict=False)
            if missing or unexpected:
                raise CompatibilityError(
                    f"Checkpoint structure mismatch for {manifest.reference()} {prefix}: "
                    f"missing={list(missing)}, unexpected={list(unexpected)}"
                )
    unused = sorted(set(state) - consumed)
    if unused:
        raise CompatibilityError(
            f"Facet checkpoint contains {len(unused)} unused tensors; first: {unused[:3]}"
        )
    loaded.directory = manifest_path.parent
    return loaded


def verify_artifact_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
) -> int:
    """Hash one opened artifact handle without allocating its payload."""
    if not path.is_file():
        raise CompatibilityError(f"Facet weights do not exist: {path}")
    digest = hashlib.sha256()
    copied = 0
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if expected_size is not None and size != expected_size:
            raise CompatibilityError(
                f"Facet weight size mismatch: {size} != {expected_size}"
            )
        while chunk := stream.read(1024 * 1024):
            copied += len(chunk)
            if copied > size:
                raise CompatibilityError(
                    "Facet weights grew while their identity was verified"
                )
            digest.update(chunk)
    if copied != size:
        raise CompatibilityError(
            f"Facet weights changed size while verified: {copied} != {size}"
        )
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        raise CompatibilityError(
            f"Facet weight hash mismatch: {digest.hexdigest()} != {expected_sha256}"
        )
    return size


def snapshot_artifact_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    destination_directory: Path,
) -> ArtifactFileSnapshot:
    """Copy one verified artifact into a caller-owned bounded directory."""
    if not path.is_file():
        raise CompatibilityError(f"Facet weights do not exist: {path}")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise CompatibilityError("Facet snapshot size must be a non-negative integer")
    destination_directory.mkdir(parents=False, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix="mi-facet-", suffix=path.suffix, dir=destination_directory
    )
    os.close(fd)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied = 0
    try:
        with path.open("rb") as source, temporary.open("wb") as target:
            size = os.fstat(source.fileno()).st_size
            if size != expected_size:
                raise CompatibilityError(
                    f"Facet weight size mismatch: {size} != {expected_size}"
                )
            while copied < expected_size:
                chunk = source.read(min(1024 * 1024, expected_size - copied))
                if not chunk:
                    break
                copied += len(chunk)
                digest.update(chunk)
                target.write(chunk)
            if source.read(1):
                raise CompatibilityError(
                    "Facet weights grew while the validated snapshot was copied"
                )
        if copied != expected_size:
            raise CompatibilityError(
                f"Facet weights changed size while copied: {copied} != {expected_size}"
            )
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            raise CompatibilityError(
                f"Facet weight hash mismatch: {digest.hexdigest()} != {expected_sha256}"
            )
        return ArtifactFileSnapshot(path=temporary, size_bytes=copied)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def save_facet(
    facet: LoadedFacet,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> FacetManifest:
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ConfigurationError(
            f"Output directory is not empty: {output}. Set checkpoint.overwrite=true to replace MI files."
        )
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / DEFAULT_WEIGHTS_NAME
    state: dict[str, Any] = {}
    rewritten_slots: list[SlotSpec] = []
    for slot in facet.manifest.slots:
        rewritten_blocks = []
        for index, spec in enumerate(slot.blocks):
            prefix = checkpoint_prefix(slot, index)
            rewritten_blocks.append(replace(spec, checkpoint_prefix=prefix))
            for name, tensor in facet.block(slot.slot_id, index).state_dict().items():
                key = f"{prefix}.{name}"
                state[key] = tensor.detach().cpu().contiguous()
        rewritten_slots.append(replace(slot, blocks=tuple(rewritten_blocks)))
    _save_safetensors_atomic(weights_path, state)
    artifact = ArtifactSpec(
        weights=DEFAULT_WEIGHTS_NAME,
        format="safetensors",
        sha256=sha256_file(weights_path),
        size_bytes=weights_path.stat().st_size,
    )
    manifest = replace(facet.manifest, slots=tuple(rewritten_slots), artifact=artifact)
    atomic_write_json(output / MANIFEST_NAME, manifest.to_dict())
    facet.manifest = manifest
    facet.directory = output
    return manifest


def checkpoint_prefix(slot: SlotSpec, index: int) -> str:
    return f"slots.{slot.slot_id}.blocks.{index:04d}"


def _save_safetensors_atomic(path: Path, state: dict[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ConfigurationError("safetensors is required to save facet checkpoints") from exc
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file(state, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_state(path: Path, format_name: str) -> dict[str, Any]:
    if format_name == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ConfigurationError("safetensors is required to load this facet") from exc
        return load_file(str(path), device="cpu")
    if format_name == "torch":
        try:
            import torch

            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(path, map_location="cpu")
        except ImportError as exc:
            raise ConfigurationError("PyTorch is required to load this facet") from exc
    raise ConfigurationError(f"Unsupported checkpoint format: {format_name}")
