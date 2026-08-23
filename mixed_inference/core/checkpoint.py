from __future__ import annotations

import os
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

    def named_parameters(self) -> Iterable[tuple[str, Any]]:
        for slot_id, blocks in self.blocks.items():
            for index, block in enumerate(blocks):
                prefix = f"{slot_id}.{index}"
                for name, parameter in block.named_parameters():
                    yield f"{prefix}.{name}", parameter


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


def load_facet(directory_or_manifest: str | Path, base_layers: Any) -> LoadedFacet:
    source = Path(directory_or_manifest)
    manifest_path = source if source.is_file() else source / MANIFEST_NAME
    manifest = FacetManifest.from_dict(load_json(manifest_path))
    if manifest.artifact is None:
        raise CompatibilityError(f"Facet manifest has no artifact: {manifest_path}")
    artifact_path = manifest_path.parent / manifest.artifact.weights
    if not artifact_path.exists():
        raise CompatibilityError(f"Facet weights do not exist: {artifact_path}")
    if manifest.artifact.sha256:
        actual_hash = sha256_file(artifact_path)
        if actual_hash != manifest.artifact.sha256:
            raise CompatibilityError(
                f"Facet weight hash mismatch for {manifest.reference()}: "
                f"{actual_hash} != {manifest.artifact.sha256}"
            )
    loaded = materialize_facet(manifest, base_layers)
    state = _load_state(artifact_path, manifest.artifact.format)
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
