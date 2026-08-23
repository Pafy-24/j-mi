from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import ConfigurationError
from .util import optional_mapping, require_list, require_mapping, string_list


FACET_SCHEMA_VERSION = "mi-facet-v1"
TRAINING_SCHEMA_VERSION = "mi-training-v1"
GENERATION_SCHEMA_VERSION = "mi-generate-v1"

_FACET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


def validate_facet_id(value: str, field_name: str = "facet.id") -> str:
    if not isinstance(value, str) or not _FACET_ID.fullmatch(value):
        raise ConfigurationError(
            f"{field_name} must match {_FACET_ID.pattern!r}; received {value!r}"
        )
    return value


@dataclass(frozen=True)
class BaseModelRequest:
    model_id: str
    revision: str | None = None
    layer_path: str | None = None
    dtype: str = "auto"
    device: str = "auto"
    allow_download: bool = False
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "BaseModelRequest":
        data = require_mapping(raw, "baseModel")
        model_id = str(data.get("id", "")).strip()
        if not model_id:
            raise ConfigurationError("baseModel.id is required")
        dtype = str(data.get("dtype", "auto")).lower()
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigurationError(
                "baseModel.dtype must be auto, float32, float16 or bfloat16"
            )
        return cls(
            model_id=model_id,
            revision=str(data["revision"]) if data.get("revision") is not None else None,
            layer_path=str(data["layerPath"]) if data.get("layerPath") else None,
            dtype=dtype,
            device=str(data.get("device", "auto")),
            allow_download=bool(data.get("allowDownload", False)),
            trust_remote_code=bool(data.get("trustRemoteCode", False)),
            model_kwargs=optional_mapping(data.get("modelKwargs"), "baseModel.modelKwargs"),
            tokenizer_kwargs=optional_mapping(
                data.get("tokenizerKwargs"), "baseModel.tokenizerKwargs"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "revision": self.revision,
            "layerPath": self.layer_path,
            "dtype": self.dtype,
            "device": self.device,
            "allowDownload": self.allow_download,
            "trustRemoteCode": self.trust_remote_code,
            "modelKwargs": self.model_kwargs,
            "tokenizerKwargs": self.tokenizer_kwargs,
        }


@dataclass(frozen=True)
class FacetReference:
    facet_id: str
    version: str | None = None

    @classmethod
    def parse(cls, value: str) -> "FacetReference":
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError("Facet references must be non-empty strings")
        text = value.strip()
        facet_id, marker, version = text.partition("@")
        validate_facet_id(facet_id, "facet reference")
        if marker and not version:
            raise ConfigurationError(f"Facet reference has an empty version: {value!r}")
        return cls(facet_id=facet_id, version=version or None)

    def key(self) -> str:
        return f"{self.facet_id}@{self.version}" if self.version else self.facet_id


@dataclass(frozen=True)
class BaseBinding:
    model_id: str
    revision: str | None
    config_sha256: str
    layer_path: str
    base_layer_count: int
    model_type: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> "BaseBinding":
        data = require_mapping(raw, "base")
        model_id = str(data.get("modelId", "")).strip()
        config_sha256 = str(data.get("configSha256", "")).strip()
        layer_path = str(data.get("layerPath", "")).strip()
        base_layer_count = int(data.get("baseLayerCount", 0))
        if not model_id or not config_sha256 or not layer_path or base_layer_count <= 0:
            raise ConfigurationError(
                "base requires modelId, configSha256, layerPath and a positive baseLayerCount"
            )
        return cls(
            model_id=model_id,
            revision=str(data["revision"]) if data.get("revision") is not None else None,
            config_sha256=config_sha256,
            layer_path=layer_path,
            base_layer_count=base_layer_count,
            model_type=str(data.get("modelType", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelId": self.model_id,
            "revision": self.revision,
            "configSha256": self.config_sha256,
            "layerPath": self.layer_path,
            "baseLayerCount": self.base_layer_count,
            "modelType": self.model_type,
        }


@dataclass(frozen=True)
class BlockSpec:
    source_layer: int
    initializer: str = "identity-copy"
    identity_targets: tuple[str, ...] = ()
    checkpoint_prefix: str = ""

    @classmethod
    def from_dict(cls, raw: Any, name: str) -> "BlockSpec":
        data = require_mapping(raw, name)
        if "sourceLayer" not in data:
            raise ConfigurationError(f"{name}.sourceLayer is required")
        source_layer = int(data["sourceLayer"])
        if source_layer < 0:
            raise ConfigurationError(f"{name}.sourceLayer must be non-negative")
        initializer = str(data.get("initializer", "identity-copy"))
        if initializer not in {"identity-copy", "copy", "reset", "zero"}:
            raise ConfigurationError(
                f"{name}.initializer must be identity-copy, copy, reset or zero"
            )
        return cls(
            source_layer=source_layer,
            initializer=initializer,
            identity_targets=string_list(data.get("identityTargets"), f"{name}.identityTargets"),
            checkpoint_prefix=str(data.get("checkpointPrefix", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sourceLayer": self.source_layer,
            "initializer": self.initializer,
        }
        if self.identity_targets:
            result["identityTargets"] = list(self.identity_targets)
        if self.checkpoint_prefix:
            result["checkpointPrefix"] = self.checkpoint_prefix
        return result


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    after_layer: int
    order: int
    blocks: tuple[BlockSpec, ...]

    @classmethod
    def from_dict(cls, raw: Any, name: str) -> "SlotSpec":
        data = require_mapping(raw, name)
        slot_id = str(data.get("id", "")).strip()
        if not slot_id or not _FACET_ID.fullmatch(slot_id):
            raise ConfigurationError(f"{name}.id must be a stable lowercase identifier")
        after_layer = int(data.get("afterLayer", -2))
        if after_layer < -1:
            raise ConfigurationError(f"{name}.afterLayer must be -1 or a base-layer index")
        block_items = require_list(data.get("blocks"), f"{name}.blocks")
        if not block_items:
            raise ConfigurationError(f"{name}.blocks cannot be empty")
        blocks = tuple(
            BlockSpec.from_dict(item, f"{name}.blocks[{index}]")
            for index, item in enumerate(block_items)
        )
        return cls(
            slot_id=slot_id,
            after_layer=after_layer,
            order=int(data.get("order", 0)),
            blocks=blocks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.slot_id,
            "afterLayer": self.after_layer,
            "order": self.order,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True)
class DependencySpec:
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    before: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    priority: int = 0

    @classmethod
    def from_dict(cls, raw: Any) -> "DependencySpec":
        data = {} if raw is None else require_mapping(raw, "dependencies")
        result = cls(
            requires=string_list(data.get("requires"), "dependencies.requires"),
            after=string_list(data.get("after"), "dependencies.after"),
            before=string_list(data.get("before"), "dependencies.before"),
            conflicts=string_list(data.get("conflicts"), "dependencies.conflicts"),
            priority=int(data.get("priority", 0)),
        )
        for item in (*result.requires, *result.after, *result.before, *result.conflicts):
            FacetReference.parse(item)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires": list(self.requires),
            "after": list(self.after),
            "before": list(self.before),
            "conflicts": list(self.conflicts),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class ArtifactSpec:
    weights: str
    format: str = "safetensors"
    sha256: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> "ArtifactSpec":
        data = require_mapping(raw, "artifact")
        weights = str(data.get("weights", "")).strip()
        format_name = str(data.get("format", "safetensors"))
        if not weights:
            raise ConfigurationError("artifact.weights is required")
        if format_name not in {"safetensors", "torch"}:
            raise ConfigurationError("artifact.format must be safetensors or torch")
        return cls(weights=weights, format=format_name, sha256=str(data.get("sha256", "")))

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights, "format": self.format, "sha256": self.sha256}


@dataclass(frozen=True)
class FacetManifest:
    facet_id: str
    version: str
    base: BaseBinding
    slots: tuple[SlotSpec, ...]
    dependencies: DependencySpec = field(default_factory=DependencySpec)
    artifact: ArtifactSpec | None = None
    description: str = ""
    module_type: str = "full-decoder-block"
    status: str = "trained"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "FacetManifest":
        data = require_mapping(raw, "facet manifest")
        if data.get("schemaVersion") != FACET_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported facet schemaVersion {data.get('schemaVersion')!r}; "
                f"expected {FACET_SCHEMA_VERSION!r}"
            )
        facet_id = validate_facet_id(str(data.get("id", "")))
        version = str(data.get("version", "")).strip()
        if not version:
            raise ConfigurationError("facet.version is required")
        module = require_mapping(data.get("module"), "module")
        module_type = str(module.get("type", "full-decoder-block"))
        if module_type != "full-decoder-block":
            raise ConfigurationError(
                f"Unsupported module.type {module_type!r}; MI v1 implements full-decoder-block"
            )
        slot_items = require_list(module.get("slots"), "module.slots")
        if not slot_items:
            raise ConfigurationError("module.slots cannot be empty")
        slots = tuple(
            SlotSpec.from_dict(item, f"module.slots[{index}]")
            for index, item in enumerate(slot_items)
        )
        ensure_unique_slots(slots)
        return cls(
            facet_id=facet_id,
            version=version,
            description=str(data.get("description", "")),
            status=str(data.get("status", "trained")),
            base=BaseBinding.from_dict(data.get("base")),
            dependencies=DependencySpec.from_dict(data.get("dependencies")),
            module_type=module_type,
            slots=slots,
            artifact=ArtifactSpec.from_dict(data["artifact"]) if data.get("artifact") else None,
            metadata=optional_mapping(data.get("metadata"), "metadata"),
        )

    def reference(self) -> str:
        return f"{self.facet_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schemaVersion": FACET_SCHEMA_VERSION,
            "id": self.facet_id,
            "version": self.version,
            "description": self.description,
            "status": self.status,
            "base": self.base.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "module": {
                "type": self.module_type,
                "slots": [slot.to_dict() for slot in self.slots],
            },
            "metadata": self.metadata,
        }
        if self.artifact is not None:
            result["artifact"] = self.artifact.to_dict()
        return result


def ensure_unique_slots(slots: Iterable[SlotSpec]) -> None:
    seen: set[str] = set()
    for slot in slots:
        if slot.slot_id in seen:
            raise ConfigurationError(f"Duplicate slot id in one facet: {slot.slot_id}")
        seen.add(slot.slot_id)
