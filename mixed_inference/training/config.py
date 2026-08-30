from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..core.errors import ConfigurationError
from ..core.specs import (
    TRAINING_SCHEMA_VERSION,
    BaseModelRequest,
    BlockSpec,
    DependencySpec,
    SlotSpec,
    ensure_unique_slots,
    validate_facet_id,
)
from ..core.util import (
    load_json,
    optional_mapping,
    require_list,
    require_mapping,
    resolve_relative,
    string_list,
)


@dataclass(frozen=True)
class DatasetSource:
    path: Path
    logical_path: str | None = None
    stages: tuple[str, ...] = ()
    repeat: int = 1
    limit: int | None = None


@dataclass(frozen=True)
class DataConfig:
    train: tuple[DatasetSource, ...]
    validation: tuple[DatasetSource, ...] = ()
    max_sequence_length: int = 1024
    mask_prompt: bool = True
    packing: bool = False
    shuffle: bool = True
    num_workers: int = 0


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.0
    class_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    class_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmConfig:
    type: str = "causal-lm"
    trainer_class_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopConfig:
    epochs: float = 1.0
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    precision: str = "float32"
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    log_every_steps: int = 10
    evaluate_every_steps: int = 0
    identity_check: bool = True
    identity_atol: float | None = None
    identity_rtol: float = 1e-4


@dataclass(frozen=True)
class CheckpointConfig:
    output_directory: Path
    overwrite: bool = False
    save_every_steps: int = 0
    save_optimizer: bool = True
    resume_from: Path | None = None


@dataclass(frozen=True)
class FacetDraft:
    facet_id: str
    version: str
    description: str
    dependencies: DependencySpec
    module: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    source_path: Path | None
    base_model: BaseModelRequest
    registry: Path
    facet: FacetDraft
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    algorithm: AlgorithmConfig
    loop: LoopConfig
    checkpoint: CheckpointConfig
    allow_custom_classes: bool = False

    @classmethod
    def load(cls, source: str | Path | dict[str, Any]) -> "TrainingConfig":
        if isinstance(source, dict):
            raw = source
            source_path = None
            base_directory = Path.cwd()
        else:
            source_path = Path(source).resolve()
            raw = load_json(source_path)
            base_directory = source_path.parent
        data = require_mapping(raw, "training config")
        if data.get("schemaVersion") != TRAINING_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported schemaVersion {data.get('schemaVersion')!r}; "
                f"expected {TRAINING_SCHEMA_VERSION!r}"
            )
        base_model = BaseModelRequest.from_dict(data.get("baseModel"))
        if source_path is not None:
            model_path = Path(base_model.model_id)
            local_candidate = base_directory / model_path
            if (
                not model_path.is_absolute()
                and (base_model.model_id.startswith(".") or local_candidate.exists())
            ):
                base_model = replace(base_model, model_id=str(local_candidate.resolve()))
        registry = resolve_relative(str(data.get("registry", "artifacts/facets")), base_directory)
        facet = _parse_facet_draft(data.get("facet"))
        data_config = _parse_data(data.get("data"), base_directory)
        optimizer = _parse_optimizer(data.get("optimizer"))
        scheduler = _parse_scheduler(data.get("scheduler"))
        algorithm = _parse_algorithm(data.get("algorithm"))
        loop = _parse_loop(data.get("training"))
        checkpoint = _parse_checkpoint(data.get("checkpoint"), base_directory)
        allow_custom = bool(data.get("allowCustomClasses", False))
        if (
            optimizer.class_path
            or scheduler.class_path
            or algorithm.trainer_class_path
        ) and not allow_custom:
            raise ConfigurationError(
                "Custom trainer/optimizer/scheduler classes require allowCustomClasses=true"
            )
        return cls(
            source_path=source_path,
            base_model=base_model,
            registry=registry,
            facet=facet,
            data=data_config,
            optimizer=optimizer,
            scheduler=scheduler,
            algorithm=algorithm,
            loop=loop,
            checkpoint=checkpoint,
            allow_custom_classes=allow_custom,
        )


def resolve_slots(module: dict[str, Any], base_layer_count: int) -> tuple[SlotSpec, ...]:
    module_type = str(module.get("type", "full-decoder-block"))
    if module_type != "full-decoder-block":
        raise ConfigurationError(
            f"MI v1 implements module.type=full-decoder-block, not {module_type!r}"
        )
    if module.get("slots") is not None:
        items = require_list(module.get("slots"), "facet.module.slots")
        if not items:
            raise ConfigurationError("facet.module.slots cannot be empty")
        slots = tuple(
            SlotSpec.from_dict(item, f"facet.module.slots[{index}]")
            for index, item in enumerate(items)
        )
    else:
        placement = require_mapping(module.get("placement"), "facet.module.placement")
        strategy = str(placement.get("strategy", "uniform"))
        if strategy != "uniform":
            raise ConfigurationError("Only explicit slots or placement.strategy=uniform are supported")
        slot_count = int(placement.get("slotCount", 0))
        if slot_count <= 0 or slot_count > base_layer_count:
            raise ConfigurationError(
                f"placement.slotCount must be between 1 and {base_layer_count}"
            )
        positions = _uniform_positions(base_layer_count, slot_count)
        counts_raw = placement.get("blocksPerSlot", 1)
        if isinstance(counts_raw, list):
            counts = [int(value) for value in counts_raw]
            if len(counts) != slot_count:
                raise ConfigurationError(
                    "placement.blocksPerSlot array length must equal placement.slotCount"
                )
        else:
            counts = [int(counts_raw)] * slot_count
        if any(count <= 0 for count in counts):
            raise ConfigurationError("placement.blocksPerSlot values must be positive")
        initializer = str(placement.get("initializer", "identity-copy"))
        identity_targets = string_list(
            placement.get("identityTargets"), "facet.module.placement.identityTargets"
        )
        source_mode = placement.get("sourceLayer", "preceding")
        slots_list: list[SlotSpec] = []
        for index, (position, block_count) in enumerate(zip(positions, counts)):
            source_layer = position if source_mode == "preceding" else int(source_mode)
            blocks = tuple(
                BlockSpec(
                    source_layer=source_layer,
                    initializer=initializer,
                    identity_targets=identity_targets,
                )
                for _ in range(block_count)
            )
            slots_list.append(
                SlotSpec(
                    slot_id=f"slot-{index:04d}",
                    after_layer=position,
                    order=int(placement.get("order", 0)),
                    blocks=blocks,
                )
            )
        slots = tuple(slots_list)
    ensure_unique_slots(slots)
    for slot in slots:
        if slot.after_layer >= base_layer_count:
            raise ConfigurationError(
                f"Slot {slot.slot_id} afterLayer={slot.after_layer} exceeds base layer count"
            )
        for block in slot.blocks:
            if block.source_layer >= base_layer_count:
                raise ConfigurationError(
                    f"Slot {slot.slot_id} sourceLayer={block.source_layer} exceeds base layer count"
                )
    return slots


def _uniform_positions(layer_count: int, slot_count: int) -> list[int]:
    positions = [((index + 1) * layer_count // slot_count) - 1 for index in range(slot_count)]
    if len(set(positions)) != slot_count or positions[0] < 0:
        raise ConfigurationError(
            f"Cannot place {slot_count} distinct uniform slots in {layer_count} layers"
        )
    return positions


def _parse_facet_draft(raw: Any) -> FacetDraft:
    data = require_mapping(raw, "facet")
    facet_id = validate_facet_id(str(data.get("id", "")))
    version = str(data.get("version", "")).strip()
    if not version:
        raise ConfigurationError("facet.version is required")
    module = dict(require_mapping(data.get("module"), "facet.module"))
    return FacetDraft(
        facet_id=facet_id,
        version=version,
        description=str(data.get("description", "")),
        dependencies=DependencySpec.from_dict(data.get("dependencies")),
        module=module,
        metadata=optional_mapping(data.get("metadata"), "facet.metadata"),
    )


def _parse_dataset_sources(raw: Any, name: str, base: Path) -> tuple[DatasetSource, ...]:
    items = require_list(raw, name)
    sources: list[DatasetSource] = []
    for index, item in enumerate(items):
        field_name = f"{name}[{index}]"
        if isinstance(item, str):
            path = item
            data = {}
        else:
            data = require_mapping(item, field_name)
            path = str(data.get("path", ""))
        if not path:
            raise ConfigurationError(f"{field_name}.path is required")
        repeat = int(data.get("repeat", 1))
        if repeat <= 0:
            raise ConfigurationError(f"{field_name}.repeat must be positive")
        limit = int(data["limit"]) if data.get("limit") is not None else None
        if limit is not None and limit <= 0:
            raise ConfigurationError(f"{field_name}.limit must be positive")
        logical_path = str(data.get("logicalPath", "")).strip() or None
        if logical_path and Path(logical_path).suffix.lower() not in {
            ".json",
            ".jsonl",
        }:
            raise ConfigurationError(
                f"{field_name}.logicalPath must identify a JSON or JSONL source"
            )
        sources.append(
            DatasetSource(
                path=resolve_relative(path, base),
                logical_path=logical_path,
                stages=string_list(data.get("stages"), f"{field_name}.stages"),
                repeat=repeat,
                limit=limit,
            )
        )
    return tuple(sources)


def _parse_data(raw: Any, base: Path) -> DataConfig:
    data = require_mapping(raw, "data")
    train = _parse_dataset_sources(data.get("train"), "data.train", base)
    if not train:
        raise ConfigurationError("data.train cannot be empty")
    validation = _parse_dataset_sources(data.get("validation", []), "data.validation", base)
    maximum = int(data.get("maxSequenceLength", 1024))
    if maximum < 8:
        raise ConfigurationError("data.maxSequenceLength must be at least 8")
    result = DataConfig(
        train=train,
        validation=validation,
        max_sequence_length=maximum,
        mask_prompt=bool(data.get("maskPrompt", True)),
        packing=bool(data.get("packing", False)),
        shuffle=bool(data.get("shuffle", True)),
        num_workers=int(data.get("numWorkers", 0)),
    )
    if result.num_workers < 0:
        raise ConfigurationError("data.numWorkers must be non-negative")
    return result


def _parse_optimizer(raw: Any) -> OptimizerConfig:
    data = {} if raw is None else require_mapping(raw, "optimizer")
    betas_raw = data.get("betas", [0.9, 0.999])
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise ConfigurationError("optimizer.betas must have two numbers")
    result = OptimizerConfig(
        type=str(data.get("type", "adamw")).lower(),
        learning_rate=float(data.get("learningRate", 2e-4)),
        weight_decay=float(data.get("weightDecay", 0.0)),
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        eps=float(data.get("eps", 1e-8)),
        momentum=float(data.get("momentum", 0.0)),
        class_path=str(data["classPath"]) if data.get("classPath") else None,
        kwargs=optional_mapping(data.get("kwargs"), "optimizer.kwargs"),
    )
    if result.learning_rate <= 0:
        raise ConfigurationError("optimizer.learningRate must be positive")
    if result.weight_decay < 0 or result.eps <= 0:
        raise ConfigurationError("optimizer.weightDecay must be non-negative and eps positive")
    return result


def _parse_scheduler(raw: Any) -> SchedulerConfig:
    data = {} if raw is None else require_mapping(raw, "scheduler")
    result = SchedulerConfig(
        type=str(data.get("type", "cosine")).lower(),
        warmup_steps=int(data.get("warmupSteps", 0)),
        warmup_ratio=float(data.get("warmupRatio", 0.0)),
        class_path=str(data["classPath"]) if data.get("classPath") else None,
        kwargs=optional_mapping(data.get("kwargs"), "scheduler.kwargs"),
    )
    if result.warmup_steps < 0 or not 0.0 <= result.warmup_ratio < 1.0:
        raise ConfigurationError("scheduler warmupSteps must be non-negative and warmupRatio in [0, 1)")
    return result


def _parse_algorithm(raw: Any) -> AlgorithmConfig:
    data = {} if raw is None else require_mapping(raw, "algorithm")
    result = AlgorithmConfig(
        type=str(data.get("type", "causal-lm")).lower(),
        trainer_class_path=(
            str(data["trainerClassPath"]) if data.get("trainerClassPath") else None
        ),
        kwargs=optional_mapping(data.get("kwargs"), "algorithm.kwargs"),
    )
    if result.type != "causal-lm" and result.trainer_class_path is None:
        raise ConfigurationError(
            "algorithm.type must be causal-lm unless algorithm.trainerClassPath is provided"
        )
    return result


def _parse_loop(raw: Any) -> LoopConfig:
    data = {} if raw is None else require_mapping(raw, "training")
    precision = str(data.get("precision", "float32")).lower()
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ConfigurationError("training.precision must be float32, float16 or bfloat16")
    result = LoopConfig(
        epochs=float(data.get("epochs", 1.0)),
        max_steps=int(data["maxSteps"]) if data.get("maxSteps") is not None else None,
        batch_size=int(data.get("batchSize", 1)),
        gradient_accumulation_steps=int(data.get("gradientAccumulationSteps", 1)),
        precision=precision,
        gradient_checkpointing=bool(data.get("gradientCheckpointing", False)),
        max_grad_norm=float(data.get("maxGradNorm", 1.0)),
        seed=int(data.get("seed", 42)),
        log_every_steps=int(data.get("logEverySteps", 10)),
        evaluate_every_steps=int(data.get("evaluateEverySteps", 0)),
        identity_check=bool(data.get("identityCheck", True)),
        identity_atol=float(data["identityAtol"]) if data.get("identityAtol") is not None else None,
        identity_rtol=float(data.get("identityRtol", 1e-4)),
    )
    if result.epochs <= 0 or result.batch_size <= 0 or result.gradient_accumulation_steps <= 0:
        raise ConfigurationError("epochs, batchSize and gradientAccumulationSteps must be positive")
    if result.max_steps is not None and result.max_steps <= 0:
        raise ConfigurationError("training.maxSteps must be positive")
    if result.log_every_steps < 0 or result.evaluate_every_steps < 0:
        raise ConfigurationError("training log/evaluation intervals must be non-negative")
    if result.max_grad_norm < 0:
        raise ConfigurationError("training.maxGradNorm must be non-negative")
    if result.identity_rtol < 0 or (result.identity_atol is not None and result.identity_atol < 0):
        raise ConfigurationError("training identity tolerances must be non-negative")
    return result


def _parse_checkpoint(raw: Any, base: Path) -> CheckpointConfig:
    data = require_mapping(raw, "checkpoint")
    output = str(data.get("outputDirectory", "")).strip()
    if not output:
        raise ConfigurationError("checkpoint.outputDirectory is required")
    resume = data.get("resumeFrom")
    result = CheckpointConfig(
        output_directory=resolve_relative(output, base),
        overwrite=bool(data.get("overwrite", False)),
        save_every_steps=int(data.get("saveEverySteps", 0)),
        save_optimizer=bool(data.get("saveOptimizer", True)),
        resume_from=resolve_relative(str(resume), base) if resume else None,
    )
    if result.save_every_steps < 0:
        raise ConfigurationError("checkpoint.saveEverySteps must be non-negative")
    return result
