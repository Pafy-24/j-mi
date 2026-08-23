from __future__ import annotations

import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable

from ..core.checkpoint import LoadedFacet, load_facet, materialize_facet, save_facet
from ..core.errors import CompatibilityError, ConfigurationError, TrainingCancelled
from ..core.graph import CompositionPlan, CompositionResolver, PlannedBlock
from ..core.modeling import (
    ActivatedComposition,
    LoadedBaseModel,
    load_base_model,
    module_device_and_dtype,
    require_torch,
    validate_base_binding,
)
from ..core.registry import FacetRegistry
from ..core.specs import FacetManifest, FacetReference
from ..core.util import atomic_write_json, import_symbol, utc_now
from .config import TrainingConfig, resolve_slots
from .data import CausalCollator, FacetTokenDataset, build_datasets
from .optim import build_optimizer, build_scheduler


ProgressCallback = Callable[[dict[str, Any]], None]


class MixedInferenceTrainer:
    """Train only newly inserted decoder blocks while keeping the base and parents frozen."""

    def __init__(
        self,
        config: TrainingConfig | str | Path | dict[str, Any],
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ):
        self.config = config if isinstance(config, TrainingConfig) else TrainingConfig.load(config)
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event or Event()
        self.base: LoadedBaseModel | None = None
        self.target: LoadedFacet | None = None
        self.plan: CompositionPlan | None = None
        self.parents: dict[str, LoadedFacet] = {}
        self._global_step = 0
        self._micro_step = 0
        self._resumed = False

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self._check_cancelled()
        self._prepare_output()
        self._seed_everything(self.config.loop.seed)
        self._emit("loading-base", message="Loading base model and tokenizer")
        base = load_base_model(self.config.base_model, load_tokenizer=True)
        self.base = base
        if base.tokenizer is None:
            raise ConfigurationError("Training requires a tokenizer")

        self._emit("building-data", base=base.summary())
        train_dataset, validation_dataset = build_datasets(base.tokenizer, self.config.data)
        target = self._prepare_target(base)
        plan, parents = self._prepare_composition(base, target)
        self.target, self.plan, self.parents = target, plan, parents
        self._move_facet_to_base(target, base)
        self._freeze_parameters(base, target, parents.values())

        collator = CausalCollator(self._pad_token_id(base.tokenizer))
        train_loader = self._data_loader(train_dataset, collator, shuffle=self.config.data.shuffle)
        validation_loader = (
            self._data_loader(validation_dataset, collator, shuffle=False)
            if validation_dataset is not None
            else None
        )
        identity = self._identity_check(train_dataset, collator)
        self._check_cancelled()

        trainable_parameters = [parameter for parameter in target.parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
        parent_count = sum(
            parameter.numel() for parent in parents.values() for parameter in parent.parameters()
        )
        total_count = (
            sum(parameter.numel() for parameter in base.model.parameters())
            + parent_count
            + trainable_count
        )
        update_steps_per_epoch = max(
            1,
            math.ceil(len(train_loader) / self.config.loop.gradient_accumulation_steps),
        )
        total_steps = self.config.loop.max_steps or max(
            1, math.ceil(self.config.loop.epochs * update_steps_per_epoch)
        )
        optimizer = build_optimizer(self.config.optimizer, trainable_parameters)
        scheduler, warmup_steps = build_scheduler(self.config.scheduler, optimizer, total_steps)
        resumed = self._restore_training_state(optimizer, scheduler)
        if self._global_step >= total_steps:
            raise ConfigurationError(
                f"resumeFrom global step {self._global_step} is not below configured total steps {total_steps}"
            )

        self._emit(
            "training-started",
            totalSteps=total_steps,
            resumed=resumed,
            trainableParameters=trainable_count,
            totalParameters=total_count,
            composition=plan.to_dict(),
            identityCheck=identity,
        )
        metrics = self._train_loop(
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
        )
        metrics.update(
            {
                "elapsedSeconds": round(time.perf_counter() - started, 6),
                "warmupSteps": warmup_steps,
                "trainExamples": len(train_dataset),
                "validationExamples": len(validation_dataset) if validation_dataset else 0,
                "trainableParameters": trainable_count,
                "baseAndFacetParameters": total_count,
                "identityCheck": identity,
            }
        )
        manifest = self._save_final(optimizer, scheduler, metrics)
        result = {
            "schemaVersion": "mi-training-result-v1",
            "status": "succeeded",
            "component": 2,
            "facet": manifest.to_dict(),
            "outputDirectory": str(self.config.checkpoint.output_directory.resolve()),
            "composition": plan.to_dict(),
            "metrics": metrics,
        }
        self._emit("training-completed", result=result)
        return result

    def _prepare_output(self) -> None:
        output = self.config.checkpoint.output_directory
        if output.exists() and any(output.iterdir()):
            if not self.config.checkpoint.overwrite and self.config.checkpoint.resume_from is None:
                raise ConfigurationError(
                    f"Output directory is not empty: {output}. Set checkpoint.overwrite=true "
                    "or checkpoint.resumeFrom."
                )
        output.mkdir(parents=True, exist_ok=True)

    def _prepare_target(self, base: LoadedBaseModel) -> LoadedFacet:
        slots = resolve_slots(self.config.facet.module, base.binding.base_layer_count)
        draft_manifest = FacetManifest(
            facet_id=self.config.facet.facet_id,
            version=self.config.facet.version,
            description=self.config.facet.description,
            status="initialized",
            base=base.binding,
            dependencies=self.config.facet.dependencies,
            module_type=str(self.config.facet.module.get("type", "full-decoder-block")),
            slots=slots,
            metadata={
                **self.config.facet.metadata,
                "createdAt": utc_now(),
                "method": "LLaMA-Pro-style block expansion",
                "trainableScope": "inserted-blocks-only",
            },
        )
        resume = self.config.checkpoint.resume_from
        if resume is None:
            target = materialize_facet(draft_manifest, base.stack.layers)
            initialization = self._initialization_report(target)
            target.manifest = replace(
                target.manifest,
                metadata={**target.manifest.metadata, "initialization": initialization},
            )
            self._emit(
                "facet-initialized",
                facet=draft_manifest.reference(),
                slots=len(slots),
                blocks=sum(len(slot.blocks) for slot in slots),
                initialization=initialization,
            )
            return target

        target = load_facet(resume, base.stack.layers)
        self._resumed = True
        validate_base_binding(base.binding, target.manifest.base)
        if target.manifest.facet_id != draft_manifest.facet_id:
            raise CompatibilityError(
                f"resumeFrom contains facet {target.manifest.facet_id!r}, expected "
                f"{draft_manifest.facet_id!r}"
            )
        if target.manifest.version != draft_manifest.version:
            raise CompatibilityError(
                f"resumeFrom version {target.manifest.version!r} differs from configured "
                f"{draft_manifest.version!r}"
            )
        if not _same_slot_layout(target.manifest, draft_manifest):
            raise CompatibilityError("resumeFrom slot/block structure differs from the training config")
        if target.manifest.dependencies != draft_manifest.dependencies:
            raise CompatibilityError("resumeFrom dependencies differ from the training config")
        target.manifest = replace(
            target.manifest,
            status="training",
            description=draft_manifest.description,
            metadata={**draft_manifest.metadata, **target.manifest.metadata, "resumedAt": utc_now()},
        )
        self._emit("facet-resumed", facet=target.manifest.reference(), path=str(resume))
        return target

    def _prepare_composition(
        self,
        base: LoadedBaseModel,
        target: LoadedFacet,
    ) -> tuple[CompositionPlan, dict[str, LoadedFacet]]:
        registry = FacetRegistry(self.config.registry)

        def load_manifest(reference: FacetReference) -> FacetManifest:
            if reference.facet_id == target.manifest.facet_id:
                if reference.version is not None and reference.version != target.manifest.version:
                    raise CompatibilityError(
                        f"Training target was referenced with version {reference.version}, but the "
                        f"configured version is {target.manifest.version}"
                    )
                return target.manifest
            return registry.get_manifest(reference)

        plan, manifests = CompositionResolver(load_manifest).resolve([target.manifest.reference()])
        parents: dict[str, LoadedFacet] = {}
        for manifest in manifests.values():
            if manifest.facet_id == target.manifest.facet_id:
                continue
            validate_base_binding(base.binding, manifest.base)
            parents[manifest.facet_id] = load_facet(registry.manifest_path(manifest), base.stack.layers)
        self._emit(
            "composition-resolved",
            target=target.manifest.reference(),
            parents=sorted(facet.manifest.reference() for facet in parents.values()),
            composition=plan.to_dict(),
        )
        return plan, parents

    @staticmethod
    def _move_facet_to_base(facet: LoadedFacet, base: LoadedBaseModel) -> None:
        for slot in facet.manifest.slots:
            for block_index, spec in enumerate(slot.blocks):
                source = base.stack.layers[spec.source_layer]
                device, dtype = module_device_and_dtype(source)
                facet.block(slot.slot_id, block_index).to(device=device, dtype=dtype)

    @staticmethod
    def _freeze_parameters(
        base: LoadedBaseModel,
        target: LoadedFacet,
        parents: Iterable[LoadedFacet],
    ) -> None:
        for parameter in base.model.parameters():
            parameter.requires_grad_(False)
        for parent in parents:
            for parameter in parent.parameters():
                parameter.requires_grad_(False)
        for parameter in target.parameters():
            parameter.requires_grad_(True)

    def _identity_check(
        self,
        dataset: FacetTokenDataset,
        collator: CausalCollator,
    ) -> dict[str, Any]:
        if not self.config.loop.identity_check:
            return {"enabled": False, "passed": None}
        if self._resumed:
            return {"enabled": True, "passed": None, "skipped": "resumed-trained-weights"}
        assert self.base is not None and self.target is not None and self.plan is not None
        torch = require_torch()
        batch = collator([dataset[0]])
        inputs = {
            "input_ids": batch["input_ids"].to(self.base.device),
            "attention_mask": batch["attention_mask"].to(self.base.device),
            "use_cache": False,
        }
        parent_plan = CompositionPlan(
            requested=tuple(
                item for item in self.plan.requested if not item.startswith(self.target.manifest.facet_id + "@")
            ),
            resolved=tuple(
                item for item in self.plan.resolved if not item.startswith(self.target.manifest.facet_id + "@")
            ),
            blocks=tuple(
                block for block in self.plan.blocks if block.facet != self.target.manifest.facet_id
            ),
        )
        previous_training = self.base.model.training
        self.base.model.eval()
        with torch.inference_mode():
            with ActivatedComposition(self.base, parent_plan, self._provide_block):
                baseline = self.base.model(**inputs).logits.detach().float().cpu()
            with ActivatedComposition(self.base, self.plan, self._provide_block):
                expanded = self.base.model(**inputs).logits.detach().float().cpu()
        self.base.model.train(previous_training)
        difference = (baseline - expanded).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
        atol = self.config.loop.identity_atol
        if atol is None:
            first_parameter = next(iter(self.target.parameters()), None)
            dtype = first_parameter.dtype if first_parameter is not None else torch.float32
            atol = 1e-5 if dtype == torch.float32 else 5e-3
        passed = bool(torch.allclose(baseline, expanded, atol=atol, rtol=self.config.loop.identity_rtol))
        report = {
            "enabled": True,
            "passed": passed,
            "maxAbsoluteLogitDifference": maximum,
            "meanAbsoluteLogitDifference": mean,
            "atol": atol,
            "rtol": self.config.loop.identity_rtol,
        }
        self._emit("identity-checked", **report)
        if not passed:
            raise CompatibilityError(
                "Identity initialization check failed: maximum absolute logit difference "
                f"{maximum:.6g} exceeds tolerances atol={atol}, rtol={self.config.loop.identity_rtol}. "
                "Review facet.module identityTargets or disable the check explicitly for non-identity experiments."
            )
        return report

    def _train_loop(
        self,
        *,
        train_loader: Any,
        validation_loader: Any,
        optimizer: Any,
        scheduler: Any,
        total_steps: int,
    ) -> dict[str, Any]:
        assert self.base is not None and self.target is not None and self.plan is not None
        torch = require_torch()
        model = self.base.model
        loop = self.config.loop
        previous_use_cache = getattr(model.config, "use_cache", None)
        gradient_checkpointing_enabled = False
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if loop.gradient_checkpointing:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if not callable(enable):
                raise ConfigurationError("This base model does not expose gradient_checkpointing_enable")
            try:
                enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable()
            input_grads = getattr(model, "enable_input_require_grads", None)
            if callable(input_grads):
                input_grads()
            gradient_checkpointing_enabled = True

        scaler = self._grad_scaler()
        running_loss = 0.0
        running_loss_count = 0
        recorded_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        epoch = 0
        validation_loss: float | None = None
        model.train()
        try:
            with ActivatedComposition(self.base, self.plan, self._provide_block):
                while self._global_step < total_steps:
                    epoch += 1
                    for batch_index, batch in enumerate(train_loader):
                        self._check_cancelled()
                        self._micro_step += 1
                        accumulated += 1
                        inputs = self._move_batch(batch)
                        with self._autocast_context():
                            outputs = model(**inputs, use_cache=False)
                            loss = outputs.loss
                        if not torch.isfinite(loss.detach()):
                            raise CompatibilityError(
                                f"Non-finite loss at optimizer step {self._global_step + 1}: {loss.item()}"
                            )
                        loss_value = float(loss.detach().float().item())
                        running_loss += loss_value
                        running_loss_count += 1
                        recorded_losses.append(loss_value)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        is_last_batch = batch_index + 1 == len(train_loader)
                        if accumulated < loop.gradient_accumulation_steps and not is_last_batch:
                            continue
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        for parameter in self.target.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(accumulated)
                        gradient_norm = None
                        if loop.max_grad_norm > 0:
                            gradient_norm = float(
                                torch.nn.utils.clip_grad_norm_(
                                    list(self.target.parameters()), loop.max_grad_norm
                                ).detach().float().item()
                            )
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated = 0
                        self._global_step += 1

                        if loop.log_every_steps and self._global_step % loop.log_every_steps == 0:
                            self._emit(
                                "training-step",
                                step=self._global_step,
                                totalSteps=total_steps,
                                epoch=epoch,
                                loss=round(running_loss / max(1, running_loss_count), 8),
                                learningRate=float(optimizer.param_groups[0]["lr"]),
                                gradientNorm=gradient_norm,
                            )
                            running_loss = 0.0
                            running_loss_count = 0
                        if (
                            validation_loader is not None
                            and loop.evaluate_every_steps
                            and self._global_step % loop.evaluate_every_steps == 0
                        ):
                            validation_loss = self._evaluate(validation_loader)
                            self._emit(
                                "validation",
                                step=self._global_step,
                                validationLoss=validation_loss,
                                perplexity=_safe_perplexity(validation_loss),
                            )
                        if (
                            self.config.checkpoint.save_every_steps
                            and self._global_step % self.config.checkpoint.save_every_steps == 0
                        ):
                            self._save_checkpoint(optimizer, scheduler, recorded_losses)
                        if self._global_step >= total_steps:
                            break
                if validation_loader is not None:
                    validation_loss = self._evaluate(validation_loader)
        finally:
            if gradient_checkpointing_enabled:
                disable = getattr(model, "gradient_checkpointing_disable", None)
                if callable(disable):
                    disable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = previous_use_cache

        mean_loss = sum(recorded_losses) / max(1, len(recorded_losses))
        return {
            "globalStep": self._global_step,
            "microStep": self._micro_step,
            "epochsTraversed": epoch,
            "meanTrainingLoss": mean_loss,
            "lastTrainingLoss": recorded_losses[-1] if recorded_losses else None,
            "validationLoss": validation_loss,
            "validationPerplexity": _safe_perplexity(validation_loss),
        }

    def _evaluate(self, loader: Any) -> float:
        assert self.base is not None
        if loader is None:
            return float("nan")
        torch = require_torch()
        model = self.base.model
        previous_training = model.training
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                self._check_cancelled()
                with self._autocast_context():
                    output = model(**self._move_batch(batch), use_cache=False)
                losses.append(float(output.loss.detach().float().item()))
        model.train(previous_training)
        return sum(losses) / max(1, len(losses))

    def _provide_block(self, planned: PlannedBlock) -> Any:
        assert self.target is not None
        if planned.facet == self.target.manifest.facet_id:
            return self.target.block(planned.slot_id, planned.block_index)
        try:
            return self.parents[planned.facet].block(planned.slot_id, planned.block_index)
        except KeyError as exc:
            raise CompatibilityError(f"No loaded block provider for parent facet {planned.facet}") from exc

    def _data_loader(self, dataset: Any, collator: CausalCollator, *, shuffle: bool) -> Any:
        torch = require_torch()
        generator = torch.Generator()
        generator.manual_seed(self.config.loop.seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.loop.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            collate_fn=collator,
            generator=generator,
            pin_memory=self.base is not None and self.base.device.type == "cuda",
        )

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert self.base is not None
        return {
            name: value.to(self.base.device)
            for name, value in batch.items()
            if name in {"input_ids", "attention_mask", "labels"}
        }

    def _autocast_context(self) -> Any:
        assert self.base is not None
        torch = require_torch()
        precision = self.config.loop.precision
        if precision == "float32":
            return nullcontext()
        if self.base.device.type not in {"cuda", "cpu"}:
            raise ConfigurationError(
                f"Automatic mixed precision {precision} is not supported by MI on {self.base.device.type}"
            )
        if precision == "float16" and self.base.device.type == "cpu":
            raise ConfigurationError("float16 training requires CUDA; use bfloat16 or float32 on CPU")
        dtype = torch.float16 if precision == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.base.device.type, dtype=dtype)

    def _grad_scaler(self) -> Any | None:
        assert self.base is not None
        if self.config.loop.precision != "float16" or self.base.device.type != "cuda":
            return None
        torch = require_torch()
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    def _restore_training_state(self, optimizer: Any, scheduler: Any) -> bool:
        resume = self.config.checkpoint.resume_from
        if resume is None:
            return False
        state_path = (resume if resume.is_dir() else resume.parent) / "trainer-state.pt"
        if not state_path.exists():
            return False
        torch = require_torch()
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        if not isinstance(state, dict):
            raise CompatibilityError(f"Invalid trainer state: {state_path}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        self._global_step = int(state.get("globalStep", 0))
        self._micro_step = int(state.get("microStep", 0))
        self._emit("optimizer-resumed", path=str(state_path), globalStep=self._global_step)
        return True

    def _save_checkpoint(
        self,
        optimizer: Any,
        scheduler: Any,
        losses: list[float],
    ) -> None:
        assert self.target is not None
        directory = (
            self.config.checkpoint.output_directory
            / "checkpoints"
            / f"step-{self._global_step:08d}"
        )
        self.target.manifest = replace(
            self.target.manifest,
            status="training",
            metadata={
                **self.target.manifest.metadata,
                "updatedAt": utc_now(),
                "globalStep": self._global_step,
            },
        )
        save_facet(self.target, directory, overwrite=True)
        if self.config.checkpoint.save_optimizer:
            self._save_training_state(directory, optimizer, scheduler, losses)
        self._emit("checkpoint-saved", step=self._global_step, path=str(directory))

    def _save_final(self, optimizer: Any, scheduler: Any, metrics: dict[str, Any]) -> FacetManifest:
        assert self.target is not None
        self.target.manifest = replace(
            self.target.manifest,
            status="trained",
            metadata={
                **self.target.manifest.metadata,
                "completedAt": utc_now(),
                "globalStep": self._global_step,
                "metrics": metrics,
                "parentsUsedDuringTraining": sorted(
                    facet.manifest.reference() for facet in self.parents.values()
                ),
            },
        )
        manifest = save_facet(
            self.target,
            self.config.checkpoint.output_directory,
            overwrite=True,
        )
        if self.config.checkpoint.save_optimizer:
            self._save_training_state(
                self.config.checkpoint.output_directory,
                optimizer,
                scheduler,
                [],
            )
        atomic_write_json(
            self.config.checkpoint.output_directory / "training-result.json",
            {
                "schemaVersion": "mi-training-result-v1",
                "facet": manifest.reference(),
                "status": "succeeded",
                "metrics": metrics,
            },
        )
        return manifest

    def _save_training_state(
        self,
        directory: Path,
        optimizer: Any,
        scheduler: Any,
        losses: list[float],
    ) -> None:
        torch = require_torch()
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "schemaVersion": "mi-trainer-state-v1",
            "globalStep": self._global_step,
            "microStep": self._micro_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "recentLosses": losses[-100:],
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=".trainer-state.", suffix=".tmp", dir=directory
        )
        os.close(fd)
        try:
            torch.save(state, temporary_name)
            os.replace(temporary_name, directory / "trainer-state.pt")
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _emit(self, event: str, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "schemaVersion": "mi-training-event-v1",
                    "event": event,
                    "time": utc_now(),
                    **payload,
                }
            )

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TrainingCancelled()

    @staticmethod
    def _pad_token_id(tokenizer: Any) -> int:
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        if pad is None:
            raise ConfigurationError("Tokenizer has neither pad_token_id nor eos_token_id")
        return int(pad)

    @staticmethod
    def _initialization_report(facet: LoadedFacet) -> dict[str, Any]:
        return {
            slot: [
                {
                    "initializer": report.initializer,
                    "zeroedParameters": list(report.zeroed_parameters),
                }
                for report in reports
            ]
            for slot, reports in facet.initialization.items()
        }

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        torch = require_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def train_from_config(
    source: TrainingConfig | str | Path | dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    config = source if isinstance(source, TrainingConfig) else TrainingConfig.load(source)
    if config.algorithm.trainer_class_path:
        trainer_class = import_symbol(config.algorithm.trainer_class_path)
        trainer = trainer_class(
            config,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            **config.algorithm.kwargs,
        )
        run = getattr(trainer, "run", None)
        if not callable(run):
            raise ConfigurationError(
                f"Custom trainer {config.algorithm.trainer_class_path!r} has no callable run()"
            )
        return run()
    return MixedInferenceTrainer(
        config,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    ).run()


def _safe_perplexity(loss: float | None) -> float | None:
    if loss is None or not math.isfinite(loss):
        return None
    return math.exp(min(loss, 80.0))


def _same_slot_layout(left: FacetManifest, right: FacetManifest) -> bool:
    def normalized(manifest: FacetManifest) -> tuple[Any, ...]:
        return tuple(
            (
                slot.slot_id,
                slot.after_layer,
                slot.order,
                tuple(
                    (block.source_layer, block.initializer, block.identity_targets)
                    for block in slot.blocks
                ),
            )
            for slot in manifest.slots
        )

    return normalized(left) == normalized(right)
