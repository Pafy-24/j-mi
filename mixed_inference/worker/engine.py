from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from safetensors.torch import load_file, save_file

from ..core.errors import CompatibilityError, ConfigurationError, TrainingCancelled
from ..core.modeling import ActivatedComposition, load_base_model, require_torch
from ..core.util import sha256_file, sha256_json
from ..training.data import CausalCollator, build_datasets
from ..training.trainer import MixedInferenceTrainer


@dataclass(frozen=True)
class GradientResult:
    gradient_path: Path
    gradient_sha256: str
    supervised_tokens: int
    examples: int
    mean_loss: float
    tensor_count: int
    tensor_specs: dict[str, dict[str, Any]]


class FacetGradientEngine:
    """Compute child-facet gradients without owning optimizer state.

    The preparation helpers are shared with the proven single-process trainer
    so base binding, dependency expansion and identity initialization keep the
    same semantics. Only the child facet has ``requires_grad=True``.
    """

    def __init__(
        self,
        training_config: dict[str, Any] | str | Path,
        *,
        cancel_event: threading.Event | None = None,
    ):
        self.cancel_event = cancel_event or threading.Event()
        self.helper = MixedInferenceTrainer(
            training_config, cancel_event=self.cancel_event
        )
        self.config = self.helper.config
        self.helper._seed_everything(self.config.loop.seed)
        self.base = load_base_model(self.config.base_model, load_tokenizer=True)
        self.helper.base = self.base
        if self.base.tokenizer is None:
            raise ConfigurationError("MI worker requires a tokenizer")
        self.dataset, self.validation_dataset = build_datasets(
            self.base.tokenizer, self.config.data
        )
        self.target = self.helper._prepare_target(self.base)
        self.plan, self.parents = self.helper._prepare_composition(
            self.base, self.target
        )
        self.helper.target = self.target
        self.helper.plan = self.plan
        self.helper.parents = self.parents
        self.helper._move_facet_to_base(self.target, self.base)
        self.helper._freeze_parameters(
            self.base, self.target, self.parents.values()
        )
        self.collator = CausalCollator(
            self.helper._pad_token_id(self.base.tokenizer)
        )
        self._compute_lock = threading.Lock()

    def summary(self) -> dict[str, Any]:
        trainable = sum(parameter.numel() for parameter in self.target.parameters())
        return {
            "base": self.base.summary(),
            "facet": self.target.manifest.to_dict(),
            "composition": self.plan.to_dict(),
            "datasetExamples": len(self.dataset),
            "validationExamples": len(self.validation_dataset)
            if self.validation_dataset
            else 0,
            "trainableParameters": trainable,
            "tensorSpecs": self.tensor_specs(),
        }

    def tensor_specs(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "numel": parameter.numel(),
            }
            for name, parameter in self.target.named_parameters()
        }

    def save_weights(self, path: Path, *, dtype: str = "float32") -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._weight_state(dtype=dtype)
        save_file(state, str(path))
        return sha256_file(path)

    def load_weights(self, path: Path) -> None:
        state = load_file(str(path), device="cpu")
        parameters = dict(self.target.named_parameters())
        self._validate_tensor_set(state, parameters, kind="weight")
        torch = require_torch()
        non_finite = [name for name, tensor in state.items() if not torch.isfinite(tensor).all()]
        if non_finite:
            raise CompatibilityError(
                f"weight tensors contain NaN or Inf: {non_finite[:5]}"
            )
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(
                    state[name].to(device=parameter.device, dtype=parameter.dtype)
                )

    def compute(
        self,
        output_path: Path,
        *,
        indices: list[int] | None = None,
        offset: int = 0,
        count: int | None = None,
        batch_size: int | None = None,
        transport_dtype: str = "float32",
    ) -> GradientResult:
        with self._compute_lock:
            self._check_cancelled()
            selected = self._selected_indices(indices, offset=offset, count=count)
            if not selected:
                raise ConfigurationError("A worker round must contain at least one example")
            batch_size = batch_size or self.config.loop.batch_size
            if batch_size <= 0:
                raise ConfigurationError("batchSize must be positive")
            torch = require_torch()
            model = self.base.model
            model.train()
            for parameter in self.target.parameters():
                parameter.grad = None
            losses: list[tuple[float, int]] = []
            supervised_total = 0
            previous_use_cache = getattr(model.config, "use_cache", None)
            gradient_checkpointing_enabled = False
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if self.config.loop.gradient_checkpointing:
                enable = getattr(model, "gradient_checkpointing_enable", None)
                if not callable(enable):
                    raise ConfigurationError(
                        "This base model does not expose gradient_checkpointing_enable"
                    )
                try:
                    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    enable()
                input_grads = getattr(model, "enable_input_require_grads", None)
                if callable(input_grads):
                    input_grads()
                gradient_checkpointing_enabled = True
            loss_scale = 65536.0 if self.config.loop.precision == "float16" else 1.0
            try:
                with ActivatedComposition(
                    self.base, self.plan, self.helper._provide_block
                ):
                    for start in range(0, len(selected), batch_size):
                        self._check_cancelled()
                        items = [
                            self.dataset[index]
                            for index in selected[start : start + batch_size]
                        ]
                        batch = self.collator(items)
                        inputs = self.helper._move_batch(batch)
                        supervised = int(
                            inputs["labels"][:, 1:].ne(-100).sum().detach().item()
                        )
                        if supervised <= 0:
                            raise CompatibilityError(
                                "Round batch has no supervised causal-LM tokens"
                            )
                        with self.helper._autocast_context():
                            output = model(**inputs, use_cache=False)
                            loss = output.loss
                        if not torch.isfinite(loss.detach()):
                            raise CompatibilityError("Worker produced a non-finite loss")
                        (loss * supervised * loss_scale).backward()
                        losses.append((float(loss.detach().float().item()), supervised))
                        supervised_total += supervised
            finally:
                if gradient_checkpointing_enabled:
                    disable = getattr(model, "gradient_checkpointing_disable", None)
                    if callable(disable):
                        disable()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = previous_use_cache

            gradients: dict[str, Any] = {}
            specs: dict[str, dict[str, Any]] = {}
            divisor = float(supervised_total) * loss_scale
            for name, parameter in self.target.named_parameters():
                if parameter.grad is None:
                    raise CompatibilityError(
                        f"Child tensor {name} received no gradient"
                    )
                gradient = parameter.grad.detach().float().div(divisor)
                if not torch.isfinite(gradient).all():
                    raise CompatibilityError(
                        f"Child tensor {name} contains NaN or Inf gradients"
                    )
                if transport_dtype == "float16":
                    gradient = gradient.to(torch.float16)
                elif transport_dtype == "bfloat16":
                    gradient = gradient.to(torch.bfloat16)
                elif transport_dtype != "float32":
                    raise ConfigurationError(
                        f"Unsupported gradient transport dtype: {transport_dtype}"
                    )
                gradient = gradient.cpu().contiguous()
                gradients[name] = gradient
                specs[name] = {
                    "shape": list(gradient.shape),
                    "dtype": str(gradient.dtype).removeprefix("torch."),
                    "numel": gradient.numel(),
                }
                parameter.grad = None
            self._assert_frozen_gradients()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                gradients,
                str(output_path),
                metadata={
                    "schemaVersion": "mi-gradient-v1",
                    "supervisedTokens": str(supervised_total),
                    "examples": str(len(selected)),
                    "tensorSetSha256": sha256_json(specs),
                },
            )
            weighted_loss = sum(loss * tokens for loss, tokens in losses) / max(
                1, supervised_total
            )
            return GradientResult(
                gradient_path=output_path,
                gradient_sha256=sha256_file(output_path),
                supervised_tokens=supervised_total,
                examples=len(selected),
                mean_loss=weighted_loss,
                tensor_count=len(gradients),
                tensor_specs=specs,
            )

    def _weight_state(self, *, dtype: str) -> dict[str, Any]:
        torch = require_torch()
        target_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype)
        if target_dtype is None:
            raise ConfigurationError(f"Unsupported weight transport dtype: {dtype}")
        return {
            name: parameter.detach().to(device="cpu", dtype=target_dtype).contiguous()
            for name, parameter in self.target.named_parameters()
        }

    @staticmethod
    def _validate_tensor_set(
        state: dict[str, Any], parameters: dict[str, Any], *, kind: str
    ) -> None:
        missing = sorted(set(parameters).difference(state))
        unexpected = sorted(set(state).difference(parameters))
        if missing or unexpected:
            raise CompatibilityError(
                f"{kind} tensor names differ: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for name, parameter in parameters.items():
            if tuple(state[name].shape) != tuple(parameter.shape):
                raise CompatibilityError(
                    f"{kind} tensor {name} shape {tuple(state[name].shape)} does not match {tuple(parameter.shape)}"
                )

    def _selected_indices(
        self, indices: list[int] | None, *, offset: int, count: int | None
    ) -> list[int]:
        selected = (
            list(indices)
            if indices is not None
            else list(range(offset, min(len(self.dataset), offset + count)))
            if count is not None
            else list(range(offset, len(self.dataset)))
        )
        invalid = [index for index in selected if index >= len(self.dataset)]
        if invalid:
            raise ConfigurationError(
                f"Dataset indices exceed shard size {len(self.dataset)}: {invalid[:5]}"
            )
        return selected

    def _assert_frozen_gradients(self) -> None:
        leaked = [
            name
            for name, parameter in self.base.model.named_parameters()
            if parameter.grad is not None
        ]
        for parent_id, facet in self.parents.items():
            leaked.extend(
                f"parent:{parent_id}:{name}"
                for name, parameter in facet.named_parameters()
                if parameter.grad is not None
            )
        if leaked:
            raise CompatibilityError(
                f"Frozen base/parent tensors received gradients: {leaked[:5]}"
            )

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TrainingCancelled()
