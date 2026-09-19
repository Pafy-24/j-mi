from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import CompatibilityError, CompositionError, ConfigurationError
from .graph import CompositionPlan, PlannedBlock
from .identity import tokenizer_identity, verify_local_artifact
from .specs import BaseBinding, BaseModelRequest
from .util import sha256_json
from ..rm_integration import apply_memory_budget, managed_device


@dataclass
class LayerStackHandle:
    path: str
    owner: Any
    attribute: str
    layers: Any

    def replace(self, layers: Any) -> None:
        setattr(self.owner, self.attribute, layers)
        self.layers = layers


@dataclass
class LoadedBaseModel:
    request: BaseModelRequest
    model: Any
    tokenizer: Any
    stack: LayerStackHandle
    binding: BaseBinding
    device: Any

    def summary(self) -> dict[str, Any]:
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "modelId": self.binding.model_id,
            "revision": self.binding.revision,
            "modelType": self.binding.model_type,
            "configSha256": self.binding.config_sha256,
            "artifactSha256": self.binding.artifact_sha256,
            "artifactIdentityStatus": self.binding.artifact_identity_status,
            "tokenizerSha256": self.binding.tokenizer_sha256,
            "tokenizerIdentityStatus": self.binding.tokenizer_identity_status,
            "comparisonEligible": self.binding.artifact_identity_status == "verified"
            and self.binding.tokenizer_identity_status == "verified",
            "layerPath": self.binding.layer_path,
            "baseLayerCount": self.binding.base_layer_count,
            "device": str(self.device),
            "parameters": {"total": total, "trainable": trainable},
        }


def require_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ConfigurationError("PyTorch is required for Mixed Inference") from exc


def resolve_dtype(name: str, torch: Any) -> Any:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name == "auto":
        return None
    return mapping[name]


def resolve_device(name: str, torch: Any) -> Any:
    name = managed_device(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        candidate = getattr(torch.backends, "mps", None)
        if candidate is not None and candidate.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid baseModel.device {name!r}: {exc}") from exc


def sanitized_model_config(model: Any) -> dict[str, Any]:
    raw = dict(model.config.to_dict())
    for key in ("_name_or_path", "transformers_version", "torch_dtype", "dtype"):
        raw.pop(key, None)
    return raw


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Stable tokenizer compatibility digest without relying on its display name."""
    vocabulary = getattr(tokenizer, "get_vocab", lambda: {})()
    special = getattr(tokenizer, "special_tokens_map", {})
    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    for key in ("name_or_path", "_name_or_path"):
        init_kwargs.pop(key, None)
    return sha256_json(
        {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocabulary": _tokenizer_json(dict(sorted(dict(vocabulary).items()))),
            "specialTokens": _tokenizer_json(special),
            "initKwargs": _tokenizer_json(init_kwargs),
        }
    )


def _tokenizer_json(value: Any) -> Any:
    """Convert tokenizer helper objects (notably AddedToken) to stable JSON."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _tokenizer_json(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_tokenizer_json(item) for item in value]
    state = getattr(value, "__getstate__", None)
    if callable(state):
        try:
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": _tokenizer_json(state()),
            }
        except (TypeError, ValueError):
            pass
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "value": str(value)}


def load_base_model(request: BaseModelRequest, *, load_tokenizer: bool = True) -> LoadedBaseModel:
    torch = require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ConfigurationError("transformers is required for Mixed Inference") from exc

    device = resolve_device(request.device, torch)
    apply_memory_budget(torch, device)
    dtype = resolve_dtype(request.dtype, torch)
    model_id = request.model_id
    local_model = Path(model_id).expanduser()
    artifact_identity = None
    if local_model.exists():
        local_model = local_model.resolve()
        if request.artifact_files:
            artifact_identity = verify_local_artifact(
                local_model,
                request.artifact_files,
                request.artifact_sha256,
                trust_remote_code=request.trust_remote_code,
                model_kwargs=request.model_kwargs,
                tokenizer_kwargs=request.tokenizer_kwargs,
            )
        model_id = str(local_model)
    if any(key in request.model_kwargs for key in ("device_map", "tp_plan")):
        raise ConfigurationError(
            "MI v1 requires a single-device base model; an RM multi-CUDA lease exposes "
            "devices but does not make device_map/tp_plan composition-safe"
        )
    common = {
        "revision": request.revision,
        "trust_remote_code": request.trust_remote_code,
        "local_files_only": not request.allow_download,
    }
    model_kwargs = {**request.model_kwargs, **common}
    if dtype is not None:
        model_kwargs.pop("torch_dtype", None)
        model_kwargs["dtype"] = dtype
    if device.type == "cuda":
        load_strategy = os.environ.get(
            "JESTER_MI_CUDA_LOAD_STRATEGY", "direct"
        ).strip().lower()
        if load_strategy not in {"direct", "host-staged"}:
            raise ConfigurationError(
                "JESTER_MI_CUDA_LOAD_STRATEGY must be direct or host-staged"
            )
        if load_strategy == "direct":
            try:
                import accelerate  # noqa: F401
            except ImportError as exc:
                raise ConfigurationError(
                    "Direct CUDA base-model loading requires accelerate"
                ) from exc
            # A one-entry map materializes weights directly on the only
            # RM-visible CUDA device and minimizes host commit. Operators can
            # explicitly select host-staged loading for runtimes where the
            # Accelerate direct loader is not stable; final placement is still
            # verified by model.to(device) below.
            model_kwargs["device_map"] = {"": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in model_kwargs:
            raise
        retry = dict(model_kwargs)
        retry["torch_dtype"] = retry.pop("dtype")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **retry)
        except Exception:
            raise exc
    except Exception as exc:
        mode = "online downloads enabled" if request.allow_download else "local files only"
        raise ConfigurationError(
            f"Cannot load base model {model_id!r} ({mode}): {exc}"
        ) from exc

    if bool(getattr(model.config, "is_encoder_decoder", False)):
        raise ConfigurationError(
            "MI v1 supports decoder-only causal language models, not encoder-decoder models"
        )
    model.to(device)
    tokenizer = None
    if load_tokenizer:
        tokenizer_kwargs = {**request.tokenizer_kwargs, **common}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        except Exception as exc:
            raise ConfigurationError(f"Cannot load tokenizer for {model_id!r}: {exc}") from exc
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    stack = locate_layer_stack(model, request.layer_path)
    model_type = str(getattr(model.config, "model_type", type(model).__name__))
    tokenizer_identity_value = tokenizer_identity(tokenizer) if tokenizer is not None else None
    binding = BaseBinding(
        # Runtime paths are host-local (and CAS checkout paths differ between
        # cluster and consumer devices). Persist the explicit portable
        # identity when one is supplied, while still loading from model_id.
        model_id=request.canonical_id or model_id,
        revision=request.revision,
        config_sha256=sha256_json(sanitized_model_config(model)),
        layer_path=stack.path,
        base_layer_count=len(stack.layers),
        model_type=model_type,
        artifact_sha256=artifact_identity.sha256 if artifact_identity is not None else "",
        tokenizer_sha256=tokenizer_identity_value.sha256 if tokenizer_identity_value is not None else "",
        artifact_identity_status=(
            artifact_identity.status if artifact_identity is not None else "unverified"
        ),
        tokenizer_identity_status=(
            tokenizer_identity_value.status if tokenizer_identity_value is not None else "unverified"
        ),
    )
    return LoadedBaseModel(
        request=request,
        model=model,
        tokenizer=tokenizer,
        stack=stack,
        binding=binding,
        device=device,
    )


def locate_layer_stack(model: Any, configured_path: str | None = None) -> LayerStackHandle:
    torch = require_torch()
    candidates = [configured_path] if configured_path else [
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.transformer.h",
        "decoder.layers",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            owner, attribute, value = resolve_attribute(model, path)
        except (AttributeError, IndexError):
            continue
        if isinstance(value, torch.nn.ModuleList) and len(value) > 0:
            return LayerStackHandle(path=path, owner=owner, attribute=attribute, layers=value)
    if configured_path:
        raise ConfigurationError(
            f"Configured layer path {configured_path!r} is not a non-empty torch.nn.ModuleList"
        )

    expected = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    discovered: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            if expected and len(module) != expected:
                continue
            discovered.append((name, module))
    if len(discovered) != 1:
        names = ", ".join(name or "<root>" for name, _ in discovered) or "none"
        raise ConfigurationError(
            "Cannot determine the decoder layer stack automatically. Set baseModel.layerPath. "
            f"Candidates matching num_hidden_layers: {names}"
        )
    path, value = discovered[0]
    owner, attribute, _ = resolve_attribute(model, path)
    return LayerStackHandle(path=path, owner=owner, attribute=attribute, layers=value)


def resolve_attribute(root: Any, path: str) -> tuple[Any, str, Any]:
    parts = path.split(".")
    owner = root
    for part in parts[:-1]:
        if part.isdigit():
            owner = owner[int(part)]
        else:
            owner = getattr(owner, part)
    attribute = parts[-1]
    value = owner[int(attribute)] if attribute.isdigit() else getattr(owner, attribute)
    if attribute.isdigit():
        raise AttributeError("A decoder stack path must end in an attribute, not an index")
    return owner, attribute, value


def module_device_and_dtype(module: Any) -> tuple[Any, Any]:
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    torch = require_torch()
    return torch.device("cpu"), torch.float32


class ActivatedComposition(AbstractContextManager["ActivatedComposition"]):
    def __init__(
        self,
        base: LoadedBaseModel,
        plan: CompositionPlan,
        block_provider: Callable[[PlannedBlock], Any],
    ):
        self.base = base
        self.plan = plan
        self.block_provider = block_provider
        self._original_layers: Any = None
        self._config_snapshots: list[tuple[Any, dict[str, Any]]] = []
        self._index_snapshots: list[tuple[Any, str, Any]] = []
        self._active = False

    def __enter__(self) -> "ActivatedComposition":
        if self._active:
            raise CompositionError("A composition context cannot be entered twice")
        torch = require_torch()
        stack = self.base.stack
        original = stack.layers
        base_count = self.base.binding.base_layer_count
        if len(original) != base_count:
            raise CompositionError(
                "The base layer stack is already modified; concurrent or nested compositions are not allowed"
            )
        injections: dict[int, list[PlannedBlock]] = {}
        for planned in self.plan.blocks:
            if planned.after_layer < -1 or planned.after_layer >= base_count:
                raise CompositionError(
                    f"Slot {planned.slot_id} targets afterLayer={planned.after_layer}, but the base has "
                    f"indices 0..{base_count - 1}"
                )
            if planned.source_layer >= base_count:
                raise CompositionError(
                    f"Facet {planned.facet} block sourceLayer={planned.source_layer} exceeds the base"
                )
            injections.setdefault(planned.after_layer, []).append(planned)

        composed: list[Any] = []
        source_indices: list[int] = []

        def append_injections(after_layer: int) -> None:
            for planned in injections.get(after_layer, []):
                block = self.block_provider(planned)
                source = original[planned.source_layer]
                device, dtype = module_device_and_dtype(source)
                block.to(device=device, dtype=dtype)
                block.train(self.base.model.training)
                composed.append(block)
                source_indices.append(planned.source_layer)

        append_injections(-1)
        for layer_index, layer in enumerate(original):
            composed.append(layer)
            source_indices.append(layer_index)
            append_injections(layer_index)

        self._original_layers = original
        new_layers = torch.nn.ModuleList(composed)
        try:
            self._snapshot_and_update_configs(source_indices)
            self._snapshot_and_reindex(composed)
            stack.replace(new_layers)
        except BaseException:
            if stack.layers is not original:
                stack.replace(original)
            self._restore_snapshots()
            self._original_layers = None
            raise
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._active:
            return None
        self.base.stack.replace(self._original_layers)
        self._restore_snapshots()
        self._active = False
        return None

    def _restore_snapshots(self) -> None:
        for owner, values in reversed(self._config_snapshots):
            for name, value in values.items():
                setattr(owner, name, value)
        for owner, name, value in reversed(self._index_snapshots):
            try:
                setattr(owner, name, value)
            except (AttributeError, RuntimeError):
                pass
        self._config_snapshots.clear()
        self._index_snapshots.clear()

    def _snapshot_and_update_configs(self, source_indices: list[int]) -> None:
        configs: list[Any] = [self.base.model.config]
        getter = getattr(self.base.model.config, "get_text_config", None)
        if callable(getter):
            try:
                text_config = getter()
                if text_config is not configs[0]:
                    configs.append(text_config)
            except (TypeError, AttributeError):
                pass
        base_count = self.base.binding.base_layer_count
        new_count = len(source_indices)
        for config in configs:
            snapshot: dict[str, Any] = {}
            for name in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(config, name) and int(getattr(config, name) or 0) == base_count:
                    snapshot[name] = getattr(config, name)
                    setattr(config, name, new_count)
            if hasattr(config, "layer_types"):
                layer_types = getattr(config, "layer_types")
                if isinstance(layer_types, (list, tuple)) and len(layer_types) == base_count:
                    snapshot["layer_types"] = layer_types
                    expanded = [layer_types[index] for index in source_indices]
                    setattr(config, "layer_types", type(layer_types)(expanded))
            if snapshot:
                self._config_snapshots.append((config, snapshot))

    def _snapshot_and_reindex(self, layers: list[Any]) -> None:
        seen: set[tuple[int, str]] = set()
        for layer_index, layer in enumerate(layers):
            for module in layer.modules():
                for name in ("layer_idx", "layer_index", "block_idx"):
                    if not hasattr(module, name):
                        continue
                    key = (id(module), name)
                    if key in seen:
                        continue
                    value = getattr(module, name)
                    if value is not None and not isinstance(value, int):
                        continue
                    try:
                        setattr(module, name, layer_index)
                    except (AttributeError, RuntimeError):
                        continue
                    self._index_snapshots.append((module, name, value))
                    seen.add(key)


def validate_base_binding(
    actual: BaseBinding,
    expected: BaseBinding,
    *,
    allow_model_id_mismatch: bool = False,
) -> None:
    problems: list[str] = []
    if not allow_model_id_mismatch and actual.model_id != expected.model_id:
        problems.append(f"modelId {actual.model_id!r} != {expected.model_id!r}")
    if expected.revision and actual.revision != expected.revision:
        problems.append(f"revision {actual.revision!r} != {expected.revision!r}")
    if actual.config_sha256 != expected.config_sha256:
        problems.append("model config hash differs")
    if expected.artifact_sha256 and (
        actual.artifact_identity_status != "verified"
        or actual.artifact_sha256 != expected.artifact_sha256
    ):
        problems.append("base artifact hash differs")
    if expected.tokenizer_sha256 and (
        actual.tokenizer_identity_status != "verified"
        or actual.tokenizer_sha256 != expected.tokenizer_sha256
    ):
        problems.append("tokenizer hash differs")
    if actual.layer_path != expected.layer_path:
        problems.append(f"layerPath {actual.layer_path!r} != {expected.layer_path!r}")
    if actual.base_layer_count != expected.base_layer_count:
        problems.append(
            f"base layer count {actual.base_layer_count} != {expected.base_layer_count}"
        )
    if problems:
        raise CompatibilityError("Facet/base incompatibility: " + "; ".join(problems))
