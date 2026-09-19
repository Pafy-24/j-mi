from __future__ import annotations

import gc
import inspect
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..core.errors import CompositionError, ConfigurationError, MIError
from ..core.graph import CompositionPlan, PlannedBlock
from ..core.identity import verify_local_artifact
from ..core.modeling import (
    ActivatedComposition,
    LoadedBaseModel,
    load_base_model,
    require_torch,
    resolve_device,
    resolve_dtype,
)
from ..core.residency import Footprint, ResidencyBudget, ResidencyManager
from ..core.registry import FacetRegistry
from ..core.specs import BaseModelRequest, GENERATION_SCHEMA_VERSION
from ..core.tensor_metadata import safetensors_tensor_bytes
from ..rm_integration import residency_budget


class MixedInferenceRuntime:
    """Single-base runtime whose process placement is owned by RM.

    The local lock protects reversible decoder-list mutations.  It is not a
    hardware scheduler and never launches training in a managed serving process.
    """

    def __init__(self, registry: str | Path, *, residency: ResidencyManager | None = None):
        self.registry = FacetRegistry(registry)
        budget = residency_budget()
        self.residency = residency or ResidencyManager(
            ResidencyBudget(
                ram_bytes=budget.get("ramBytes"),
                accelerator_bytes=budget.get("acceleratorBytes"),
                operational_margin_bytes=budget.get("operationalMarginBytes", 0),
            )
        )
        self.base: LoadedBaseModel | None = None
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._generation_count = 0

    def load_base(self, request: BaseModelRequest | dict[str, Any]) -> dict[str, Any]:
        spec = request if isinstance(request, BaseModelRequest) else BaseModelRequest.from_dict(request)
        with self._lock:
            # A replacement coexists with the healthy runtime while loading.
            # Reserve its local artifact bytes first; bounded deployments fail
            # before construction when the source cannot be sized.
            torch = require_torch()
            target_device = resolve_device(spec.device, torch)
            target_element_bytes = _requested_element_bytes(
                spec, torch, bounded=_has_bounded_budget(self.residency)
            )
            local_source = Path(spec.model_id).expanduser()
            if local_source.is_dir() and spec.artifact_files:
                # Verify loader-input closure before the staging calculation or
                # constructor can allocate unlisted shards/code.
                verify_local_artifact(
                    local_source,
                    spec.artifact_files,
                    spec.artifact_sha256,
                    trust_remote_code=spec.trust_remote_code,
                    model_kwargs=spec.model_kwargs,
                    tokenizer_kwargs=spec.tokenizer_kwargs,
                )
            self.residency.preflight_staging([
                _base_staging_footprint(
                    spec,
                    target_device=target_device,
                    target_element_bytes=target_element_bytes,
                    bounded=_has_bounded_budget(self.residency),
                )
            ])
            candidate: LoadedBaseModel | None = None
            try:
                candidate = load_base_model(spec, load_tokenizer=True)
                footprint = _module_footprint(candidate.model, candidate.device)
                # Reconcile the preflight estimate with actual module bytes
                # while the old base/cache is still retained.
                self.residency.preflight_staging([footprint])
                candidate_residency = ResidencyManager(self.residency.budget)
                candidate_residency.reset(base=footprint)
                # Parse and validate registry changes before replacing a usable
                # base. A refresh failure leaves the existing runtime intact.
                self.registry.refresh()
            except BaseException:
                del candidate
                gc.collect()
                raise
            generation_config = getattr(candidate.model, "generation_config", None)
            if generation_config is not None and hasattr(generation_config, "max_length"):
                # MI always supplies max_new_tokens (128 by default). Recent
                # Transformers versions otherwise treat a model-provided
                # max_length as a second explicit limit and log a warning for
                # every request, even though max_new_tokens takes precedence.
                generation_config.max_length = None
            self.registry.unload(all_facets=True)
            self.base = candidate
            self.residency = candidate_residency
            return self.base.summary()

    def unload_base(self) -> dict[str, Any]:
        with self._lock:
            previous = self.base.binding.model_id if self.base is not None else None
            self.registry.unload(all_facets=True)
            self.base = None
            self.residency.reset()
            gc.collect()
            torch = require_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"unloaded": previous}

    def plan(self, requested_facets: Iterable[str]) -> dict[str, Any]:
        with self._lock:
            plan, _ = self.registry.resolve(requested_facets)
            result = plan.to_dict()
            if self.base is not None:
                self._validate_plan_for_base(plan)
                result["base"] = self.base.binding.to_dict()
                result["provenance"] = self._identity_provenance()
                result["composedLayerCount"] = (
                    self.base.binding.base_layer_count + len(plan.blocks)
                )
            return result

    def list_facets(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.registry.list()

    def refresh_registry(self) -> dict[str, Any]:
        with self._lock:
            self.residency.discard(self.registry.refresh())
            return {"facets": self.registry.list()}

    def load_facets(
        self,
        requested_facets: Iterable[str],
        *,
        allow_model_id_mismatch: bool = False,
        allow_unsafe_legacy_base_binding: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            base = self._require_base()
            resolved = self.registry.resolve(requested_facets)
            plan, manifests = resolved
            self._validate_plan_for_base(plan)
            self._preflight_facet_staging(base, manifests)
            admission = None

            def admit(loaded_facets: dict[str, Any]) -> None:
                nonlocal admission
                requested_footprints = {
                    self.registry.cache_key(base, facet): _facet_footprint(facet, base.device)
                    for facet in loaded_facets.values()
                }
                admission = self.residency.admit(requested_footprints)
                self.registry.evict_cache_keys(admission.evicted)

            try:
                plan, _ = self.registry.load_resolved(
                    requested_facets,
                    base,
                    allow_model_id_mismatch=allow_model_id_mismatch,
                    allow_unsafe_legacy_base_binding=allow_unsafe_legacy_base_binding,
                    before_commit=admit,
                    resolved=resolved,
                )
                return {
                    "loadedFacets": self.registry.loaded(),
                    "composition": plan.to_dict(),
                    "provenance": self._identity_provenance(
                        unsafe_legacy_base_binding=allow_model_id_mismatch
                    ),
                }
            finally:
                if admission is not None:
                    self.residency.release(admission)

    def unload_facets(
        self,
        facets: Iterable[str] | None = None,
        *,
        all_facets: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            cache_keys = self.registry.cache_keys_for(facets, all_facets=all_facets)
            removed = self.registry.unload(facets, all_facets=all_facets)
            self.residency.discard(cache_keys)
            gc.collect()
            torch = require_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"unloadedFacets": removed, "loadedFacets": self.registry.loaded()}

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ConfigurationError("Generate request must be a JSON object")
        with self._lock:
            started = time.perf_counter()
            base = self._require_base()
            inputs = self._prepare_inputs(request)
            requested = extract_facet_references(request)
            allow_mismatch = bool(request.get("allowModelIdMismatch", False))
            if allow_mismatch and not bool(request.get("allowUnsafeLegacyBaseBinding", False)):
                raise ConfigurationError(
                    "allowModelIdMismatch is an unsafe legacy compatibility bypass; "
                    "set allowUnsafeLegacyBaseBinding=true explicitly and exclude this request from comparative results"
                )
            admission = None
            previous_training = None
            composition_blocks = 0

            def admit(loaded_facets: dict[str, Any]) -> None:
                nonlocal admission
                requested_footprints = {
                    self.registry.cache_key(base, facet): _facet_footprint(facet, base.device)
                    for facet in loaded_facets.values()
                }
                admission = self.residency.admit(
                    requested_footprints,
                    kv_cache=_estimate_request_footprint(
                        base,
                        inputs,
                        request,
                        composition_blocks=composition_blocks,
                        bounded=_has_bounded_budget(self.residency),
                    ),
                )
                self.registry.evict_cache_keys(admission.evicted)

            try:
                resolved = self.registry.resolve(requested)
                plan, manifests = resolved
                self._validate_plan_for_base(plan)
                composition_blocks = len(plan.blocks)
                self._preflight_facet_staging(
                    base,
                    manifests,
                    kv_cache=_estimate_request_footprint(
                        base,
                        inputs,
                        request,
                        composition_blocks=composition_blocks,
                        bounded=_has_bounded_budget(self.residency),
                    ),
                )
                plan, loaded = self.registry.load_resolved(
                    requested,
                    base,
                    allow_model_id_mismatch=allow_mismatch,
                    allow_unsafe_legacy_base_binding=bool(request.get("allowUnsafeLegacyBaseBinding", False)),
                    before_commit=admit,
                    resolved=resolved,
                )
                self._validate_plan_for_base(plan)
                load_elapsed = time.perf_counter() - started
                generation = normalize_generation_options(request.get("generation"))
                seed = request.get("seed")
                torch = require_torch()
                if seed is not None:
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))

                def provide(planned: PlannedBlock) -> Any:
                    return loaded[planned.facet].block(planned.slot_id, planned.block_index)

                model = base.model
                previous_training = model.training
                model.eval()
                inputs = _move_inputs(inputs, base.device)
                composed_at = time.perf_counter()
                with ActivatedComposition(base, plan, provide):
                    compose_elapsed = time.perf_counter() - composed_at
                    generation_started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, **generation)
                    generation_elapsed = time.perf_counter() - generation_started
            finally:
                try:
                    if previous_training is not None:
                        base.model.train(previous_training)
                finally:
                    if admission is not None:
                        self.residency.release(admission)

            output_ids = generated.sequences if hasattr(generated, "sequences") else generated

            input_length = int(inputs["input_ids"].shape[-1])
            generated_ids = output_ids[:, input_length:]
            texts = base.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            full_texts = None
            if bool(request.get("returnFullText", False)):
                full_texts = base.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            total_elapsed = time.perf_counter() - started
            self._generation_count += 1
            response: dict[str, Any] = {
                "schemaVersion": GENERATION_SCHEMA_VERSION,
                "requestId": str(request.get("requestId") or uuid.uuid4()),
                "base": base.binding.to_dict(),
                "requestedFacets": list(plan.requested),
                "resolvedFacets": list(plan.resolved),
                "composition": plan.to_dict(),
                "provenance": self._identity_provenance(
                    unsafe_legacy_base_binding=allow_mismatch
                ),
                "output": {"text": texts[0]},
                "usage": {
                    "inputTokens": input_length,
                    "outputTokens": int(generated_ids.shape[-1]),
                    "totalTokens": int(output_ids.shape[-1]),
                    "sequences": int(output_ids.shape[0]),
                },
                "timingMs": {
                    "facetLoad": round(load_elapsed * 1000, 3),
                    "composition": round(compose_elapsed * 1000, 3),
                    "generation": round(generation_elapsed * 1000, 3),
                    "total": round(total_elapsed * 1000, 3),
                },
            }
            if len(texts) > 1:
                response["output"]["texts"] = texts
            if full_texts is not None:
                response["output"]["fullText"] = full_texts[0]
                if len(full_texts) > 1:
                    response["output"]["fullTexts"] = full_texts
            return response

    def health(self) -> dict[str, Any]:
        with self._lock:
            torch = require_torch()
            memory: dict[str, Any] = {}
            if torch.cuda.is_available():
                memory = {
                    "cudaAllocatedBytes": int(torch.cuda.memory_allocated()),
                    "cudaReservedBytes": int(torch.cuda.memory_reserved()),
                }
            return {
                "service": "Jester Mixed Inference",
                "component": 2,
                "componentCount": 5,
                "schemaVersion": "mi-health-v1",
                "status": "ready" if self.base is not None else "waiting-for-base",
                "uptimeSeconds": round(time.time() - self._started_at, 3),
                "base": self.base.summary() if self.base is not None else None,
                "registry": str(self.registry.root),
                "availableFacets": self.registry.list(),
                "loadedFacets": self.registry.loaded(),
                "generationCount": self._generation_count,
                "memory": memory,
                "residency": self.residency.snapshot(),
            }

    def _prepare_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        base = self._require_base()
        tokenizer = base.tokenizer
        messages = request.get("messages")
        prompt = request.get("prompt")
        if messages is not None:
            if not isinstance(messages, list) or not messages:
                raise ConfigurationError("messages must be a non-empty JSON array")
            if any(not isinstance(item, dict) for item in messages):
                raise ConfigurationError("Every messages entry must be a JSON object")
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=not bool(request.get("continueFinalMessage", False)),
                    continue_final_message=bool(request.get("continueFinalMessage", False)),
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            if hasattr(encoded, "items"):
                inputs = dict(encoded)
            else:
                inputs = {"input_ids": encoded}
        elif isinstance(prompt, str) and prompt:
            inputs = dict(tokenizer(prompt, return_tensors="pt"))
        else:
            raise ConfigurationError("Generate request requires prompt or messages")
        # Some generic/fast tokenizers emit ``token_type_ids`` even when the
        # loaded decoder-only model (for example Llama) does not accept them.
        # Newer Transformers versions reject such unused kwargs before the
        # forward pass.  Keep token type IDs for architectures that explicitly
        # support them and remove only this tokenizer-derived optional input
        # when the model's forward contract does not.
        if "token_type_ids" in inputs:
            forward_parameters = inspect.signature(base.model.forward).parameters
            if "token_type_ids" not in forward_parameters:
                inputs.pop("token_type_ids")
        return inputs

    def _require_base(self) -> LoadedBaseModel:
        if self.base is None:
            raise MIError(
                "No base model is loaded. Use CLI --base-model or POST /base/load first.",
                code="base_not_loaded",
                status=409,
            )
        return self.base

    def _validate_plan_for_base(self, plan: CompositionPlan) -> None:
        base = self._require_base()
        for block in plan.blocks:
            if block.after_layer >= base.binding.base_layer_count:
                raise CompositionError(
                    f"Slot {block.slot_id} afterLayer={block.after_layer} exceeds base layer count "
                    f"{base.binding.base_layer_count}"
                )

    def _preflight_facet_staging(
        self,
        base: LoadedBaseModel,
        manifests: dict[str, Any],
        *,
        kv_cache: Footprint = Footprint(),
    ) -> None:
        staged = [
            _facet_staging_footprint(
                base,
                self.registry,
                manifest,
                bounded=_has_bounded_budget(self.residency),
            )
            for manifest in manifests.values()
            if not self.registry.is_cached(base, manifest)
        ]
        self.residency.preflight_staging(staged, kv_cache=kv_cache)

    def _identity_provenance(
        self, *, unsafe_legacy_base_binding: bool = False
    ) -> dict[str, Any]:
        base = self._require_base().binding
        eligible = (
            base.artifact_identity_status == "verified"
            and base.tokenizer_identity_status == "verified"
            and not unsafe_legacy_base_binding
        )
        return {
            "artifactIdentityStatus": base.artifact_identity_status,
            "tokenizerIdentityStatus": base.tokenizer_identity_status,
            "unsafeLegacyBaseBinding": unsafe_legacy_base_binding,
            "comparisonEligible": eligible,
        }


def extract_facet_references(payload: dict[str, Any]) -> list[str]:
    direct = payload.get("facets")
    if direct is not None:
        if not isinstance(direct, list):
            raise ConfigurationError("facets must be a JSON array")
        result = []
        for item in direct:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                value = item.get("reference") or item.get("id") or item.get("facet")
                if value:
                    result.append(str(value))
            else:
                raise ConfigurationError("facets entries must be strings or objects")
        return result
    for key in ("resolvedActiveFacets", "directSelectedFacets", "selectedFacets"):
        selected = payload.get(key)
        if selected is None:
            continue
        if not isinstance(selected, list):
            raise ConfigurationError(f"{key} must be a JSON array")
        result = []
        for item in selected:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                value = item.get("reference") or item.get("id") or item.get("facet") or item.get("name")
                if value:
                    result.append(str(value))
        return result
    route = payload.get("route")
    if isinstance(route, dict):
        return extract_facet_references(route)
    return []


def _has_bounded_budget(residency: ResidencyManager) -> bool:
    return (
        residency.budget.ram_bytes is not None
        or residency.budget.accelerator_bytes is not None
    )


def _requested_element_bytes(
    request: BaseModelRequest, torch: Any, *, bounded: bool
) -> int:
    dtype = resolve_dtype(request.dtype, torch)
    if dtype is None:
        if bounded:
            raise CompositionError(
                "base_residency_denied: bounded preflight requires an explicit baseModel.dtype"
            )
        dtype = torch.float32
    return int(torch.empty((), dtype=dtype).element_size())


def _base_staging_footprint(
    request: BaseModelRequest,
    *,
    target_device: Any,
    target_element_bytes: int,
    bounded: bool,
) -> Footprint:
    """Account source tensors and target dtype before constructing a replacement."""
    source = Path(request.model_id).expanduser()
    if not source.is_dir() or not request.artifact_files:
        if bounded:
            raise CompositionError(
                "base_residency_denied: bounded base replacement requires verified local tensor metadata"
            )
        return Footprint()
    source_bytes = 0
    target_bytes = 0
    for item in request.artifact_files:
        relative = str(item.get("path", "")) if isinstance(item, dict) else ""
        path = (source / relative).resolve()
        try:
            path.relative_to(source.resolve())
        except ValueError as exc:
            raise CompositionError("base_residency_denied: artifact path escapes local model") from exc
        if path.suffix.lower() != ".safetensors":
            continue
        source_count, target_count = safetensors_tensor_bytes(
            path, target_element_bytes=target_element_bytes
        )
        source_bytes += source_count
        target_bytes += target_count
    if not source_bytes:
        if bounded:
            raise CompositionError(
                "base_residency_denied: bounded base replacement requires safetensors tensor metadata"
            )
        return Footprint()
    if str(target_device).startswith("cuda"):
        return Footprint(ram_bytes=source_bytes, accelerator_bytes=target_bytes)
    return Footprint(ram_bytes=source_bytes + target_bytes)


def _facet_staging_footprint(
    base: LoadedBaseModel,
    registry: FacetRegistry,
    manifest: Any,
    *,
    bounded: bool,
) -> Footprint:
    """Account safetensors source and instantiated target tensors before load."""
    artifact = manifest.artifact
    if artifact is None or artifact.format != "safetensors":
        if bounded:
            raise CompositionError(
                "composition_residency_denied: bounded facet loading requires safetensors tensor metadata"
            )
        return Footprint()
    artifact_path = registry.manifest_path(manifest).parent / artifact.weights
    try:
        source_bytes, header_target_bytes = safetensors_tensor_bytes(
            artifact_path, target_element_bytes=_model_element_bytes(base.model)
        )
    except ConfigurationError as exc:
        if bounded:
            raise CompositionError(
                "composition_residency_denied: bounded facet tensor metadata is unavailable"
            ) from exc
        return Footprint()
    clone_target_bytes = _facet_clone_target_bytes(base, manifest)
    target_bytes = max(header_target_bytes, clone_target_bytes)
    if str(base.device).startswith("cuda"):
        return Footprint(ram_bytes=source_bytes, accelerator_bytes=target_bytes)
    return Footprint(ram_bytes=source_bytes + target_bytes)


def _model_element_bytes(model: Any) -> int:
    for tensor in (*model.parameters(), *model.buffers()):
        return int(tensor.element_size())
    raise CompositionError("composition_residency_denied: base has no tensors to size")


def _facet_clone_target_bytes(base: LoadedBaseModel, manifest: Any) -> int:
    """Count every cloned source layer tensor, including persistent buffers."""
    total = 0
    layers = base.stack.layers
    for slot in manifest.slots:
        for block in slot.blocks:
            if block.source_layer >= len(layers):
                raise CompositionError(
                    "composition_residency_denied: facet source layer is outside loaded base"
                )
            total += _tensor_bytes(layers[block.source_layer])
    if total <= 0:
        raise CompositionError(
            "composition_residency_denied: facet clone has no verifiable tensors"
        )
    return total


def _tensor_bytes(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in (*module.parameters(), *module.buffers()):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            total += int(tensor.numel()) * int(tensor.element_size())
    return total


def _module_footprint(module: Any, target_device: Any) -> Footprint:
    """Count tensors at their residency target, without querying hardware."""
    byte_count = 0
    seen: set[int] = set()
    for tensor in (*module.parameters(), *module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        byte_count += int(tensor.numel()) * int(tensor.element_size())
    if str(target_device).startswith("cuda"):
        return Footprint(accelerator_bytes=byte_count)
    return Footprint(ram_bytes=byte_count)


def _facet_footprint(facet: Any, target_device: Any) -> Footprint:
    byte_count = 0
    seen: set[int] = set()
    buffers = getattr(facet, "buffers", lambda: ())()
    for tensor in (*facet.parameters(), *buffers):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        byte_count += int(tensor.numel()) * int(tensor.element_size())
    if str(target_device).startswith("cuda"):
        return Footprint(accelerator_bytes=byte_count)
    return Footprint(ram_bytes=byte_count)


def _estimate_request_footprint(
    base: LoadedBaseModel,
    inputs: dict[str, Any],
    request: dict[str, Any],
    *,
    composition_blocks: int,
    bounded: bool,
) -> Footprint:
    """Request input plus conservative non-reused KV reservation."""
    generation = normalize_generation_options(request.get("generation"))
    input_ids = inputs.get("input_ids")
    batch = int(getattr(input_ids, "shape", [1])[0]) if input_ids is not None else 1
    prompt_tokens = int(getattr(input_ids, "shape", [1, 0])[-1]) if input_ids is not None else 0
    config = base.model.config
    layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    if not hidden or not layers:
        if bounded:
            raise CompositionError(
                "composition_residency_denied: bounded KV admission requires hidden_size and num_hidden_layers"
            )
        return _input_footprint(inputs, base.device)
    if bounded and layers != base.binding.base_layer_count:
        raise CompositionError(
            "composition_residency_denied: base layer metadata differs from model KV metadata"
        )
    torch = require_torch()
    parameter = next(base.model.parameters(), None)
    element_size = int(parameter.element_size()) if parameter is not None else int(torch.tensor([], dtype=torch.float32).element_size())
    expanded_batch = max(int(generation["num_beams"]), int(generation["num_return_sequences"]))
    # K and V, one hidden-state-sized allocation per resolved decoder layer/token.
    # Using hidden_size is an upper bound for grouped-KV models.
    byte_count = 2 * batch * expanded_batch * (prompt_tokens + int(generation["max_new_tokens"])) * (layers + composition_blocks) * hidden * element_size
    if str(base.device).startswith("cuda"):
        return _input_footprint(inputs, base.device) + Footprint(accelerator_bytes=byte_count)
    return _input_footprint(inputs, base.device) + Footprint(ram_bytes=byte_count)


def _input_footprint(inputs: dict[str, Any], target_device: Any) -> Footprint:
    source_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in inputs.values()
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size")
    )
    if str(target_device).startswith("cuda"):
        return Footprint(ram_bytes=source_bytes, accelerator_bytes=source_bytes)
    return Footprint(ram_bytes=source_bytes)


def _move_inputs(inputs: dict[str, Any], target_device: Any) -> dict[str, Any]:
    return {name: tensor.to(target_device) for name, tensor in inputs.items()}


def normalize_generation_options(raw: Any) -> dict[str, Any]:
    if raw is None:
        options: dict[str, Any] = {}
    elif isinstance(raw, dict):
        options = dict(raw)
    else:
        raise ConfigurationError("generation must be a JSON object")
    options.setdefault("max_new_tokens", 128)
    aliases = {
        "maxNewTokens": "max_new_tokens",
        "cacheImplementation": "cache_implementation",
        "doSample": "do_sample",
        "topP": "top_p",
        "topK": "top_k",
        "numReturnSequences": "num_return_sequences",
        "repetitionPenalty": "repetition_penalty",
    }
    for source, target in aliases.items():
        if source in options:
            options[target] = options.pop(source)
    cache = options.get("cache_implementation")
    try:
        options["max_new_tokens"] = int(options["max_new_tokens"])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("generation.max_new_tokens must be an integer") from exc
    if options["max_new_tokens"] <= 0:
        raise ConfigurationError("generation.max_new_tokens must be positive")
    for name in ("num_beams", "num_return_sequences"):
        try:
            options[name] = int(options.get(name, 1))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"generation.{name} must be an integer") from exc
        if options[name] <= 0:
            raise ConfigurationError(f"generation.{name} must be positive")
    if options["num_beams"] == 1 and options["num_return_sequences"] > 1 and not bool(options.get("do_sample", False)):
        raise ConfigurationError(
            "generation.num_return_sequences > 1 requires do_sample=true when num_beams=1"
        )
    if options["num_beams"] > 1 and options["num_return_sequences"] > options["num_beams"]:
        raise ConfigurationError(
            "generation.num_return_sequences cannot exceed generation.num_beams"
        )
    if cache in {"static", "offloaded_static", "quantized"}:
        raise ConfigurationError(
            "MI v1 supports the default dynamic cache or offloaded dynamic cache only; "
            "static/quantized cache plans are not yet compatible with dynamic MI composition"
        )
    return options
