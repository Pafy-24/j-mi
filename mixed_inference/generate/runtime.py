from __future__ import annotations

import gc
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..core.errors import CompositionError, ConfigurationError, MIError
from ..core.graph import CompositionPlan, PlannedBlock
from ..core.modeling import ActivatedComposition, LoadedBaseModel, load_base_model, require_torch
from ..core.registry import FacetRegistry
from ..core.specs import BaseModelRequest, GENERATION_SCHEMA_VERSION


class MixedInferenceRuntime:
    """Single-base runtime whose process placement is owned by RM.

    The local lock protects reversible decoder-list mutations.  It is not a
    hardware scheduler and never launches training in a managed serving process.
    """

    def __init__(self, registry: str | Path):
        self.registry = FacetRegistry(registry)
        self.base: LoadedBaseModel | None = None
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._generation_count = 0

    def load_base(self, request: BaseModelRequest | dict[str, Any]) -> dict[str, Any]:
        spec = request if isinstance(request, BaseModelRequest) else BaseModelRequest.from_dict(request)
        with self._lock:
            if self.base is not None:
                self.unload_base()
            self.base = load_base_model(spec, load_tokenizer=True)
            generation_config = getattr(self.base.model, "generation_config", None)
            if generation_config is not None and hasattr(generation_config, "max_length"):
                # MI always supplies max_new_tokens (128 by default). Recent
                # Transformers versions otherwise treat a model-provided
                # max_length as a second explicit limit and log a warning for
                # every request, even though max_new_tokens takes precedence.
                generation_config.max_length = None
            self.registry.refresh()
            return self.base.summary()

    def unload_base(self) -> dict[str, Any]:
        with self._lock:
            previous = self.base.binding.model_id if self.base is not None else None
            self.registry.unload(all_facets=True)
            self.base = None
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
                result["composedLayerCount"] = (
                    self.base.binding.base_layer_count + len(plan.blocks)
                )
            return result

    def list_facets(self) -> list[dict[str, Any]]:
        with self._lock:
            return self.registry.list()

    def refresh_registry(self) -> dict[str, Any]:
        with self._lock:
            self.registry.refresh()
            return {"facets": self.registry.list()}

    def load_facets(
        self,
        requested_facets: Iterable[str],
        *,
        allow_model_id_mismatch: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            base = self._require_base()
            plan, _ = self.registry.load_resolved(
                requested_facets,
                base,
                allow_model_id_mismatch=allow_model_id_mismatch,
            )
            self._validate_plan_for_base(plan)
            return {
                "loadedFacets": self.registry.loaded(),
                "composition": plan.to_dict(),
            }

    def unload_facets(
        self,
        facets: Iterable[str] | None = None,
        *,
        all_facets: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            removed = self.registry.unload(facets, all_facets=all_facets)
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
            requested = extract_facet_references(request)
            allow_mismatch = bool(request.get("allowModelIdMismatch", False))
            plan, loaded = self.registry.load_resolved(
                requested,
                base,
                allow_model_id_mismatch=allow_mismatch,
            )
            self._validate_plan_for_base(plan)
            load_elapsed = time.perf_counter() - started
            inputs = self._prepare_inputs(request)
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
            try:
                composed_at = time.perf_counter()
                with ActivatedComposition(base, plan, provide):
                    compose_elapsed = time.perf_counter() - composed_at
                    generation_started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, **generation)
                    generation_elapsed = time.perf_counter() - generation_started
            finally:
                model.train(previous_training)

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
        return {name: tensor.to(base.device) for name, tensor in inputs.items()}

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
    if cache in {"static", "offloaded_static", "quantized"}:
        raise ConfigurationError(
            "MI v1 supports the default dynamic cache or offloaded dynamic cache only; "
            "static/quantized cache plans are not yet compatible with dynamic MI composition"
        )
    return options
