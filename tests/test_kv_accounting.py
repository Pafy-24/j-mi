from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import save_file
import pytest

from mixed_inference.core.errors import CompositionError, ConfigurationError
from mixed_inference.core.graph import CompositionPlan, PlannedBlock
from mixed_inference.core.modeling import LayerStackHandle, LoadedBaseModel
from mixed_inference.core.residency import Footprint, ResidencyBudget, ResidencyManager
from mixed_inference.core.specs import ArtifactSpec, BaseBinding, BaseModelRequest, BlockSpec, FacetManifest, SlotSpec
from mixed_inference.core.registry import FacetRegistry
from mixed_inference.generate.runtime import (
    MixedInferenceRuntime,
    _estimate_request_footprint,
    _facet_clone_target_bytes,
    _facet_staging_footprint,
    normalize_generation_options,
)


def _base() -> LoadedBaseModel:
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])
    model.config = SimpleNamespace(num_hidden_layers=2, hidden_size=8, num_key_value_heads=1, head_dim=4)
    return LoadedBaseModel(
        request=BaseModelRequest("base"), model=model, tokenizer=None,
        stack=LayerStackHandle("layers", model, "layers", model.layers),
        binding=BaseBinding("base", None, "config", "layers", 2), device=torch.device("cpu"),
    )


def test_kv_reservation_accounts_batch_beams_returns_tokens_and_composed_layers() -> None:
    base = _base()
    inputs = {"input_ids": torch.ones((2, 3), dtype=torch.long)}
    footprint = _estimate_request_footprint(
        base,
        inputs,
        {"generation": {"max_new_tokens": 5, "num_beams": 4, "num_return_sequences": 2}},
        composition_blocks=3,
        bounded=True,
    )
    # Explicit grouped-KV geometry is 1 x 4, not hidden_size=8.
    kv = 2 * 2 * 4 * (3 + 5) * (2 + 3) * 1 * 4 * 4
    input_bytes = inputs["input_ids"].numel() * inputs["input_ids"].element_size()
    # Beam expansion retains the tokenizer output and reserves its expanded copy.
    assert footprint.ram_bytes == kv + input_bytes + input_bytes * 4
    manager = ResidencyManager(ResidencyBudget(ram_bytes=footprint.ram_bytes - 1))
    manager.reset(base=Footprint())
    with pytest.raises(CompositionError, match="base_residency_denied"):
        manager.preflight_staging([], kv_cache=footprint)


def test_kv_requires_complete_model_metadata_and_valid_expansion_profile() -> None:
    base = _base()
    base.model.config = SimpleNamespace(num_hidden_layers=0, hidden_size=0)
    with pytest.raises(CompositionError, match="requires num_hidden_layers"):
        _estimate_request_footprint(
            base, {"input_ids": torch.ones((1, 1), dtype=torch.long)}, {"generation": {}},
            composition_blocks=0, bounded=True,
        )
    with pytest.raises(ConfigurationError, match="requires do_sample"):
        normalize_generation_options({"num_return_sequences": 2})
    with pytest.raises(ConfigurationError, match="cannot exceed"):
        normalize_generation_options({"num_beams": 2, "num_return_sequences": 3})


def test_runtime_rejects_expanded_kv_profile_before_facet_load(tmp_path, monkeypatch) -> None:
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=100))
    )
    runtime.base = _base()
    runtime.residency.reset(base=Footprint(ram_bytes=1))
    blocks = tuple(
        PlannedBlock("facet", "1", f"slot-{index}", 0, index, 0, 0)
        for index in range(3)
    )
    plan = CompositionPlan(requested=("facet",), resolved=("facet@1",), blocks=blocks)
    monkeypatch.setattr(runtime.registry, "resolve", lambda _requested: (plan, {}))
    monkeypatch.setattr(
        runtime.registry,
        "load_resolved",
        lambda *_args, **_kwargs: pytest.fail("KV denial must happen before facet loading"),
    )
    monkeypatch.setattr(
        runtime,
        "_prepare_inputs",
        lambda _request: {"input_ids": torch.ones((2, 3), dtype=torch.long)},
    )
    with pytest.raises(CompositionError, match="base_residency_denied"):
        runtime.generate(
            {"prompt": "x", "facets": ["facet"], "generation": {"max_new_tokens": 5, "num_beams": 4}}
        )


def test_facet_clone_preflight_counts_nonpersistent_buffers(tmp_path) -> None:
    base = _base()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    layer.register_buffer("persistent", torch.ones(1, dtype=torch.float32))
    layer.register_buffer("nonpersistent", torch.ones(3, dtype=torch.float32), persistent=False)
    base.model.layers = torch.nn.ModuleList([layer, torch.nn.Linear(8, 8)])
    base.stack.layers = base.model.layers
    manifest = FacetManifest(
        facet_id="facet", version="1", base=base.binding,
        slots=(SlotSpec("slot", 0, 0, (BlockSpec(0),)),),
        artifact=ArtifactSpec(weights="facet.safetensors", format="safetensors"),
    )
    save_file({"slots.slot.blocks.0000.weight": torch.ones(1, dtype=torch.float16)}, str(tmp_path / "facet.safetensors"))
    registry = FacetRegistry(tmp_path)
    registry._manifests = {("facet", "1"): (manifest, tmp_path / "facet.mi.json")}
    # Parameter (16) + persistent buffer (4) + nonpersistent buffer (12).
    assert _facet_clone_target_bytes(base, manifest) == 32
    footprint = _facet_staging_footprint(base, registry, manifest, bounded=True)
    # The fp16 serialized source is 2 bytes; the clone shape, including its
    # nonpersistent buffer, controls target residency.
    assert footprint.ram_bytes == 2 + 32


def test_kv_geometry_uses_explicit_kv_heads_not_hidden_size() -> None:
    base = _base()
    base.model.config = SimpleNamespace(
        num_hidden_layers=2, hidden_size=64, num_key_value_heads=2, head_dim=3
    )
    footprint = _estimate_request_footprint(
        base, {"input_ids": torch.ones((1, 2), dtype=torch.long)},
        {"generation": {"max_new_tokens": 1}}, composition_blocks=0, bounded=True,
    )
    # 2 K/V * batch * tokens * layers * kv_heads * head_dim * float32.
    assert footprint.ram_bytes == 2 * 1 * 3 * 2 * 2 * 3 * 4 + 16


def test_bounded_kv_rejects_offloaded_or_unknown_cache_and_external_state() -> None:
    base = _base()
    inputs = {"input_ids": torch.ones((1, 1), dtype=torch.long)}
    with pytest.raises(CompositionError, match="offloaded KV"):
        _estimate_request_footprint(
            base, inputs, {"generation": {"cache_implementation": "offloaded"}},
            composition_blocks=0, bounded=True,
        )
    with pytest.raises(ConfigurationError, match="dynamic or offloaded"):
        normalize_generation_options({"cache_implementation": "surprise"})
    with pytest.raises(ConfigurationError, match="caller-supplied cache state"):
        normalize_generation_options({"past_key_values": object()})


def test_generate_rejects_top_level_cache_state_before_loading(tmp_path) -> None:
    runtime = MixedInferenceRuntime(tmp_path / "facets")
    with pytest.raises(ConfigurationError, match="does not accept caller-supplied cache state"):
        runtime.generate({"prompt": "x", "past_key_values": object()})
