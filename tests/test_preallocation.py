from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from mixed_inference.core.errors import CompatibilityError, CompositionError
from mixed_inference.core.graph import CompositionPlan
from mixed_inference.core.modeling import LayerStackHandle, LoadedBaseModel
from mixed_inference.core.residency import Footprint, ResidencyBudget, ResidencyManager
from mixed_inference.core.specs import BaseBinding, BaseModelRequest
from mixed_inference.core.tensor_metadata import safetensors_tensor_bytes
from mixed_inference.generate.runtime import (
    MixedInferenceRuntime,
    _base_staging_footprint,
    _facet_staging_footprint,
)


def _artifact_files(root):
    weights = root / "model.safetensors"
    config = root / "config.json"
    save_file({"weight": torch.ones(4, dtype=torch.float16)}, str(weights))
    config.write_text("{}", encoding="utf-8")
    return [
        {"path": item.name, "sizeBytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in (weights, config)
    ]


def test_fp16_safetensors_preflight_uses_fp32_tensor_bytes_before_constructor(tmp_path, monkeypatch) -> None:
    files = _artifact_files(tmp_path)
    source_bytes, fp32_bytes = safetensors_tensor_bytes(
        tmp_path / "model.safetensors", target_element_bytes=4
    )
    assert source_bytes == 8
    assert fp32_bytes == 16
    from mixed_inference.core.specs import BaseModelRequest
    request = BaseModelRequest.from_dict(
        {"id": str(tmp_path), "device": "cpu", "dtype": "float32", "artifactFiles": files}
    )
    footprint = _base_staging_footprint(
        request, target_device=torch.device("cpu"), target_element_bytes=4, bounded=True
    )
    assert footprint.ram_bytes == source_bytes + fp32_bytes
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=31))
    )
    runtime.residency.reset(base=Footprint(ram_bytes=8))
    old = object()
    runtime.base = old
    monkeypatch.setattr(
        "mixed_inference.generate.runtime.load_base_model",
        lambda *_args, **_kwargs: pytest.fail("preflight must deny before constructor"),
    )
    with pytest.raises(CompositionError, match="base_residency_denied"):
        runtime.load_base(request)
    assert runtime.base is old


def test_incomplete_local_artifact_manifest_is_rejected_before_constructor(tmp_path, monkeypatch) -> None:
    files = _artifact_files(tmp_path)
    config_only = [item for item in files if item["path"] == "config.json"]
    from mixed_inference.core.identity import canonical_artifact_manifest

    request = BaseModelRequest.from_dict(
        {
            "id": str(tmp_path), "device": "cpu", "dtype": "float32",
            "artifactFiles": config_only,
            "artifactSha256": canonical_artifact_manifest(config_only),
        }
    )
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=100))
    )
    monkeypatch.setattr(
        "mixed_inference.generate.runtime.load_base_model",
        lambda *_args, **_kwargs: pytest.fail("incomplete identity must deny before constructor"),
    )
    with pytest.raises(CompatibilityError, match="omits loader input"):
        runtime.load_base(request)


def test_bounded_facet_without_artifact_metadata_is_rejected_before_clone(tmp_path) -> None:
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=100))
    )
    manifest = SimpleNamespace(artifact=None)
    with pytest.raises(CompositionError, match="requires safetensors tensor metadata"):
        _facet_staging_footprint(
            None,
            runtime.registry,
            manifest,
            bounded=True,
        )


def test_pin_release_survives_model_train_restore_failure(tmp_path, monkeypatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=1, num_key_value_heads=1, head_dim=1)
            self._saw_eval = False

        def train(self, mode: bool = True):
            if mode and self._saw_eval:
                raise RuntimeError("restore failed")
            if not mode:
                self._saw_eval = True
            return super().train(mode)

        def generate(self, **_kwargs):
            raise RuntimeError("generation failed")

    class Facet:
        def __init__(self) -> None:
            self.manifest = SimpleNamespace(facet_id="facet", version="1")
            self.weight = torch.nn.Parameter(torch.ones(1))

        def parameters(self):
            return iter((self.weight,))

    model = Model()
    runtime = MixedInferenceRuntime(
        tmp_path / "facets", residency=ResidencyManager(ResidencyBudget(ram_bytes=10_000))
    )
    runtime.base = LoadedBaseModel(
        request=BaseModelRequest("base"), model=model, tokenizer=object(),
        stack=LayerStackHandle("layers", model, "layers", model.layers),
        binding=BaseBinding("base", None, "config", "layers", 1), device=torch.device("cpu"),
    )
    runtime.residency.reset(base=Footprint(ram_bytes=8))
    plan = CompositionPlan(requested=("facet",), resolved=("facet@1",), blocks=())
    facet = Facet()
    monkeypatch.setattr(runtime.registry, "resolve", lambda _requested: (plan, {"facet": object()}))
    monkeypatch.setattr(runtime, "_preflight_facet_staging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime.registry, "cache_key", lambda *_args: "facet-cache")
    monkeypatch.setattr(runtime.registry, "evict_cache_keys", lambda _keys: None)
    monkeypatch.setattr(runtime.registry, "is_cached", lambda *_args: False)
    monkeypatch.setattr(
        runtime.registry,
        "load_resolved",
        lambda *_args, before_commit, **_kwargs: (
            before_commit({"facet": facet}) or (plan, {"facet": facet})
        ),
    )
    monkeypatch.setattr(
        runtime, "_prepare_inputs", lambda _request: {"input_ids": torch.ones((1, 1), dtype=torch.long)}
    )
    with pytest.raises(RuntimeError, match="restore failed"):
        runtime.generate({"prompt": "x", "facets": ["facet"], "generation": {}})
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
