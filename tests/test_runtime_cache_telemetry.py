from __future__ import annotations

from pathlib import Path

import pytest


CPU_LOGIT_RTOL = 1e-5
CPU_LOGIT_ATOL = 1e-6


def _tiny_runtime(tmp_path: Path):
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from mixed_inference.core.checkpoint import materialize_facet, save_facet
    from mixed_inference.core.identity import canonical_artifact_manifest
    from mixed_inference.core.residency import ResidencyBudget, ResidencyManager
    from mixed_inference.core.util import sha256_file
    from mixed_inference.core.specs import BlockSpec, FacetManifest, SlotSpec
    from mixed_inference.generate.runtime import MixedInferenceRuntime

    Tokenizer = tokenizers.Tokenizer
    WordLevel = tokenizers.models.WordLevel
    Whitespace = tokenizers.pre_tokenizers.Whitespace
    PreTrainedTokenizerFast = transformers.PreTrainedTokenizerFast
    LlamaConfig = transformers.LlamaConfig
    LlamaForCausalLM = transformers.LlamaForCausalLM

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "hello": 4, "world": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32, pad_token_id=0,
        bos_token_id=2, eos_token_id=3,
    ))
    model.save_pretrained(model_dir)

    facets_root = tmp_path / "facets"
    required_names = [
        name for name in (
            "config.json", "model.safetensors", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        ) if (model_dir / name).is_file()
    ]
    artifact_files = [
        {"path": name, "sha256": sha256_file(model_dir / name), "sizeBytes": (model_dir / name).stat().st_size}
        for name in required_names
    ]
    base_request = {
        "id": str(model_dir), "device": "cpu", "dtype": "float32",
        "artifactFiles": artifact_files, "artifactSha256": canonical_artifact_manifest(artifact_files),
    }
    runtime = MixedInferenceRuntime(
        facets_root,
        residency=ResidencyManager(ResidencyBudget(ram_bytes=4_000_000)),
        staging_directory=tmp_path / "staging",
        staging_limit_bytes=4_000_000,
    )
    runtime.load_base(base_request)
    assert runtime.base is not None
    for facet_id in ("facet-a", "facet-b"):
        manifest = FacetManifest(
            facet_id=facet_id,
            version="1",
            base=runtime.base.binding,
            slots=(SlotSpec(
                slot_id=f"{facet_id}-slot", after_layer=0, order=0,
                blocks=(BlockSpec(source_layer=0, initializer="identity-copy"),),
            ),),
        )
        facet = materialize_facet(manifest, runtime.base.stack.layers)
        if facet_id == "facet-b":
            # Deterministic non-neutral checkpoint: every floating tensor is
            # perturbed, so A and B prove distinct composition behavior.
            generator = torch.Generator().manual_seed(811)
            with torch.no_grad():
                for parameter in facet.parameters():
                    parameter.add_(
                        torch.randn(
                            parameter.shape,
                            generator=generator,
                            dtype=parameter.dtype,
                        ) * 0.2
                    )
        save_facet(facet, facets_root / facet_id / "1")
    runtime.refresh_registry()
    return runtime, base_request, facets_root


def _request(facet: str) -> dict[str, object]:
    return {
        "prompt": "hello",
        "facets": [f"{facet}@1"],
        "generation": {"max_new_tokens": 2, "do_sample": False},
    }


def _multi_request(*facets: str) -> dict[str, object]:
    return {
        "prompt": "hello",
        "facets": [f"{facet}@1" for facet in facets],
        "generation": {"max_new_tokens": 2, "do_sample": False},
    }


def _composition_logits(runtime, facet: str):
    import torch
    from mixed_inference.core.modeling import ActivatedComposition

    reference = f"{facet}@1"
    resolved = runtime.registry.resolve([reference])
    staged = runtime.registry.stage_resolved([reference], runtime.base, resolved=resolved)
    assert staged.staged == {}

    def provide(planned):
        return staged.loaded[planned.facet].block(planned.slot_id, planned.block_index)

    inputs = runtime._prepare_inputs(_request(facet))
    with torch.inference_mode(), ActivatedComposition(runtime.base, staged.plan, provide):
        return runtime.base.model(**inputs).logits.detach().clone()


def _tighten_cpu_budget(runtime) -> int:
    from mixed_inference.core.residency import ResidencyBudget, ResidencyManager
    from mixed_inference.generate.runtime import (
        _estimate_request_footprint,
        _facet_staging_footprint,
        _module_footprint,
    )

    margin = 4_096
    _, manifests = runtime.registry.resolve(["facet-a@1"])
    staging = _facet_staging_footprint(
        runtime.base,
        runtime.registry,
        next(iter(manifests.values())),
        bounded=True,
    )
    inputs = runtime._prepare_inputs(_request("facet-a"))
    kv = _estimate_request_footprint(
        runtime.base,
        inputs,
        _request("facet-a"),
        composition_blocks=1,
        bounded=True,
    )
    base = _module_footprint(runtime.base.model, runtime.base.device)
    budget_bytes = base.ram_bytes + staging.ram_bytes + kv.ram_bytes + margin
    runtime.residency = ResidencyManager(
        ResidencyBudget(ram_bytes=budget_bytes, operational_margin_bytes=margin)
    )
    runtime.residency.reset(base=base)
    return budget_bytes


def test_cpu_alternating_compositions_reuse_cache_and_match_fresh_load(tmp_path: Path) -> None:
    """Exercise the actual tokenizer/model/facet path without downloads or CUDA."""
    torch = pytest.importorskip("torch")
    from mixed_inference.core.residency import ResidencyBudget, ResidencyManager
    from mixed_inference.generate.runtime import MixedInferenceRuntime

    runtime, base_request, facets_root = _tiny_runtime(tmp_path)
    assert runtime.base is not None
    base_layer_count = len(runtime.base.stack.layers)
    base_state = {name: value.detach().clone() for name, value in runtime.base.model.state_dict().items()}

    first_a = runtime.generate(_request("facet-a"))
    first_b = runtime.generate(_request("facet-b"))
    second_a = runtime.generate(_request("facet-a"))

    assert first_a["telemetry"]["cache"]["misses"] == 1
    assert first_a["telemetry"]["cache"]["loadedRamBytes"] > 0
    assert first_b["telemetry"]["cache"]["misses"] == 1
    assert second_a["telemetry"]["cache"] == {
        "status": "measured", "hits": 1, "misses": 0, "evictions": 0,
        "loadedRamBytes": 0, "loadedAcceleratorBytes": 0,
    }
    assert second_a["telemetry"]["composition"]["status"] == "measured"
    assert second_a["telemetry"]["admission"]["status"] == "estimated"
    assert second_a["telemetry"]["hardwarePeak"]["status"] == "unavailable"
    assert runtime.residency.snapshot()["entries"] == 2
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    health_cache = runtime.health()["telemetry"]["cache"]
    assert health_cache["status"] == "measured"
    assert (health_cache["hits"], health_cache["misses"], health_cache["evictions"]) == (1, 2, 0)
    assert health_cache["loadedRamBytes"] > 0
    assert len(runtime.base.stack.layers) == base_layer_count
    assert all(torch.equal(value, runtime.base.model.state_dict()[name]) for name, value in base_state.items())

    fresh = MixedInferenceRuntime(
        facets_root,
        residency=ResidencyManager(ResidencyBudget(ram_bytes=4_000_000)),
        staging_directory=tmp_path / "fresh-staging-a",
        staging_limit_bytes=4_000_000,
    )
    fresh.load_base(base_request)
    fresh_a = fresh.generate(_request("facet-a"))
    assert second_a["output"] == fresh_a["output"]
    assert fresh.residency.snapshot()["pinnedEntries"] == 0


def test_cpu_distinct_facets_swap_under_tight_budget_and_match_fresh_logits(
    tmp_path: Path,
) -> None:
    """Qualify deterministic CPU composition without downloads or CUDA."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from mixed_inference.core.residency import ResidencyBudget, ResidencyManager
    from mixed_inference.core.util import sha256_file
    from mixed_inference.generate.runtime import MixedInferenceRuntime

    runtime, base_request, facets_root = _tiny_runtime(tmp_path)
    budget_bytes = _tighten_cpu_budget(runtime)
    base_layers = len(runtime.base.stack.layers)
    base_state = {
        name: value.detach().clone()
        for name, value in runtime.base.model.state_dict().items()
    }
    a_path = facets_root / "facet-a" / "1" / "facet.safetensors"
    b_path = facets_root / "facet-b" / "1" / "facet.safetensors"
    a_state, b_state = load_file(str(a_path)), load_file(str(b_path))
    assert sha256_file(a_path) != sha256_file(b_path)
    assert any(not torch.equal(a_state[name], b_state[name.replace("facet-a", "facet-b")])
               for name in a_state)

    first_a = runtime.generate(_request("facet-a"))
    logits_a = _composition_logits(runtime, "facet-a")
    first_b = runtime.generate(_request("facet-b"))
    logits_b = _composition_logits(runtime, "facet-b")
    second_a = runtime.generate(_request("facet-a"))
    logits_a_reloaded = _composition_logits(runtime, "facet-a")

    assert first_a["resolvedFacets"] == ["facet-a@1"]
    assert first_b["resolvedFacets"] == ["facet-b@1"]
    assert second_a["composition"] == first_a["composition"]
    assert first_a["telemetry"]["cache"]["evictions"] == 0
    assert first_b["telemetry"]["cache"]["evictions"] == 1
    assert second_a["telemetry"]["cache"]["evictions"] == 1
    assert first_b["telemetry"]["cache"]["misses"] == 1
    assert second_a["telemetry"]["cache"]["misses"] == 1
    assert runtime.residency.snapshot() == {
        "ramBytes": runtime.residency.snapshot()["ramBytes"],
        "acceleratorBytes": 0,
        "entries": 1,
        "pinnedEntries": 0,
        "ramBudgetBytes": budget_bytes,
        "acceleratorBudgetBytes": None,
    }
    assert not torch.allclose(logits_a, logits_b, rtol=CPU_LOGIT_RTOL, atol=CPU_LOGIT_ATOL)
    torch.testing.assert_close(
        logits_a_reloaded, logits_a, rtol=CPU_LOGIT_RTOL, atol=CPU_LOGIT_ATOL
    )

    fresh_outputs = {}
    fresh_logits = {}
    for facet in ("facet-a", "facet-b"):
        fresh = MixedInferenceRuntime(
            facets_root,
            residency=ResidencyManager(ResidencyBudget(ram_bytes=4_000_000)),
            staging_directory=tmp_path / f"fresh-staging-{facet}",
            staging_limit_bytes=4_000_000,
        )
        fresh.load_base(base_request)
        _tighten_cpu_budget(fresh)
        fresh_outputs[facet] = fresh.generate(_request(facet))["output"]
        fresh_logits[facet] = _composition_logits(fresh, facet)
        assert fresh.residency.snapshot()["pinnedEntries"] == 0

    assert first_a["output"] == second_a["output"] == fresh_outputs["facet-a"]
    assert first_b["output"] == fresh_outputs["facet-b"]
    torch.testing.assert_close(
        logits_a, fresh_logits["facet-a"], rtol=CPU_LOGIT_RTOL, atol=CPU_LOGIT_ATOL
    )
    torch.testing.assert_close(
        logits_b, fresh_logits["facet-b"], rtol=CPU_LOGIT_RTOL, atol=CPU_LOGIT_ATOL
    )
    assert len(runtime.base.stack.layers) == base_layers
    assert all(
        torch.equal(value, runtime.base.model.state_dict()[name])
        for name, value in base_state.items()
    )
    staging = tmp_path / "staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_wrong_digest_precedes_eviction_then_load_failure_keeps_stores_coherent(
    tmp_path: Path, monkeypatch
) -> None:
    from mixed_inference.core.errors import CompatibilityError
    from mixed_inference.core import registry as registry_module

    runtime, _, facets_root = _tiny_runtime(tmp_path)
    _tighten_cpu_budget(runtime)
    runtime.generate(_request("facet-a"))
    assert runtime.registry.loaded() == ["facet-a@1"]
    before = runtime.residency.snapshot()

    b_weights = facets_root / "facet-b" / "1" / "facet.safetensors"
    original = b_weights.read_bytes()
    b_weights.write_bytes(original + b"corrupt")
    with pytest.raises(CompatibilityError, match="weight size mismatch|weight hash mismatch"):
        runtime.generate(_request("facet-b"))
    assert runtime.registry.loaded() == ["facet-a@1"]
    assert runtime.residency.snapshot() == before
    b_weights.write_bytes(original)

    original_loader = registry_module.load_facet

    def fail_b(path, layers, **kwargs):
        if "facet-b" in str(path):
            raise CompatibilityError("injected facet-b load failure")
        return original_loader(path, layers, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(registry_module, "load_facet", fail_b)
        with pytest.raises(CompatibilityError, match="injected facet-b"):
            runtime.generate(_request("facet-b"))
    assert runtime.registry.loaded() == []
    assert runtime.residency.snapshot()["entries"] == 0
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    failed_health = runtime.health()["telemetry"]["cache"]
    assert failed_health["evictions"] == 1
    assert failed_health["loadFailures"] == 1
    assert failed_health["committedEntries"] == 1
    assert failed_health["failedLoadBytes"]["status"] == "unavailable"

    recovered = runtime.generate(_request("facet-b"))
    assert recovered["resolvedFacets"] == ["facet-b@1"]
    assert runtime.registry.loaded() == ["facet-b@1"]
    assert runtime.residency.snapshot()["entries"] == 1
    assert runtime.residency.snapshot()["pinnedEntries"] == 0


def test_registry_commit_failure_with_hit_and_miss_preserves_only_the_hit(
    tmp_path: Path, monkeypatch
) -> None:
    runtime, _, _ = _tiny_runtime(tmp_path)
    runtime.generate(_request("facet-a"))
    before = runtime.residency.snapshot()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            runtime.registry,
            "commit_staged",
            lambda _staged: (_ for _ in ()).throw(RuntimeError("commit failed")),
        )
        with pytest.raises(RuntimeError, match="commit failed"):
            runtime.generate(_multi_request("facet-a", "facet-b"))
    assert runtime.registry.loaded() == ["facet-a@1"]
    assert runtime.residency.snapshot()["entries"] == before["entries"] == 1
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    failed_health = runtime.health()["telemetry"]["cache"]
    assert failed_health["commitFailures"] == 1
    assert failed_health["committedEntries"] == 1

    # The failed miss was not published and loads cleanly on the next request.
    recovered = runtime.generate(_request("facet-b"))
    assert recovered["telemetry"]["cache"]["misses"] == 1


def test_oversized_multi_facet_request_is_rejected_without_hidden_drop_or_load(
    tmp_path: Path, monkeypatch
) -> None:
    from mixed_inference.core.errors import CompositionError
    from mixed_inference.core import registry as registry_module

    runtime, _, _ = _tiny_runtime(tmp_path)
    _tighten_cpu_budget(runtime)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            registry_module,
            "load_facet",
            lambda *_args, **_kwargs: pytest.fail("oversize must fail before allocation"),
        )
        with pytest.raises(CompositionError, match="composition_residency_denied"):
            runtime.generate(_multi_request("facet-a", "facet-b"))
    assert runtime.registry.loaded() == []
    assert runtime.residency.snapshot()["entries"] == 0
    assert runtime.residency.snapshot()["pinnedEntries"] == 0


def test_staging_quota_counts_complete_missing_facet_closure(
    tmp_path: Path, monkeypatch
) -> None:
    from mixed_inference.core.errors import ConfigurationError
    from mixed_inference.core import checkpoint as checkpoint_module

    runtime, _, facets_root = _tiny_runtime(tmp_path)
    sizes = [
        (facets_root / facet / "1" / "facet.safetensors").stat().st_size
        for facet in ("facet-a", "facet-b")
    ]
    runtime._staging_limit_bytes = max(sizes)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            checkpoint_module,
            "materialize_facet",
            lambda *_args, **_kwargs: pytest.fail(
                "the full closure must be denied before materialization"
            ),
        )
        with pytest.raises(ConfigurationError, match="facet_staging_denied"):
            runtime.generate(_multi_request("facet-a", "facet-b"))
    assert runtime.registry.loaded() == []
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    staging = tmp_path / "staging"
    assert not staging.exists() or list(staging.iterdir()) == []


def test_generation_failure_after_cache_commit_is_counted_and_releases_pins(
    tmp_path: Path, monkeypatch
) -> None:
    runtime, _, _ = _tiny_runtime(tmp_path)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            runtime.base.model,
            "generate",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("generation failed")),
        )
        with pytest.raises(RuntimeError, match="generation failed"):
            runtime.generate(_request("facet-a"))

    assert runtime.registry.loaded() == ["facet-a@1"]
    assert runtime.residency.snapshot()["entries"] == 1
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    health = runtime.health()["telemetry"]["cache"]
    assert health["misses"] == 1
    assert health["committedEntries"] == 1
    assert health["loadedRamBytes"] > 0
    assert health["loadFailures"] == 0
    assert health["commitFailures"] == 0

    recovered = runtime.generate(_request("facet-a"))
    assert recovered["telemetry"]["cache"]["hits"] == 1


def test_manifest_change_invalidates_registry_and_residency_before_fresh_load(
    tmp_path: Path,
) -> None:
    import json

    runtime, _, facets_root = _tiny_runtime(tmp_path)
    runtime.generate(_request("facet-a"))
    old_key = runtime.registry.cache_keys_for(["facet-a"])[0]
    manifest_path = facets_root / "facet-a" / "1" / "facet.mi.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["description"] = "changed identity"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    runtime.refresh_registry()
    assert runtime.registry.loaded() == []
    assert runtime.residency.snapshot()["entries"] == 0
    runtime.generate(_request("facet-a"))
    new_key = runtime.registry.cache_keys_for(["facet-a"])[0]
    assert new_key != old_key
    assert runtime.residency.snapshot()["pinnedEntries"] == 0


def test_manifest_replacement_between_preflight_and_load_refuses_before_materialize(
    tmp_path: Path, monkeypatch
) -> None:
    import json
    from mixed_inference.core import checkpoint as checkpoint_module
    from mixed_inference.core.errors import CompatibilityError

    runtime, _, facets_root = _tiny_runtime(tmp_path)
    _tighten_cpu_budget(runtime)
    runtime.generate(_request("facet-a"))
    manifest_path = facets_root / "facet-b" / "1" / "facet.mi.json"
    original_stage = runtime.registry.stage_resolved

    def replace_then_stage(*args, **kwargs):
        replacement = json.loads(manifest_path.read_text(encoding="utf-8"))
        replacement["description"] = "replaced after validated snapshot"
        manifest_path.write_text(json.dumps(replacement), encoding="utf-8")
        return original_stage(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(runtime.registry, "stage_resolved", replace_then_stage)
        scoped.setattr(
            checkpoint_module,
            "materialize_facet",
            lambda *_args, **_kwargs: pytest.fail(
                "changed manifest must be rejected before materialization"
            ),
        )
        with pytest.raises(CompatibilityError, match="changed after resolution"):
            runtime.generate(_request("facet-b"))

    # The already applied inactive eviction remains coherent, while the
    # replacement publishes no old-key facet and leaves no request pin.
    assert runtime.registry.loaded() == []
    assert runtime.residency.snapshot()["entries"] == 0
    assert runtime.residency.snapshot()["pinnedEntries"] == 0
    health = runtime.health()["telemetry"]["cache"]
    assert health["evictions"] == 1
    assert health["loadFailures"] == 1


def test_artifact_replacement_after_snapshot_cannot_change_loaded_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    from mixed_inference.core.errors import CompatibilityError

    runtime, _, facets_root = _tiny_runtime(tmp_path)
    weights_path = facets_root / "facet-b" / "1" / "facet.safetensors"
    original_stage = runtime.registry.stage_resolved

    def replace_then_stage(*args, **kwargs):
        weights_path.write_bytes(b"replacement after verified snapshot")
        return original_stage(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(runtime.registry, "stage_resolved", replace_then_stage)
        response = runtime.generate(_request("facet-b"))
    assert response["resolvedFacets"] == ["facet-b@1"]
    assert runtime.registry.loaded() == ["facet-b@1"]
    assert runtime.residency.snapshot()["pinnedEntries"] == 0

    # Every request validates current registry bytes, including cache hits.
    with pytest.raises(CompatibilityError, match="weight size mismatch|weight hash mismatch"):
        runtime.generate(_request("facet-b"))


def test_artifact_replacement_before_snapshot_refuses_without_materialization(
    tmp_path: Path, monkeypatch
) -> None:
    from mixed_inference.core import checkpoint as checkpoint_module
    from mixed_inference.core.errors import CompatibilityError

    runtime, _, facets_root = _tiny_runtime(tmp_path)
    runtime.generate(_request("facet-a"))
    before = runtime.residency.snapshot()
    weights_path = facets_root / "facet-b" / "1" / "facet.safetensors"
    original_snapshot = runtime.registry.snapshot_artifacts

    def replace_then_snapshot(*args, **kwargs):
        weights_path.write_bytes(b"replacement after identity validation")
        return original_snapshot(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(runtime.registry, "snapshot_artifacts", replace_then_snapshot)
        scoped.setattr(
            checkpoint_module,
            "materialize_facet",
            lambda *_args, **_kwargs: pytest.fail(
                "changed artifact must be rejected before materialization"
            ),
        )
        with pytest.raises(CompatibilityError, match="weight size mismatch|weight hash mismatch"):
            runtime.generate(_request("facet-b"))

    assert runtime.registry.loaded() == ["facet-a@1"]
    assert runtime.residency.snapshot() == before
    assert list((tmp_path / "staging").iterdir()) == []
