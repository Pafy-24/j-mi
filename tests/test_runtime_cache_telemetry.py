from __future__ import annotations

from pathlib import Path

import pytest


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
        save_facet(
            materialize_facet(manifest, runtime.base.stack.layers),
            facets_root / facet_id / "1",
        )
    runtime.refresh_registry()
    return runtime, base_request, facets_root


def _request(facet: str) -> dict[str, object]:
    return {
        "prompt": "hello",
        "facets": [f"{facet}@1"],
        "generation": {"max_new_tokens": 2, "do_sample": False},
    }


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
    )
    fresh.load_base(base_request)
    fresh_a = fresh.generate(_request("facet-a"))
    assert second_a["output"] == fresh_a["output"]
    assert fresh.residency.snapshot()["pinnedEntries"] == 0
