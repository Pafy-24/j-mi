from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mixed_inference.core.errors import CompatibilityError, ConfigurationError
from mixed_inference.core.identity import canonical_artifact_manifest, tokenizer_identity, verify_local_artifact
from mixed_inference.core.modeling import LayerStackHandle, LoadedBaseModel, load_base_model, validate_base_binding
from mixed_inference.core.registry import _cache_key
from mixed_inference.core.specs import BaseBinding, BaseModelRequest, BlockSpec, FacetManifest, SlotSpec
from mixed_inference.generate.runtime import MixedInferenceRuntime


def _files(root):
    from safetensors.torch import save_file

    weights = root / "model.safetensors"
    config = root / "config.json"
    save_file({"weight": torch.ones(2, dtype=torch.float16)}, str(weights))
    config.write_text('{"layers": 1}', encoding="utf-8")
    return [
        {"path": item.name, "sizeBytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in (weights, config)
    ]


def test_local_artifact_manifest_is_confined_and_verified(tmp_path) -> None:
    files = _files(tmp_path)
    digest = canonical_artifact_manifest(reversed(files))
    identity = verify_local_artifact(tmp_path, files, digest)
    assert identity.status == "verified"
    assert identity.sha256 == digest
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(CompatibilityError, match="size differs|hash differs"):
        verify_local_artifact(tmp_path, files, digest)
    with pytest.raises(ConfigurationError, match="confined relative"):
        canonical_artifact_manifest([{"path": "../escape", "sizeBytes": 1, "sha256": "0" * 64}])
    config_only = [item for item in files if item["path"] == "config.json"]
    with pytest.raises(CompatibilityError, match="omits loader input"):
        verify_local_artifact(tmp_path, config_only, canonical_artifact_manifest(config_only))
    note = tmp_path / "README.txt"
    note.write_text("not a loader input", encoding="utf-8")
    extra = files + [{"path": note.name, "sizeBytes": note.stat().st_size, "sha256": hashlib.sha256(note.read_bytes()).hexdigest()}]
    with pytest.raises(ConfigurationError, match="non-loader input"):
        verify_local_artifact(tmp_path, extra, canonical_artifact_manifest(extra))


def test_sharded_safetensors_contract_requires_every_selected_shard(tmp_path) -> None:
    from safetensors.torch import save_file

    config = tmp_path / "config.json"
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text("{}", encoding="utf-8")
    save_file({"one": torch.ones(1)}, str(first))
    save_file({"two": torch.ones(1)}, str(second))
    index.write_text(
        json.dumps({"weight_map": {"one": first.name, "two": second.name}}),
        encoding="utf-8",
    )
    files = [
        {"path": item.name, "sizeBytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in (config, index, first, second)
    ]
    digest = canonical_artifact_manifest(files)
    assert verify_local_artifact(tmp_path, files, digest).status == "verified"
    omitted = [item for item in files if item["path"] != second.name]
    with pytest.raises(CompatibilityError, match="omits loader input"):
        verify_local_artifact(tmp_path, omitted, canonical_artifact_manifest(omitted))


def test_bin_and_loader_selection_overrides_cannot_be_verified(tmp_path) -> None:
    config = tmp_path / "config.json"
    binary = tmp_path / "pytorch_model.bin"
    config.write_text("{}", encoding="utf-8")
    binary.write_bytes(b"not verified weights")
    files = [
        {"path": item.name, "sizeBytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in (config, binary)
    ]
    with pytest.raises(ConfigurationError, match="supports only model.safetensors"):
        verify_local_artifact(tmp_path, files, canonical_artifact_manifest(files))
    valid = tmp_path / "model.safetensors"
    from safetensors.torch import save_file
    save_file({"weight": torch.ones(1)}, str(valid))
    supported_files = [
        {"path": item.name, "sizeBytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in (config, valid)
    ]
    with pytest.raises(ConfigurationError, match="loader-selection overrides"):
        verify_local_artifact(
            tmp_path,
            supported_files,
            canonical_artifact_manifest(supported_files),
            model_kwargs={"use_safetensors": False},
        )


class _Backend:
    def __init__(self, pipeline: str) -> None:
        self.pipeline = pipeline

    def to_str(self) -> str:
        return self.pipeline


class _Tokenizer:
    special_tokens_map = {"eos_token": "</s>"}
    init_kwargs = {"normalizer": "nfc", "name_or_path": "host-specific"}

    def __init__(self, pipeline: str) -> None:
        self.backend_tokenizer = _Backend(pipeline)

    def get_added_vocab(self):
        return {"<extra>": 9}


def test_tokenizer_identity_covers_pipeline_and_marks_unsupported_unverified() -> None:
    first = tokenizer_identity(_Tokenizer('{"normalizer":"nfc"}'))
    second = tokenizer_identity(_Tokenizer('{"normalizer":"lowercase"}'))
    assert first.status == "verified"
    assert first.sha256 != second.sha256
    assert tokenizer_identity(object()).status == "unverified"
    unsupported = _Tokenizer('{"normalizer":"nfc"}')
    unsupported.init_kwargs = {"unordered": {"a", "b"}}
    assert tokenizer_identity(unsupported).status == "unverified"


def test_unverified_request_identity_cannot_satisfy_verified_facet_or_share_cache() -> None:
    expected = BaseBinding(
        "base", None, "config", "layers", 1,
        artifact_sha256="a" * 64, tokenizer_sha256="b" * 64,
        artifact_identity_status="verified", tokenizer_identity_status="verified",
    )
    unverified = BaseBinding(
        "base", None, "config", "layers", 1,
        artifact_sha256="", tokenizer_sha256="",
    )
    with pytest.raises(CompatibilityError, match="artifact hash differs"):
        validate_base_binding(unverified, expected)
    manifest = FacetManifest(
        facet_id="facet", version="1", base=expected,
        slots=(SlotSpec("slot", 0, 0, (BlockSpec(0),)),),
    )
    verified_key = _cache_key(
        type("Base", (), {"binding": expected, "request": BaseModelRequest("base"), "device": torch.device("cpu")})(),
        manifest,
    )
    unverified_key = _cache_key(
        type("Base", (), {"binding": unverified, "request": BaseModelRequest("base"), "device": torch.device("cpu")})(),
        manifest,
    )
    assert verified_key != unverified_key


def test_plan_provenance_marks_legacy_identity_comparison_ineligible(tmp_path) -> None:
    runtime = MixedInferenceRuntime(tmp_path / "facets")
    model = torch.nn.Linear(1, 1)
    model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
    runtime.base = LoadedBaseModel(
        request=BaseModelRequest("legacy"), model=model, tokenizer=None,
        stack=LayerStackHandle("layers", model, "layers", model.layers),
        binding=BaseBinding("legacy", None, "config", "layers", 1), device=torch.device("cpu"),
    )
    provenance = runtime._identity_provenance()
    assert provenance["comparisonEligible"] is False
    assert provenance["artifactIdentityStatus"] == "unverified"


def test_load_base_records_verified_local_identity_instead_of_request_claim(tmp_path) -> None:
    files = _files(tmp_path)
    digest = canonical_artifact_manifest(files)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.config = SimpleNamespace(
                is_encoder_decoder=False,
                model_type="tiny",
                num_hidden_layers=1,
                to_dict=lambda: {"model_type": "tiny", "num_hidden_layers": 1},
            )

        def to(self, _device):
            return self

    tokenizer = _Tokenizer('{"normalizer":"nfc"}')
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    request = BaseModelRequest.from_dict(
        {"id": str(tmp_path), "device": "cpu", "artifactSha256": digest, "artifactFiles": files}
    )
    with (
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=Model()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        loaded = load_base_model(request)
    assert loaded.binding.artifact_sha256 == digest
    assert loaded.binding.artifact_identity_status == "verified"
    assert loaded.binding.tokenizer_identity_status == "verified"

    mismatch = BaseModelRequest.from_dict(
        {"id": str(tmp_path), "device": "cpu", "artifactSha256": "0" * 64, "artifactFiles": files}
    )
    with pytest.raises(CompatibilityError, match="manifest digest differs"):
        load_base_model(mismatch)
