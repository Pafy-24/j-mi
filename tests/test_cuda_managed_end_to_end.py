from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_managed_cuda_trains_and_generates_a_tiny_facet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.skip("NVIDIA driver is required")
    cuda_uuid = subprocess.check_output(
        [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).splitlines()[0].strip()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_uuid)

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch and a GPU are required")
    pytest.importorskip("jester_rm_client")
    from jester_rm_client import current_lease
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from mixed_inference.generate.runtime import MixedInferenceRuntime
    from mixed_inference.training.trainer import train_from_config

    model_dir = tmp_path / "model"
    output = tmp_path / "facets" / "tiny-cuda" / "1"
    dataset = tmp_path / "train.jsonl"
    model_dir.mkdir()
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "hello": 4,
        "world": 5,
        "write": 6,
        "code": 7,
        "answer": 8,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )
    )
    model.save_pretrained(model_dir)
    del model
    dataset.write_text(
        json.dumps({"id": "cuda-one", "text": "hello world write code answer"}) + "\n",
        encoding="utf-8",
    )

    instance = {"metadata": {"id": "mi-cuda-smoke"}}
    config_hash = hashlib.sha256(
        json.dumps(
            instance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    effective = tmp_path / "runtime" / "config" / "effective.json"
    effective.parent.mkdir(parents=True)
    effective.write_text(
        json.dumps(
            {
                "schemaVersion": "jester-rm-effective-v1",
                "instance": instance,
                "lease": {
                    "apiVersion": "jester.rm/v1",
                    "kind": "Lease",
                    "leaseId": "mi-cuda-lease",
                    "instanceId": "mi-cuda-smoke",
                    "hostId": "hardware-host",
                    "state": "active",
                    "configSha256": config_hash,
                    "telemetryToken": "test-token",
                    "bindings": [
                        {
                            "resourceId": f"gpu-nvidia-{cuda_uuid.lower()}",
                            "runtime": {
                                "kind": "gpu",
                                "backend": "torch.cuda",
                                "cudaUuid": cuda_uuid,
                                "device": "cuda:0",
                                "memoryBudgetBytes": 1024**3,
                            },
                        }
                    ],
                    "paths": {"run": str(tmp_path / "runtime" / "run")},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JESTER_RM_REQUIRED", "1")
    monkeypatch.setenv("JESTER_RM_LEASE_FILE", str(effective))
    monkeypatch.setenv("JESTER_RM_LEASE_ID", "mi-cuda-lease")
    monkeypatch.setenv("JESTER_RM_INSTANCE_ID", "mi-cuda-smoke")
    monkeypatch.setenv("JESTER_RM_HOST_ID", "hardware-host")
    monkeypatch.setenv("JESTER_RM_DEVICE_BACKEND", "torch.cuda")
    monkeypatch.setenv("JESTER_RM_DEVICE_ID", f"gpu-nvidia-{cuda_uuid.lower()}")
    current_lease(required=True, refresh=True)

    base_model = {"id": str(model_dir), "device": "managed", "dtype": "float32"}
    config = {
        "schemaVersion": "mi-training-v1",
        "baseModel": base_model,
        "registry": str(tmp_path / "facets"),
        "facet": {
            "id": "tiny-cuda",
            "version": "1",
            "description": "Managed CUDA integration test facet",
            "dependencies": {},
            "module": {
                "type": "full-decoder-block",
                "slots": [
                    {
                        "id": "cuda-slot",
                        "afterLayer": 0,
                        "blocks": [{"sourceLayer": 0, "initializer": "identity-copy"}],
                    }
                ],
            },
        },
        "data": {"train": [{"path": str(dataset)}], "maxSequenceLength": 32},
        "optimizer": {"type": "adamw", "learningRate": 0.001},
        "scheduler": {"type": "constant"},
        "training": {
            "maxSteps": 1,
            "batchSize": 1,
            "precision": "float32",
            "logEverySteps": 1,
            "identityCheck": True,
        },
        "checkpoint": {"outputDirectory": str(output), "overwrite": True},
    }
    result = train_from_config(config)
    assert result["status"] == "succeeded"
    assert result["metrics"]["identityCheck"]["passed"] is True
    assert torch.cuda.max_memory_allocated() > 0

    runtime = MixedInferenceRuntime(tmp_path / "facets")
    loaded = runtime.load_base(base_model)
    assert loaded["device"] == "cuda:0"
    generated = runtime.generate(
        {
            "prompt": "hello",
            "facets": ["tiny-cuda@1"],
            "generation": {"max_new_tokens": 1, "do_sample": False},
        }
    )
    assert generated["resolvedFacets"] == ["tiny-cuda@1"]
    assert generated["usage"]["outputTokens"] == 1
