from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from mixed_inference.core.errors import CompatibilityError, MIError
from mixed_inference.worker.models import MIWorkerTask
from mixed_inference.worker.service import MIWorkerService


class WorkerGradientTests(unittest.TestCase):
    def test_two_worker_token_weighted_gradient_matches_combined_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            services = [
                MIWorkerService(root / "worker-a", require_rm=False),
                MIWorkerService(root / "worker-b", require_rm=False),
                MIWorkerService(root / "worker-combined", require_rm=False),
            ]
            sessions = ["worker-a", "worker-b", "worker-combined"]
            summaries = [
                service.create_session(
                    {
                        "sessionId": session_id,
                        "trainingConfig": config,
                        "weightVersion": "initial",
                        "transport": {"dtype": "float32", "compression": "none"},
                    }
                )
                for service, session_id in zip(services, sessions)
            ]
            rounds = [
                services[0].compute_round(
                    sessions[0],
                    0,
                    {"weightVersion": summaries[0]["weightVersion"], "indices": [0]},
                ),
                services[1].compute_round(
                    sessions[1],
                    0,
                    {"weightVersion": summaries[1]["weightVersion"], "indices": [1]},
                ),
                services[2].compute_round(
                    sessions[2],
                    0,
                    {
                        "weightVersion": summaries[2]["weightVersion"],
                        "indices": [0, 1],
                        "batchSize": 1,
                    },
                ),
            ]
            paths = [
                service.gradient_file(session_id, 0)[0]
                for service, session_id in zip(services, sessions)
            ]
            gradients = [load_file(str(path), device="cpu") for path in paths]
            total_tokens = rounds[0]["supervisedTokens"] + rounds[1]["supervisedTokens"]
            for name in gradients[2]:
                aggregate = (
                    gradients[0][name] * rounds[0]["supervisedTokens"]
                    + gradients[1][name] * rounds[1]["supervisedTokens"]
                ) / total_tokens
                self.assertTrue(
                    torch.allclose(aggregate, gradients[2][name], atol=2e-6, rtol=2e-5),
                    name,
                )

    def test_child_only_gradient_round_is_idempotent_and_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            service = MIWorkerService(root / "worker", require_rm=False)
            summary = service.create_session(
                {
                    "sessionId": "session-one",
                    "trainingConfig": config,
                    "weightVersion": "initial",
                    "transport": {"dtype": "float32", "compression": "none"},
                }
            )
            session = service._sessions["session-one"]
            base_before = {
                name: tensor.detach().clone()
                for name, tensor in session.engine.base.model.named_parameters()
            }
            request = {
                "weightVersion": summary["weightVersion"],
                "indices": [0, 1],
                "batchSize": 2,
            }
            first = service.compute_round("session-one", 0, request)
            second = service.compute_round("session-one", 0, request)
            self.assertEqual(first["gradientSha256"], second["gradientSha256"])
            gradient_path, _ = service.gradient_file("session-one", 0)
            gradients = load_file(str(gradient_path), device="cpu")
            self.assertEqual(set(gradients), set(summary["engine"]["tensorSpecs"]))
            self.assertTrue(all(torch.isfinite(item).all() for item in gradients.values()))
            for name, parameter in session.engine.base.model.named_parameters():
                self.assertTrue(torch.equal(base_before[name], parameter.detach()), name)
                self.assertIsNone(parameter.grad)
            acknowledged = service.ack_round(
                "session-one",
                0,
                {"gradientSha256": first["gradientSha256"]},
            )
            self.assertFalse(acknowledged["gradientRetained"])
            self.assertFalse(gradient_path.exists())
            repeated_ack = service.ack_round(
                "session-one",
                0,
                {"gradientSha256": first["gradientSha256"]},
            )
            self.assertEqual(repeated_ack["state"], "acked")
            with self.assertRaises(MIError) as conflict:
                service.compute_round(
                    "session-one",
                    0,
                    {"weightVersion": "initial", "indices": [1]},
                )
            self.assertEqual(conflict.exception.code, "round_idempotency_conflict")
            with self.assertRaises(MIError) as stale:
                service.compute_round(
                    "session-one",
                    1,
                    {"weightVersion": "stale", "indices": [0]},
                )
            self.assertEqual(stale.exception.code, "stale_weight_version")

    def test_weight_upload_rejects_nonfinite_and_deleted_session_stays_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MIWorkerService(root / "worker", require_rm=False)
            summary = service.create_session(
                {
                    "sessionId": "finite-session",
                    "trainingConfig": self._config(root),
                    "weightVersion": "initial",
                }
            )
            weight_path, _ = service.weight_file("finite-session", "initial")
            state = load_file(str(weight_path), device="cpu")
            first = next(iter(state))
            state[first] = torch.full_like(state[first], float("nan"))
            invalid = root / "invalid.safetensors"
            save_file(state, str(invalid))
            body = invalid.read_bytes()
            from mixed_inference.core.util import sha256_file

            with self.assertRaises(CompatibilityError):
                service.upload_weights(
                    "finite-session",
                    "next",
                    body,
                    offset=0,
                    total_bytes=len(body),
                    expected_sha256=sha256_file(invalid),
                    final=True,
                )
            service.delete_session("finite-session")
            recovered = MIWorkerService(root / "worker", require_rm=False)
            self.assertEqual(recovered.list_sessions(), [])

    def test_identifiers_and_fallback_are_fail_closed(self) -> None:
        with self.assertRaises(ValidationError):
            MIWorkerTask.model_validate(
                {
                    "sessionId": "../escape",
                    "trainingConfig": {},
                }
            )
        with self.assertRaises(ValidationError):
            MIWorkerTask.model_validate(
                {
                    "sessionId": "safe",
                    "trainingConfig": {},
                    "requestedBackend": "windowsml.vitisai",
                    "effectiveBackend": "cpu",
                }
            )

    def test_rm_artifact_bindings_are_applied_and_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            model = Path(config["baseModel"]["id"])
            dataset = Path(config["data"]["train"][0]["path"])
            registry = Path(config["registry"])
            registry.mkdir(parents=True, exist_ok=True)
            digests = {
                "model": "sha256:" + "1" * 64,
                "dataset": "sha256:" + "2" * 64,
                "registry": "sha256:" + "3" * 64,
            }
            manifest = root / "lease-artifacts.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": "jester-rm-runtime-artifacts-v1",
                        "entries": [
                            {
                                "digest": digests["model"],
                                "kind": "directory",
                                "checkoutPath": str(model),
                            },
                            {
                                "digest": digests["dataset"],
                                "kind": "file",
                                "checkoutPath": str(dataset),
                            },
                            {
                                "digest": digests["registry"],
                                "kind": "directory",
                                "checkoutPath": str(registry),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = {
                "sessionId": "cas-session",
                "trainingConfig": {
                    **config,
                    "baseModel": {**config["baseModel"], "id": "rm-artifact-pending"},
                    "registry": "rm-artifact-pending",
                    "data": {
                        **config["data"],
                        "train": [{"path": "rm-artifact-pending"}],
                    },
                },
                "artifactBindings": [
                    {"target": "/baseModel/id", "digest": digests["model"]},
                    {
                        "target": "/data/train/0/path",
                        "digest": digests["dataset"],
                        "logicalPath": "training.jsonl",
                    },
                    {"target": "/registry", "digest": digests["registry"]},
                ],
            }
            from unittest.mock import patch

            worker_root = root / "worker"
            with patch.dict(
                "os.environ", {"JESTER_RM_ARTIFACT_MANIFEST": str(manifest)}
            ):
                service = MIWorkerService(worker_root, require_rm=False)
                created = service.create_session(task)
                self.assertEqual(created["id"], "cas-session")
                recovered = MIWorkerService(worker_root, require_rm=False)
                self.assertEqual(recovered.session("cas-session")["state"], "idle")

    @staticmethod
    def _config(root: Path) -> dict:
        model_dir = root / "model"
        if not model_dir.exists():
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
            torch.manual_seed(9)
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
        dataset = root / "train.jsonl"
        dataset.write_text(
            json.dumps({"id": "one", "text": "hello world write code answer"})
            + "\n"
            + json.dumps({"id": "two", "text": "write code answer"})
            + "\n",
            encoding="utf-8",
        )
        return {
            "schemaVersion": "mi-training-v1",
            "baseModel": {"id": str(model_dir), "device": "cpu", "dtype": "float32"},
            "registry": str(root / "facets"),
            "facet": {
                "id": "worker-test",
                "version": "1",
                "dependencies": {},
                "module": {
                    "type": "full-decoder-block",
                    "slots": [
                        {
                            "id": "slot",
                            "afterLayer": 0,
                            "blocks": [
                                {"sourceLayer": 0, "initializer": "identity-copy"}
                            ],
                        }
                    ],
                },
            },
            "data": {"train": [{"path": str(dataset)}], "maxSequenceLength": 32},
            "training": {"maxSteps": 1, "batchSize": 2, "precision": "float32"},
            "checkpoint": {"outputDirectory": str(root / "unused")},
        }


if __name__ == "__main__":
    unittest.main()
