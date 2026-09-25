from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class EndToEndTests(unittest.TestCase):
    def test_tiny_llama_train_checkpoint_and_generate(self) -> None:
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        except ImportError as exc:
            self.skipTest(str(exc))

        from mixed_inference.core.registry import FacetRegistry
        from mixed_inference.generate.runtime import MixedInferenceRuntime
        from mixed_inference.training.trainer import train_from_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            output = root / "facets" / "tiny-coder" / "1"
            dataset = root / "train.jsonl"
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
                "python": 8,
                "answer": 9,
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
            dataset.write_text(
                json.dumps({"id": "one", "text": "hello world write code answer"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "schemaVersion": "mi-training-v1",
                "baseModel": {
                    "id": str(model_dir),
                    "canonicalId": "sha256:" + "b" * 64,
                    "device": "cpu",
                    "dtype": "float32",
                },
                "registry": str(root / "facets"),
                "facet": {
                    "id": "tiny-coder",
                    "version": "1",
                    "description": "End-to-end test facet",
                    "dependencies": {},
                    "module": {
                        "type": "full-decoder-block",
                        "slots": [
                            {
                                "id": "coder-slot",
                                "afterLayer": 0,
                                "blocks": [
                                    {"sourceLayer": 0, "initializer": "identity-copy"}
                                ],
                            }
                        ],
                    },
                },
                "data": {
                    "train": [{"path": str(dataset)}],
                    "maxSequenceLength": 32,
                },
                "optimizer": {"type": "adamw", "learningRate": 0.001},
                "scheduler": {"type": "constant"},
                "training": {
                    "maxSteps": 1,
                    "batchSize": 1,
                    "precision": "float32",
                    "logEverySteps": 1,
                    "identityCheck": True,
                },
                "checkpoint": {
                    "outputDirectory": str(output),
                    "overwrite": True,
                    "saveOptimizer": True,
                },
            }
            result = train_from_config(config)
            self.assertEqual(result["status"], "succeeded")
            self.assertTrue(result["metrics"]["identityCheck"]["passed"])
            self.assertTrue((output / "facet.mi.json").exists())
            self.assertTrue((output / "facet.safetensors").exists())
            manifest = json.loads((output / "facet.mi.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["base"]["modelId"], "sha256:" + "b" * 64)
            self.assertEqual(FacetRegistry(root / "facets").list()[0]["blocks"], 1)

            resumed_config = json.loads(json.dumps(config))
            resumed_config["training"]["maxSteps"] = 2
            resumed_config["checkpoint"]["resumeFrom"] = str(output)
            resumed = train_from_config(resumed_config)
            self.assertEqual(resumed["metrics"]["globalStep"], 2)
            self.assertEqual(
                resumed["metrics"]["identityCheck"]["skipped"],
                "resumed-trained-weights",
            )

            runtime = MixedInferenceRuntime(
                root / "facets",
                staging_directory=root / "mi-staging",
                staging_limit_bytes=4_000_000,
            )
            runtime.load_base(config["baseModel"])
            generated = runtime.generate(
                {
                    "prompt": "hello",
                    "facets": ["tiny-coder@1"],
                    "generation": {"max_new_tokens": 2, "do_sample": False},
                }
            )
            self.assertEqual(generated["resolvedFacets"], ["tiny-coder@1"])
            self.assertGreaterEqual(generated["usage"]["outputTokens"], 1)
            self.assertLessEqual(generated["usage"]["outputTokens"], 2)
            self.assertEqual(len(runtime.base.stack.layers), 2)


if __name__ == "__main__":
    unittest.main()
