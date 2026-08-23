from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mixed_inference.core.errors import ConfigurationError
from mixed_inference.training.config import TrainingConfig, resolve_slots
from mixed_inference.training.data import read_records


class ConfigAndDataTests(unittest.TestCase):
    def test_uniform_slots_allow_different_block_counts(self) -> None:
        slots = resolve_slots(
            {
                "type": "full-decoder-block",
                "placement": {
                    "strategy": "uniform",
                    "slotCount": 3,
                    "blocksPerSlot": [1, 2, 3],
                },
            },
            12,
        )
        self.assertEqual([slot.after_layer for slot in slots], [3, 7, 11])
        self.assertEqual([len(slot.blocks) for slot in slots], [1, 2, 3])

    def test_duplicate_explicit_slots_fail(self) -> None:
        slot = {
            "id": "same",
            "afterLayer": 0,
            "blocks": [{"sourceLayer": 0}],
        }
        with self.assertRaises(ConfigurationError):
            resolve_slots({"type": "full-decoder-block", "slots": [slot, slot]}, 2)

    def test_dataset_directory_excludes_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training.jsonl").write_text('{"text":"train"}\n', encoding="utf-8")
            (root / "test.jsonl").write_text('{"text":"test"}\n', encoding="utf-8")
            (root / "facet.json").write_text('{"id":"metadata"}\n', encoding="utf-8")
            records = list(read_records(root))
        self.assertEqual([item["text"] for item in records], ["train"])

    def test_relative_local_model_is_anchored_to_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model").mkdir()
            config = root / "training.json"
            config.write_text(
                """{
                  "schemaVersion": "mi-training-v1",
                  "baseModel": {"id": "./model"},
                  "facet": {
                    "id": "facet", "version": "1",
                    "module": {"type": "full-decoder-block", "slots": [
                      {"id": "slot", "afterLayer": 0, "blocks": [{"sourceLayer": 0}]}
                    ]}
                  },
                  "data": {"train": ["train.jsonl"]},
                  "checkpoint": {"outputDirectory": "output"}
                }""",
                encoding="utf-8",
            )
            parsed = TrainingConfig.load(config)
        self.assertEqual(parsed.base_model.model_id, str((root / "model").resolve()))


if __name__ == "__main__":
    unittest.main()
