from __future__ import annotations

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mixed_inference.cli import main


class RMConfigMigrationTests(unittest.TestCase):
    def test_processing_unit_inventory_has_moved_to_rm(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            self.assertEqual(main(["--PUs"]), 2)
        self.assertIn("jester-rm resources list", stderr.getvalue())

    def test_migrate_rm_config_sets_managed_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training.json"
            output = Path(directory) / "managed.json"
            source.write_text(
                json.dumps(
                    {
                        "schemaVersion": "mi-training-v1",
                        "baseModel": {"id": "small-model", "device": "cuda:0"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                main(["migrate-rm-config", str(source), str(output)]), 0
            )
            migrated = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(migrated["baseModel"]["device"], "managed")
            self.assertEqual(
                json.loads(source.read_text(encoding="utf-8"))["baseModel"]["device"],
                "cuda:0",
            )


if __name__ == "__main__":
    unittest.main()
