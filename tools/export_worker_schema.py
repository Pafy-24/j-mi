from __future__ import annotations

import json
from pathlib import Path

from mixed_inference.worker.models import MIWorkerTask


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "schemas" / "mi-worker-task-v1.schema.json"
    schema = MIWorkerTask.model_json_schema(by_alias=True)
    schema["$id"] = "https://jester.local/schemas/mi/mi-worker-task-v1.schema.json"
    target.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
