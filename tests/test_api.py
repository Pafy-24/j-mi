from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from mixed_inference.api.jobs import TrainingJobManager
from mixed_inference.api.server import MixedInferenceHTTPServer, MixedInferenceService


def minimal_config(root: Path) -> dict[str, object]:
    return {
        "schemaVersion": "mi-training-v1",
        "baseModel": {"id": "unused-local-model"},
        "registry": str(root / "facets"),
        "facet": {
            "id": "test-facet",
            "version": "1",
            "module": {
                "type": "full-decoder-block",
                "slots": [
                    {
                        "id": "slot",
                        "afterLayer": 0,
                        "blocks": [{"sourceLayer": 0}],
                    }
                ],
            },
        },
        "data": {"train": [{"path": str(root / "unused.jsonl")}]},
        "checkpoint": {"outputDirectory": str(root / "output")},
    }


class TrainingJobTests(unittest.TestCase):
    def test_job_lifecycle_persists_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = TrainingJobManager(root / "jobs")
            with patch(
                "mixed_inference.api.jobs.train_from_config",
                return_value={"status": "succeeded", "outputDirectory": "fake"},
            ):
                submitted = manager.submit(minimal_config(root))
                deadline = time.time() + 2
                current = submitted
                while current["status"] not in {"succeeded", "failed"} and time.time() < deadline:
                    time.sleep(0.01)
                    current = manager.get(submitted["id"])
            manager.close(wait=True)
            self.assertEqual(current["status"], "succeeded")
            self.assertTrue((root / "jobs" / f"{submitted['id']}.json").exists())


class HTTPTests(unittest.TestCase):
    def test_health_compose_and_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MixedInferenceService(root / "facets", jobs_directory=root / "jobs")
            server = MixedInferenceHTTPServer(("127.0.0.1", 0), service, workers=2)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                health = self._request(url + "/health", method="GET")
                self.assertEqual(health["component"], 2)
                plan = self._request(url + "/compose", {"facets": []})
                self.assertEqual(plan["expandedBlockCount"], 0)
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self._request(url + "/generate", {"prompt": "hello", "facets": []})
                body = json.loads(raised.exception.read().decode("utf-8"))
                self.assertEqual(body["error"]["code"], "base_not_loaded")
            finally:
                server.shutdown()
                server.server_close()
                service.close()
                thread.join(timeout=2)

    def test_managed_server_refuses_in_process_training_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "mixed_inference.api.server.is_managed", return_value=True
            ):
                service = MixedInferenceService(
                    root / "facets", jobs_directory=root / "jobs"
                )
            server = MixedInferenceHTTPServer(("127.0.0.1", 0), service)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self._request(url + "/training/jobs", minimal_config(root))
                self.assertEqual(raised.exception.code, 409)
                body = json.loads(raised.exception.read().decode("utf-8"))
                self.assertEqual(
                    body["error"]["code"], "rm_training_instance_required"
                )
                self.assertEqual(service.jobs.list(), [])
            finally:
                server.shutdown()
                server.server_close()
                service.close()
                thread.join(timeout=2)

    @staticmethod
    def _request(url: str, payload: object | None = None, *, method: str = "POST") -> object:
        data = None if method == "GET" else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
