from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..core.errors import ConfigurationError, MIError
from ..core.specs import BaseModelRequest
from ..generate.runtime import MixedInferenceRuntime
from .jobs import TrainingJobManager
from ..rm_integration import is_managed


DEFAULT_MAX_REQUEST_BYTES = 2 * 1024 * 1024


class MixedInferenceService:
    def __init__(
        self,
        registry: str | Path,
        *,
        jobs_directory: str | Path = "artifacts/training-jobs",
        training_workers: int = 1,
        allow_custom_training_classes: bool = False,
    ):
        self.resource_lock = threading.RLock()
        self.managed = is_managed()
        self.runtime = MixedInferenceRuntime(registry)
        self.jobs = TrainingJobManager(
            jobs_directory,
            max_workers=training_workers,
            resource_lock=self.resource_lock,
            allow_custom_classes=allow_custom_training_classes,
            completion_callback=lambda _: self.runtime.refresh_registry(),
            start_callback=self._training_preflight,
        )

    def health(self) -> dict[str, Any]:
        health = self.runtime.health()
        health["trainingJobs"] = self.jobs.health()
        health["resourcePolicy"] = {
            "mode": "rm-managed" if self.managed else "legacy-single-base-serialized",
            "note": (
                "Resource Manager owns placement and process scheduling"
                if self.managed
                else "Unmanaged construction is retained only for library-level compatibility"
            ),
        }
        return health

    def close(self) -> None:
        self.jobs.close(wait=False)
        with self.resource_lock:
            self.runtime.unload_base()

    def _training_preflight(self) -> None:
        if self.managed:
            raise MIError(
                "Training must be started as a separate mi.train Resource Manager instance.",
                code="rm_training_instance_required",
                status=409,
            )
        if self.runtime.base is not None:
            raise MIError(
                "Unload the inference base before starting a training job in MI v1. "
                "Use separate Resource Manager instances for concurrent train and serve.",
                code="base_loaded_for_training",
                status=409,
            )


class MixedInferenceHTTPServer(HTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        service: MixedInferenceService,
        *,
        workers: int = 4,
        max_pending: int = 16,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ):
        super().__init__(address, MixedInferenceRequestHandler)
        self.service = service
        self.max_request_bytes = max_request_bytes
        active_workers = max(1, workers)
        self.executor = ThreadPoolExecutor(
            max_workers=active_workers,
            thread_name_prefix="mi-http",
        )
        self.request_slots = threading.BoundedSemaphore(active_workers + max(0, max_pending))

    def process_request(self, request: Any, client_address: Any) -> None:
        self.request_slots.acquire()
        try:
            self.executor.submit(self._process_request, request, client_address)
        except BaseException:
            self.request_slots.release()
            self.shutdown_request(request)
            raise

    def _process_request(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            self.request_slots.release()

    def server_close(self) -> None:
        super().server_close()
        self.executor.shutdown(wait=True, cancel_futures=True)


class MixedInferenceRequestHandler(BaseHTTPRequestHandler):
    server: MixedInferenceHTTPServer

    def do_GET(self) -> None:
        try:
            path = _path(self.path)
            if path in {"/", "/health"}:
                response = self.server.service.health()
            elif path == "/facets":
                response = {
                    "registry": str(self.server.service.runtime.registry.root),
                    "facets": self.server.service.runtime.list_facets(),
                }
            elif path == "/training/jobs":
                response = {"jobs": self.server.service.jobs.list()}
            elif path.startswith("/training/jobs/"):
                job_id, action = _job_route(path)
                if action:
                    raise MIError("Not found", code="not_found", status=404)
                response = self.server.service.jobs.get(job_id)
            else:
                raise MIError("Not found", code="not_found", status=404)
            self._write_json(HTTPStatus.OK, response)
        except MIError as exc:
            self._write_error(exc)
        except BaseException as exc:
            self._write_internal_error(exc)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            path = _path(self.path)
            service = self.server.service
            if path == "/base/load":
                with service.resource_lock:
                    response = service.runtime.load_base(BaseModelRequest.from_dict(payload))
            elif path == "/base/unload":
                with service.resource_lock:
                    response = service.runtime.unload_base()
            elif path == "/registry/refresh":
                response = service.runtime.refresh_registry()
            elif path == "/facets/load":
                facets = _facet_list(payload, required=True)
                with service.resource_lock:
                    response = service.runtime.load_facets(
                        facets,
                        allow_model_id_mismatch=bool(payload.get("allowModelIdMismatch", False)),
                    )
            elif path == "/facets/unload":
                facets = _facet_list(payload, required=False)
                with service.resource_lock:
                    response = service.runtime.unload_facets(
                        facets,
                        all_facets=bool(payload.get("all", False)),
                    )
            elif path in {"/compose", "/plan"}:
                response = service.runtime.plan(_facet_list(payload, required=False))
            elif path == "/generate":
                with service.resource_lock:
                    response = service.runtime.generate(payload)
            elif path == "/training/jobs":
                if service.managed:
                    raise MIError(
                        "This managed MI server cannot launch training. Create a separate "
                        "mi.train instance through Jester Resource Manager.",
                        code="rm_training_instance_required",
                        status=409,
                    )
                response = service.jobs.submit(_training_config(payload))
                self._write_json(HTTPStatus.ACCEPTED, response)
                return
            elif path.startswith("/training/jobs/"):
                job_id, action = _job_route(path)
                if action != "cancel":
                    raise MIError("Not found", code="not_found", status=404)
                response = service.jobs.cancel(job_id)
            else:
                raise MIError("Not found", code="not_found", status=404)
            self._write_json(HTTPStatus.OK, response)
        except MIError as exc:
            self._write_error(exc)
        except json.JSONDecodeError:
            self._write_error(ConfigurationError("Invalid JSON body"))
        except UnicodeDecodeError:
            self._write_error(ConfigurationError("JSON body must be UTF-8"))
        except BaseException as exc:
            self._write_internal_error(exc)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Content-Length must be an integer") from exc
        if length <= 0:
            raise ConfigurationError("Missing JSON body")
        if length > self.server.max_request_bytes:
            raise MIError(
                f"JSON body exceeds {self.server.max_request_bytes} bytes",
                code="request_too_large",
                status=413,
            )
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _write_error(self, error: MIError) -> None:
        self._write_json(
            error.status,
            {"error": {"code": error.code, "message": str(error)}},
        )

    def _write_internal_error(self, error: BaseException) -> None:
        self._write_json(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "error": {
                    "code": "internal_error",
                    "message": f"{type(error).__name__}: {error}",
                }
            },
        )

    def _write_json(self, status: int | HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _path(value: str) -> str:
    return urlparse(value).path.rstrip("/") or "/"


def _job_route(path: str) -> tuple[str, str | None]:
    parts = [unquote(part) for part in path.split("/") if part]
    if len(parts) not in {3, 4} or parts[:2] != ["training", "jobs"]:
        raise MIError("Not found", code="not_found", status=404)
    return parts[2], parts[3] if len(parts) == 4 else None


def _facet_list(payload: Any, *, required: bool) -> list[str]:
    if isinstance(payload, str):
        result = [payload]
    elif isinstance(payload, list):
        result = payload
    elif isinstance(payload, dict):
        value = payload.get("facets")
        if value is None and payload.get("facet") is not None:
            value = [payload["facet"]]
        result = [] if value is None else value
    else:
        result = []
    if not isinstance(result, list) or not all(
        isinstance(item, str) and item.strip() for item in result
    ):
        raise ConfigurationError("facets must be a JSON array of non-empty strings")
    if required and not result:
        raise ConfigurationError("At least one facet is required")
    return [item.strip() for item in result]


def _training_config(payload: Any) -> dict[str, Any] | str:
    if not isinstance(payload, dict):
        raise ConfigurationError("Training job body must be a JSON object")
    if isinstance(payload.get("config"), dict):
        return payload["config"]
    if isinstance(payload.get("configPath"), str) and payload["configPath"].strip():
        return payload["configPath"].strip()
    if payload.get("schemaVersion") == "mi-training-v1":
        return payload
    raise ConfigurationError(
        "Training job requires config, configPath, or a direct mi-training-v1 document"
    )


def serve(
    *,
    host: str,
    port: int,
    registry: str | Path,
    base_model: BaseModelRequest | None = None,
    jobs_directory: str | Path = "artifacts/training-jobs",
    workers: int = 4,
    training_workers: int = 1,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    allow_custom_training_classes: bool = False,
) -> None:
    service = MixedInferenceService(
        registry,
        jobs_directory=jobs_directory,
        training_workers=training_workers,
        allow_custom_training_classes=allow_custom_training_classes,
    )
    if base_model is not None:
        with service.resource_lock:
            service.runtime.load_base(base_model)
    server = MixedInferenceHTTPServer(
        (host, port),
        service,
        workers=workers,
        max_request_bytes=max_request_bytes,
    )
    print(f"Jester MI component 2/4 ready on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()
