from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from ..core.errors import ConfigurationError, MIError, TrainingCancelled
from ..core.util import atomic_write_json, load_json, utc_now
from ..training.config import TrainingConfig
from ..training.trainer import train_from_config


class TrainingJobManager:
    def __init__(
        self,
        jobs_directory: str | Path,
        *,
        max_workers: int = 1,
        resource_lock: threading.RLock | None = None,
        max_events: int = 500,
        allow_custom_classes: bool = False,
        completion_callback: Callable[[dict[str, Any]], None] | None = None,
        start_callback: Callable[[], None] | None = None,
    ):
        if max_workers < 1:
            raise ConfigurationError("trainingWorkers must be at least 1")
        self.jobs_directory = Path(jobs_directory)
        self.jobs_directory.mkdir(parents=True, exist_ok=True)
        self.max_events = max(10, int(max_events))
        self.resource_lock = resource_lock or threading.RLock()
        self.allow_custom_classes = allow_custom_classes
        self.completion_callback = completion_callback
        self.start_callback = start_callback
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mi-training",
        )
        self._load_existing_jobs()

    def submit(self, config: dict[str, Any] | str | Path) -> dict[str, Any]:
        if not isinstance(config, (dict, str, Path)):
            raise ConfigurationError("Training job config must be an object or a JSON file path")
        job_id = uuid.uuid4().hex
        created = utc_now()
        record: dict[str, Any] = {
            "schemaVersion": "mi-training-job-v1",
            "id": job_id,
            "status": "queued",
            "createdAt": created,
            "updatedAt": created,
            "startedAt": None,
            "completedAt": None,
            "currentEvent": None,
            "events": [],
            "result": None,
            "error": None,
        }
        cancel_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = record
            self._cancel[job_id] = cancel_event
            self._persist(job_id)
            future = self._executor.submit(self._run, job_id, config, cancel_event)
            self._futures[job_id] = future
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return deepcopy(self._jobs[job_id])
            except KeyError as exc:
                raise MIError(
                    f"Training job not found: {job_id}",
                    code="job_not_found",
                    status=404,
                ) from exc

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: str(item["createdAt"]),
                reverse=True,
            )
            return [
                {
                    key: deepcopy(job.get(key))
                    for key in (
                        "id",
                        "status",
                        "createdAt",
                        "updatedAt",
                        "startedAt",
                        "completedAt",
                        "currentEvent",
                        "error",
                    )
                }
                for job in jobs
            ]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._require_job(job_id)
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                return deepcopy(job)
            self._cancel[job_id].set()
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                self._set_status(job_id, "cancelled", completedAt=utc_now())
            else:
                self._set_status(job_id, "cancelling")
            return deepcopy(self._jobs[job_id])

    def health(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                status = str(job["status"])
                counts[status] = counts.get(status, 0) + 1
            return {"counts": counts, "jobsDirectory": str(self.jobs_directory)}

    def close(self, *, wait: bool = False) -> None:
        with self._lock:
            for job_id, job in self._jobs.items():
                if job["status"] in {"queued", "running", "cancelling"}:
                    self._cancel[job_id].set()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(
        self,
        job_id: str,
        config: dict[str, Any] | str | Path,
        cancel_event: threading.Event,
    ) -> None:
        if cancel_event.is_set():
            self._set_status(job_id, "cancelled", completedAt=utc_now())
            return
        self._set_status(job_id, "waiting-for-resource")
        try:
            with self.resource_lock:
                if cancel_event.is_set():
                    raise TrainingCancelled()
                if self.start_callback is not None:
                    self.start_callback()
                self._set_status(job_id, "running", startedAt=utc_now())
                parsed = TrainingConfig.load(config)
                uses_custom_classes = bool(
                    parsed.optimizer.class_path
                    or parsed.scheduler.class_path
                    or parsed.algorithm.trainer_class_path
                )
                if uses_custom_classes and not self.allow_custom_classes:
                    raise ConfigurationError(
                        "This MI server disables custom Python training classes. Start it with "
                        "--allow-custom-training-classes only in a trusted environment."
                    )
                result = train_from_config(
                    parsed,
                    progress_callback=lambda event: self._event(job_id, event),
                    cancel_event=cancel_event,
                )
                if self.completion_callback is not None:
                    self.completion_callback(result)
            self._set_status(
                job_id,
                "succeeded",
                completedAt=utc_now(),
                result=result,
                currentEvent="training-completed",
            )
        except TrainingCancelled as exc:
            self._set_status(
                job_id,
                "cancelled",
                completedAt=utc_now(),
                error={"code": exc.code, "message": str(exc)},
            )
        except MIError as exc:
            self._set_status(
                job_id,
                "failed",
                completedAt=utc_now(),
                error={"code": exc.code, "message": str(exc)},
            )
        except BaseException as exc:
            self._set_status(
                job_id,
                "failed",
                completedAt=utc_now(),
                error={
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )

    def _event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._require_job(job_id)
            job["events"].append(deepcopy(event))
            if len(job["events"]) > self.max_events:
                job["events"] = job["events"][-self.max_events :]
            job["currentEvent"] = event.get("event")
            job["updatedAt"] = utc_now()
            self._persist(job_id)

    def _set_status(self, job_id: str, status: str, **updates: Any) -> None:
        with self._lock:
            job = self._require_job(job_id)
            job["status"] = status
            job["updatedAt"] = utc_now()
            job.update(updates)
            self._persist(job_id)

    def _require_job(self, job_id: str) -> dict[str, Any]:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise MIError(
                f"Training job not found: {job_id}",
                code="job_not_found",
                status=404,
            ) from exc

    def _persist(self, job_id: str) -> None:
        atomic_write_json(self.jobs_directory / f"{job_id}.json", self._jobs[job_id])

    def _load_existing_jobs(self) -> None:
        terminal = {"succeeded", "failed", "cancelled"}
        for path in sorted(self.jobs_directory.glob("*.json")):
            try:
                record = load_json(path)
            except MIError:
                continue
            if (
                not isinstance(record, dict)
                or record.get("schemaVersion") != "mi-training-job-v1"
                or not isinstance(record.get("id"), str)
            ):
                continue
            job_id = record["id"]
            if record.get("status") not in terminal:
                record["status"] = "failed"
                record["updatedAt"] = utc_now()
                record["completedAt"] = record["updatedAt"]
                record["error"] = {
                    "code": "server_restarted",
                    "message": "The MI service restarted before this job reached a terminal state.",
                }
            self._jobs[job_id] = record
            self._cancel[job_id] = threading.Event()
            self._persist(job_id)
