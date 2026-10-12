from __future__ import annotations

import json
import copy
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import CompatibilityError, ConfigurationError, MIError
from ..core.util import atomic_write_json, sha256_file, sha256_json, utc_now
from .engine import FacetGradientEngine
from .models import MIWorkerTask, RoundRequest


@dataclass
class WorkerSession:
    task: MIWorkerTask
    root: Path
    engine: FacetGradientEngine
    weight_version: str
    weight_sha256: str
    created_at: str = field(default_factory=utc_now)
    state: str = "idle"
    rounds: dict[int, dict[str, Any]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


class MIWorkerService:
    MAX_TRANSFER_BYTES = 8 * 1024**3

    def __init__(self, root: str | Path | None = None, *, require_rm: bool = True):
        self.lease: Any | None = None
        if require_rm:
            try:
                from jester_rm_client import require_lease

                self.lease = require_lease()
            except ImportError as exc:
                raise ConfigurationError(
                    "RM_LEASE_REQUIRED: install jester-rm-client and launch mi.worker through RM"
                ) from exc
        elif os.environ.get("JESTER_RM_LEASE_FILE"):
            try:
                from jester_rm_client import current_lease

                self.lease = current_lease(required=False)
            except Exception:
                self.lease = None
        if root is None:
            if self.lease is None:
                raise ConfigurationError("MI worker root is required outside RM")
            root = self.lease.run_directory / "mi-worker"
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_root = self.root / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, WorkerSession] = {}
        self._lock = threading.RLock()
        self._compute_lock = threading.Lock()
        self._recovery_errors: list[dict[str, str]] = []
        self._recover_sessions()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "component": 2,
            "role": "mi.worker",
            "managed": self.lease is not None,
            "sessions": len(self._sessions),
            "recoveryErrors": self._recovery_errors,
            "capabilities": self.capabilities(),
        }

    def capabilities(self) -> dict[str, Any]:
        discovered: list[dict[str, Any]] = []
        effective_backend = os.environ.get("JESTER_RM_DEVICE_BACKEND")
        try:
            import torch

            discovered.append(
                {
                    "backend": "cpu",
                    "discovered": True,
                    "allocatable": True,
                    "workerCompatible": True,
                    "autogradCapable": True,
                    "hardwareVerified": True,
                }
            )
            # RM has already performed host discovery and bound this worker.
            # A CPU-bound worker must not initialize another accelerator runtime
            # merely to report capabilities that it cannot allocate under its lease.
            if effective_backend != "cpu" and torch.cuda.is_available():
                discovered.append(
                    {
                        "backend": "torch.cuda",
                        "discovered": True,
                        "allocatable": True,
                        "workerCompatible": True,
                        "autogradCapable": True,
                        "hardwareVerified": True,
                        "deviceCount": torch.cuda.device_count(),
                    }
                )
            if getattr(torch.version, "hip", None):
                discovered.append(
                    {
                        "backend": "torch.rocm",
                        "discovered": True,
                        "allocatable": True,
                        "workerCompatible": True,
                        "autogradCapable": True,
                        "hardwareVerified": False,
                    }
                )
        except ImportError:
            discovered.append(
                {
                    "backend": "cpu",
                    "discovered": True,
                    "allocatable": False,
                    "workerCompatible": False,
                    "autogradCapable": False,
                    "hardwareVerified": False,
                    "reason": "torch is not installed",
                }
            )
        try:
            import torch_xla  # noqa: F401

            discovered.append(
                {
                    "backend": "torch_xla",
                    "discovered": True,
                    "allocatable": bool(os.environ.get("PJRT_DEVICE")),
                    "workerCompatible": False,
                    "autogradCapable": True,
                    "hardwareVerified": False,
                    "reason": "MI dynamic block engine has not completed XLA qualification",
                }
            )
        except ImportError:
            pass
        for backend in ("windowsml.dml", "windowsml.vitisai"):
            if os.environ.get("JESTER_RM_REQUESTED_BACKEND") == backend:
                discovered.append(
                    {
                        "backend": backend,
                        "discovered": True,
                        "allocatable": True,
                        "workerCompatible": False,
                        "autogradCapable": False,
                        "hardwareVerified": False,
                    }
                )
        return {
            "schemaVersion": "mi-worker-capabilities-v1",
            "backends": discovered,
            "transport": {
                "formats": ["safetensors"],
                "compression": ["none", "zstd"],
                "dtypes": ["float32", "float16", "bfloat16"],
                "rangeRequests": True,
            },
        }

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = MIWorkerTask.model_validate(payload)
        self._validate_lease_backend(task)
        with self._lock:
            existing = self._sessions.get(task.session_id)
            request_hash = sha256_json(task.api_dict())
            if existing is not None:
                current_hash = sha256_json(existing.task.api_dict())
                if current_hash != request_hash:
                    raise MIError(
                        f"Session {task.session_id} already exists with different configuration",
                        code="session_conflict",
                        status=409,
                    )
                return self.session(task.session_id)
            session_root = self._safe_session_root(task.session_id)
            if session_root.exists():
                # A worker can be terminated while the model is loading, before
                # session.json is durable.  That directory has no recoverable
                # state and must not make every Trainer retry fail with
                # FileExistsError.  Conversely, never overwrite a persisted
                # session which failed recovery: it may contain the only copy
                # of an acknowledged weight version.
                if (session_root / "deleted.json").is_file():
                    shutil.rmtree(session_root)
                elif (session_root / "session.json").is_file():
                    raise MIError(
                        f"Session {task.session_id} exists on disk but could not be recovered",
                        code="session_recovery_failed",
                        status=409,
                    )
                else:
                    shutil.rmtree(session_root)
            session_root.mkdir(parents=True, exist_ok=False)
            try:
                training_config = copy.deepcopy(task.training_config)
                self._apply_artifact_bindings(training_config, task)
                if self.lease is not None:
                    base_model = training_config.get("baseModel")
                    if not isinstance(base_model, dict):
                        raise ConfigurationError(
                            "trainingConfig.baseModel must be an object"
                        )
                    base_model["device"] = "managed"
                engine = FacetGradientEngine(training_config)
                weights_path = session_root / "weights" / f"{task.weight_version}.safetensors"
                print(
                    json.dumps(
                        {
                            "event": "mi.worker.session.stage",
                            "sessionId": task.session_id,
                            "stage": "saving-initial-weights",
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                weight_sha = engine.save_weights(
                    weights_path, dtype=task.transport.dtype
                )
                print(
                    json.dumps(
                        {
                            "event": "mi.worker.session.stage",
                            "sessionId": task.session_id,
                            "stage": "initial-weights-saved",
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                session = WorkerSession(
                    task=task,
                    root=session_root,
                    engine=engine,
                    weight_version=task.weight_version,
                    weight_sha256=weight_sha,
                )
                self._sessions[task.session_id] = session
                self._persist(session)
                return self.session(task.session_id)
            except BaseException:
                shutil.rmtree(session_root, ignore_errors=True)
                raise

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.session(session_id) for session_id in sorted(self._sessions)]

    def session(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        with session.lock:
            return {
                "schemaVersion": "mi-worker-session-v1",
                "id": session.task.session_id,
                "state": session.state,
                "weightVersion": session.weight_version,
                "weightSha256": session.weight_sha256,
                "createdAt": session.created_at,
                "task": session.task.api_dict(),
                "engine": session.engine.summary(),
                "rounds": [
                    session.rounds[index] for index in sorted(session.rounds)
                ],
            }

    def delete_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._get(session_id)
            if session.state == "running":
                raise MIError(
                    "A running session must be cancelled before deletion",
                    code="session_running",
                    status=409,
                )
            # Explicitly destroy the model/composition graph while creation of
            # another session is excluded by the service lock.  Merely popping
            # the dictionary entry leaves cyclic PyTorch references alive and
            # can exhaust the Windows commit limit on a parent -> child job.
            session.engine.close()
            self._sessions.pop(session_id)
            (session.root / "session.json").unlink(missing_ok=True)
            atomic_write_json(
                session.root / "deleted.json",
                {"id": session_id, "deletedAt": utc_now()},
            )
        return {"id": session_id, "deleted": True, "artifactsRetained": True}

    def weight_file(self, session_id: str, version: str) -> tuple[Path, str]:
        version = self._safe_version(version)
        session = self._get(session_id)
        if version != session.weight_version:
            raise MIError(
                f"Weight version {version} is stale; current version is {session.weight_version}",
                code="stale_weight_version",
                status=409,
            )
        path = session.root / "weights" / f"{version}.safetensors"
        if not path.is_file():
            raise MIError("Weight artifact is unavailable", code="weight_not_found", status=404)
        return path, session.weight_sha256

    def upload_weights(
        self,
        session_id: str,
        version: str,
        body: bytes,
        *,
        offset: int,
        total_bytes: int,
        expected_sha256: str,
        final: bool,
        encoding: str = "none",
    ) -> dict[str, Any]:
        version = self._safe_version(version)
        expected_sha256 = expected_sha256.lower().removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ConfigurationError("X-Content-SHA256 must be a SHA-256 digest")
        if total_bytes < 0 or total_bytes > self.MAX_TRANSFER_BYTES:
            raise MIError(
                "Declared weight transfer exceeds the worker limit",
                code="weight_transfer_too_large",
                status=413,
            )
        session = self._get(session_id)
        with session.lock:
            previous_version = session.weight_version
            weights = session.root / "weights"
            weights.mkdir(parents=True, exist_ok=True)
            final_path = weights / f"{version}.safetensors"
            staging = weights / f".{version}.staging"
            if final_path.exists():
                actual = sha256_file(final_path)
                if actual == expected_sha256 and version == session.weight_version:
                    return {
                        "version": version,
                        "sha256": actual,
                        "complete": True,
                        "idempotent": True,
                    }
                raise MIError(
                    "The same weight version already exists with another hash",
                    code="weight_version_conflict",
                    status=409,
                )
            current_size = staging.stat().st_size if staging.exists() else 0
            if offset != current_size:
                raise MIError(
                    f"Upload offset {offset} does not match staged size {current_size}",
                    code="upload_offset_conflict",
                    status=409,
                )
            if offset + len(body) > total_bytes:
                raise ConfigurationError("Weight upload exceeds declared totalBytes")
            mode = "ab" if staging.exists() else "wb"
            with staging.open(mode) as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            complete = final or staging.stat().st_size == total_bytes
            if not complete:
                return {
                    "version": version,
                    "complete": False,
                    "nextOffset": staging.stat().st_size,
                    "totalBytes": total_bytes,
                }
            if staging.stat().st_size != total_bytes:
                raise MIError(
                    "Final weight body is incomplete",
                    code="incomplete_weight_upload",
                    status=409,
                )
            decoded = staging
            temporary_decoded: Path | None = None
            if encoding == "zstd":
                try:
                    import zstandard
                except ImportError as exc:
                    raise ConfigurationError(
                        "zstandard is required for Content-Encoding: zstd"
                    ) from exc
                temporary_decoded = weights / f".{version}.decoded"
                with staging.open("rb") as source, temporary_decoded.open("wb") as target:
                    zstandard.ZstdDecompressor().copy_stream(source, target)
                decoded = temporary_decoded
            elif encoding not in {"", "none", "identity"}:
                raise ConfigurationError(
                    f"Unsupported weight Content-Encoding: {encoding}"
                )
            actual = sha256_file(decoded)
            if actual != expected_sha256:
                if temporary_decoded is not None:
                    temporary_decoded.unlink(missing_ok=True)
                raise MIError(
                    f"Weight hash mismatch: {actual} != {expected_sha256}",
                    code="weight_hash_mismatch",
                    status=409,
                )
            try:
                session.engine.load_weights(decoded)
            except BaseException:
                if temporary_decoded is not None:
                    temporary_decoded.unlink(missing_ok=True)
                staging.unlink(missing_ok=True)
                raise
            if temporary_decoded is not None:
                os.replace(temporary_decoded, final_path)
                staging.unlink(missing_ok=True)
            else:
                os.replace(staging, final_path)
            session.weight_version = version
            session.weight_sha256 = actual
            self._persist(session)
            if previous_version != version:
                previous_path = weights / f"{previous_version}.safetensors"
                previous_path.unlink(missing_ok=True)
                previous_path.with_suffix(previous_path.suffix + ".zst").unlink(
                    missing_ok=True
                )
            return {
                "version": version,
                "sha256": actual,
                "complete": True,
                "nextOffset": total_bytes,
            }

    def upload_status(
        self, session_id: str, version: str
    ) -> dict[str, Any]:
        version = self._safe_version(version)
        session = self._get(session_id)
        weights = session.root / "weights"
        final_path = weights / f"{version}.safetensors"
        staging = weights / f".{version}.staging"
        if final_path.is_file():
            return {
                "version": version,
                "complete": True,
                "sha256": sha256_file(final_path),
                "nextOffset": final_path.stat().st_size,
            }
        return {
            "version": version,
            "complete": False,
            "nextOffset": staging.stat().st_size if staging.is_file() else 0,
        }

    def compute_round(
        self, session_id: str, round_index: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if round_index < 0:
            raise ConfigurationError("round must be non-negative")
        request = RoundRequest.model_validate(payload)
        session = self._get(session_id)
        request_hash = sha256_json(request.api_dict())
        with session.lock:
            existing = session.rounds.get(round_index)
            if existing is not None:
                if existing["requestSha256"] != request_hash:
                    raise MIError(
                        "The same round already exists with a different request hash",
                        code="round_idempotency_conflict",
                        status=409,
                    )
                return existing
            if request.weight_version != session.weight_version:
                raise MIError(
                    f"Round expects weight version {request.weight_version}; current is {session.weight_version}",
                    code="stale_weight_version",
                    status=409,
                )
            session.state = "running"
            record = {
                "round": round_index,
                "state": "collecting",
                "weightVersion": request.weight_version,
                "requestSha256": request_hash,
                "startedAt": utc_now(),
            }
            session.rounds[round_index] = record
            self._persist(session)
        gradient_path = session.root / "rounds" / f"{round_index:08d}.safetensors"
        try:
            with self._compute_lock:
                result = session.engine.compute(
                    gradient_path,
                    indices=request.indices,
                    offset=request.offset,
                    count=request.count,
                    batch_size=request.batch_size,
                    transport_dtype=session.task.transport.dtype,
                )
            with session.lock:
                record.update(
                    {
                        "state": "collected",
                        "completedAt": utc_now(),
                        "gradientSha256": result.gradient_sha256,
                        "gradientBytes": result.gradient_path.stat().st_size,
                        "supervisedTokens": result.supervised_tokens,
                        "examples": result.examples,
                        "meanLoss": result.mean_loss,
                        "tensorCount": result.tensor_count,
                        "tensorSpecs": result.tensor_specs,
                    }
                )
                session.state = "idle"
                self._persist(session)
                return dict(record)
        except BaseException as exc:
            with session.lock:
                record.update(
                    {
                        "state": "failed",
                        "completedAt": utc_now(),
                        "error": {
                            "code": getattr(exc, "code", "gradient_compute_failed"),
                            "detail": f"{type(exc).__name__}: {exc}",
                        },
                    }
                )
                session.state = "idle"
                self._persist(session)
            raise

    def round(self, session_id: str, round_index: int) -> dict[str, Any]:
        session = self._get(session_id)
        try:
            return dict(session.rounds[round_index])
        except KeyError as exc:
            raise MIError("Round not found", code="round_not_found", status=404) from exc

    def gradient_file(
        self, session_id: str, round_index: int
    ) -> tuple[Path, str]:
        record = self.round(session_id, round_index)
        if record.get("state") not in {"collected", "acked"}:
            raise MIError(
                "Gradient is not ready", code="gradient_not_ready", status=409
            )
        session = self._get(session_id)
        path = session.root / "rounds" / f"{round_index:08d}.safetensors"
        if not path.is_file():
            raise MIError("Gradient file is missing", code="gradient_not_found", status=404)
        return path, str(record["gradientSha256"])

    def ack_round(
        self, session_id: str, round_index: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        session = self._get(session_id)
        with session.lock:
            record = self.round(session_id, round_index)
            expected = str(payload.get("gradientSha256", ""))
            if expected and expected != record.get("gradientSha256"):
                raise MIError(
                    "Acknowledgement gradient hash differs",
                    code="gradient_hash_conflict",
                    status=409,
                )
            record["state"] = "acked"
            record["ackedAt"] = utc_now()
            gradient_path = session.root / "rounds" / f"{round_index:08d}.safetensors"
            gradient_path.unlink(missing_ok=True)
            gradient_path.with_suffix(gradient_path.suffix + ".zst").unlink(
                missing_ok=True
            )
            record["gradientRetained"] = False
            session.rounds[round_index] = record
            self._persist(session)
            return record

    def cancel(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        session.engine.cancel_event.set()
        session.state = "draining"
        self._persist(session)
        return {"id": session_id, "state": session.state, "cancelRequested": True}

    def _get(self, session_id: str) -> WorkerSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise MIError(
                    f"Session {session_id!r} not found",
                    code="session_not_found",
                    status=404,
                ) from exc

    def _safe_session_root(self, session_id: str) -> Path:
        target = (self.sessions_root / session_id).resolve()
        try:
            target.relative_to(self.sessions_root)
        except ValueError as exc:
            raise ConfigurationError("sessionId escapes the worker volume") from exc
        return target

    @staticmethod
    def _safe_version(version: str) -> str:
        normalized = version.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", normalized):
            raise ConfigurationError("weight version contains unsupported characters")
        return normalized

    def _persist(self, session: WorkerSession) -> None:
        atomic_write_json(
            session.root / "session.json",
            {
                "schemaVersion": "mi-worker-session-state-v1",
                "id": session.task.session_id,
                "state": session.state,
                "createdAt": session.created_at,
                "weightVersion": session.weight_version,
                "weightSha256": session.weight_sha256,
                "task": session.task.api_dict(),
                "rounds": {
                    str(index): record for index, record in session.rounds.items()
                },
            },
        )

    def _validate_lease_backend(self, task: MIWorkerTask) -> None:
        if self.lease is None:
            return
        lease = self.lease.lease
        requested = str(lease.get("requestedBackend", "cpu"))
        effective = str(lease.get("effectiveBackend", "cpu"))
        reason = lease.get("fallbackReason")
        if task.requested_backend != requested or task.effective_backend != effective:
            raise MIError(
                "Worker task backend does not match the RM lease",
                code="lease_backend_mismatch",
                status=409,
            )
        if requested != effective and (
            not task.allow_cpu_fallback or reason != task.fallback_reason
        ):
            raise MIError(
                "CPU fallback was not explicitly acknowledged by the Trainer task",
                code="cpu_fallback_not_acknowledged",
                status=409,
            )

    def _apply_artifact_bindings(
        self, config: dict[str, Any], task: MIWorkerTask
    ) -> None:
        if not task.artifact_bindings:
            return
        manifest_path = os.environ.get("JESTER_RM_ARTIFACT_MANIFEST")
        if not manifest_path:
            raise ConfigurationError(
                "artifactBindings require an RM artifact manifest"
            )
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot read RM artifact manifest: {exc}") from exc
        entries = {
            str(item.get("digest")): item
            for item in manifest.get("entries", [])
            if isinstance(item, dict)
        }
        for binding in task.artifact_bindings:
            entry = entries.get(binding.digest)
            if entry is None:
                raise CompatibilityError(
                    f"Artifact {binding.digest} is not present in the worker lease"
                )
            checkout = Path(str(entry.get("checkoutPath", ""))).resolve()
            if not checkout.exists():
                raise CompatibilityError(
                    f"Artifact checkout is unavailable for {binding.digest}"
                )
            resolved = checkout
            entry_kind = str(entry.get("kind", ""))
            if binding.logical_path and entry_kind == "directory":
                resolved = (checkout / binding.logical_path).resolve()
                try:
                    resolved.relative_to(checkout)
                except ValueError as exc:
                    raise ConfigurationError("artifact logicalPath escapes checkout") from exc
            elif binding.logical_path and entry_kind not in {"", "file"}:
                raise ConfigurationError(
                    f"Unsupported RM artifact kind for logicalPath: {entry_kind}"
                )
            if not resolved.exists():
                raise CompatibilityError(
                    f"Artifact logical path is unavailable: {binding.logical_path}"
                )
            parts = binding.target.strip("/").split("/")
            target: Any = config
            for part in parts[:-1]:
                if isinstance(target, list):
                    target = target[int(part)]
                else:
                    target = target[part]
            if isinstance(target, list):
                target[int(parts[-1])] = str(resolved)
            else:
                target[parts[-1]] = str(resolved)
            if (
                binding.logical_path
                and len(parts) == 4
                and parts[:2] == ["data", "train"]
                and parts[3] == "path"
            ):
                source_config = config["data"]["train"][int(parts[2])]
                if not isinstance(source_config, dict):
                    raise ConfigurationError(
                        "dataset artifact binding requires an object source"
                    )
                source_config["logicalPath"] = binding.logical_path

    def _recover_sessions(self) -> None:
        for state_path in sorted(self.sessions_root.glob("*/session.json")):
            try:
                if (state_path.parent / "deleted.json").exists():
                    continue
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                task = MIWorkerTask.model_validate(payload["task"])
                self._validate_lease_backend(task)
                config = copy.deepcopy(task.training_config)
                self._apply_artifact_bindings(config, task)
                if self.lease is not None:
                    config.setdefault("baseModel", {})["device"] = "managed"
                engine = FacetGradientEngine(config)
                version = str(payload["weightVersion"])
                weights = state_path.parent / "weights" / f"{version}.safetensors"
                if not weights.is_file():
                    raise ConfigurationError(
                        f"Recovered session weight is missing: {weights}"
                    )
                engine.load_weights(weights)
                rounds = {
                    int(index): record
                    for index, record in dict(payload.get("rounds", {})).items()
                }
                for record in rounds.values():
                    if record.get("state") == "collecting":
                        record["state"] = "failed"
                        record["error"] = {
                            "code": "worker_restarted_during_round",
                            "detail": "The uncommitted round must be retried by Trainer",
                        }
                session = WorkerSession(
                    task=task,
                    root=state_path.parent,
                    engine=engine,
                    weight_version=version,
                    weight_sha256=str(payload["weightSha256"]),
                    created_at=str(payload.get("createdAt", utc_now())),
                    state="idle",
                    rounds=rounds,
                )
                self._sessions[task.session_id] = session
                self._persist(session)
            except BaseException as exc:
                self._recovery_errors.append(
                    {
                        "path": str(state_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
