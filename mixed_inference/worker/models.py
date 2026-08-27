from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


WORKER_TASK_VERSION = "mi-worker-task-v1"


class WorkerModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    def api_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json", exclude_none=exclude_none)


class TransportSpec(WorkerModel):
    compression: Literal["none", "zstd"] = "none"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"


class MIWorkerTask(WorkerModel):
    schema_version: Literal[WORKER_TASK_VERSION] = Field(
        default=WORKER_TASK_VERSION, alias="schemaVersion"
    )
    session_id: str = Field(alias="sessionId")
    training_config: dict[str, Any] = Field(alias="trainingConfig")
    weight_version: str = Field(default="0", alias="weightVersion")
    requested_backend: str = Field(default="cpu", alias="requestedBackend")
    effective_backend: str = Field(default="cpu", alias="effectiveBackend")
    fallback_reason: str | None = Field(default=None, alias="fallbackReason")
    allow_cpu_fallback: bool = Field(default=False, alias="allowCpuFallback")
    transport: TransportSpec = Field(default_factory=TransportSpec)

    @field_validator("session_id")
    @classmethod
    def safe_session_id(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized):
            raise ValueError("sessionId must be a safe stable identifier")
        return normalized

    @field_validator("weight_version")
    @classmethod
    def safe_weight_version(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", normalized):
            raise ValueError("weightVersion contains unsupported characters")
        return normalized

    @model_validator(mode="after")
    def validate_backend(self) -> "MIWorkerTask":
        requested = self.requested_backend.lower()
        effective = self.effective_backend.lower()
        if requested != effective:
            if effective != "cpu" or not self.allow_cpu_fallback or not self.fallback_reason:
                raise ValueError(
                    "backend fallback requires effectiveBackend=cpu, allowCpuFallback=true and fallbackReason"
                )
        return self


class RoundRequest(WorkerModel):
    weight_version: str = Field(alias="weightVersion")
    indices: list[int] | None = None
    offset: int = Field(default=0, ge=0)
    count: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, alias="batchSize", gt=0)
    seed: int = 42

    @field_validator("indices")
    @classmethod
    def valid_indices(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not value or any(index < 0 for index in value)):
            raise ValueError("indices must contain non-negative dataset indices")
        return value


class WeightUpload(WorkerModel):
    version: str
    sha256: str
    offset: int = Field(default=0, ge=0)
    total_bytes: int = Field(alias="totalBytes", ge=0)
    final: bool = False

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        normalized = value.lower().removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        return normalized
