from __future__ import annotations

import os
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from jester_sdk_runtime.lifecycle import start_telemetry

from .core.errors import ConfigurationError


LOCAL_RESOURCE_COMMANDS = {"serve", "inspect-base", "worker"}
_STAGING_LOCK = RLock()


@dataclass(frozen=True)
class FacetStagingAllocation:
    staging_root: Path
    usage_root: Path
    limit_bytes: int
    managed: bool
    quota_enforcement: str

    def create_request(self, required_bytes: int) -> "FacetStagingRequest":
        if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes < 0:
            raise ConfigurationError("facet staging bytes must be a non-negative integer")
        _STAGING_LOCK.acquire()
        try:
            used = _directory_usage(self.usage_root)
            if used + required_bytes > self.limit_bytes:
                raise ConfigurationError(
                    "facet_staging_denied: instance storage usage plus facet snapshots "
                    f"exceeds the lease allocation ({used}+{required_bytes}>{self.limit_bytes})"
                )
            self.staging_root.mkdir(parents=True, exist_ok=True)
            _require_within(self.staging_root.resolve(), self.usage_root.resolve())
            request_directory = self.staging_root / uuid.uuid4().hex
            request_directory.mkdir(exist_ok=False)
            _require_within(request_directory.resolve(), self.staging_root.resolve())
            return FacetStagingRequest(request_directory, required_bytes)
        except BaseException:
            _STAGING_LOCK.release()
            raise


@dataclass
class FacetStagingRequest:
    directory: Path
    reserved_bytes: int
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # Snapshot owners remove their exact files first. An unexpected
            # sibling is never recursively deleted by MI.
            pass
        finally:
            self._closed = True
            _STAGING_LOCK.release()


def facet_staging_allocation(
    *,
    local_root: str | Path | None = None,
    local_limit_bytes: int | None = None,
) -> FacetStagingAllocation:
    """Resolve bounded facet staging from an RM lease or explicit local input."""
    try:
        from jester_rm_client import current_lease

        lease = current_lease(required=None)
    except ImportError:
        lease = None
    except Exception as exc:
        raise ConfigurationError(f"invalid_rm_lease: {exc}") from exc

    if lease is not None:
        _require_active_lease(lease)
        if local_root is not None or local_limit_bytes is not None:
            raise ConfigurationError(
                "managed facet staging cannot be overridden by local paths"
            )
        paths = lease.lease.get("paths")
        if not isinstance(paths, dict) or not paths.get("root") or not paths.get("run"):
            raise ConfigurationError(
                "facet_staging_denied: RM lease requires explicit paths.root and paths.run"
            )
        instance_path = Path(str(paths["root"])).expanduser()
        run_path = Path(str(paths["run"])).expanduser()
        if not instance_path.is_absolute() or not run_path.is_absolute():
            raise ConfigurationError(
                "facet_staging_denied: RM lease paths.root and paths.run must be absolute"
            )
        instance_root = instance_path.resolve()
        run_root = run_path.resolve()
        _require_within(run_root, instance_root)
        if not instance_root.is_dir() or not run_root.is_dir():
            raise ConfigurationError(
                "facet_staging_denied: RM instance root and run directory must exist"
            )
        storage = [
            binding
            for binding in lease.bindings
            if "quotaBytes" in binding.get("runtime", {})
        ]
        if len(storage) != 1:
            raise ConfigurationError(
                "facet_staging_denied: RM lease requires exactly one storage quota binding"
            )
        binding = storage[0]
        quota = _positive_int(
            binding["runtime"]["quotaBytes"],
            "RM storage runtime.quotaBytes",
        )
        enforcement = str(binding.get("enforcement", {}).get("quota") or "unknown")
        return FacetStagingAllocation(
            staging_root=run_root / "mi-runtime" / "facet-staging",
            usage_root=instance_root,
            limit_bytes=quota,
            managed=True,
            quota_enforcement=enforcement,
        )

    if local_root is None or local_limit_bytes is None:
        raise ConfigurationError(
            "facet_staging_denied: unmanaged MI requires an explicit staging directory and byte limit"
        )
    if isinstance(local_limit_bytes, bool) or not isinstance(local_limit_bytes, int) or local_limit_bytes <= 0:
        raise ConfigurationError("local facet staging limit must be a positive integer")
    root = Path(local_root).expanduser().resolve()
    return FacetStagingAllocation(
        staging_root=root,
        usage_root=root,
        limit_bytes=local_limit_bytes,
        managed=False,
        quota_enforcement="local-logical",
    )


def _directory_usage(root: Path) -> int:
    if not root.exists():
        return 0
    if not root.is_dir():
        raise ConfigurationError(f"facet_staging_denied: usage root is not a directory: {root}")
    resolved_root = root.resolve()
    total = 0
    try:
        def fail_scan(error: OSError) -> None:
            raise error

        for directory, names, files in os.walk(
            resolved_root,
            followlinks=False,
            onerror=fail_scan,
        ):
            directory_path = Path(directory)
            _require_within(directory_path.resolve(), resolved_root)
            for name in names:
                child = directory_path / name
                metadata = child.lstat()
                if _is_reparse(metadata):
                    raise ConfigurationError(
                        f"facet_staging_denied: reparse directory inside instance root: {child}"
                    )
            for name in files:
                child = directory_path / name
                metadata = child.lstat()
                if _is_reparse(metadata):
                    raise ConfigurationError(
                        f"facet_staging_denied: reparse file inside instance root: {child}"
                    )
                _require_within(child.resolve(), resolved_root)
                total += int(metadata.st_size)
    except ConfigurationError:
        raise
    except OSError as exc:
        raise ConfigurationError(
            f"facet_staging_denied: cannot account instance storage: {exc}"
        ) from exc
    return total


def _is_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _require_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(
            f"facet_staging_denied: path escapes owned instance root: {path}"
        ) from exc


def command_requires_rm(args: Any) -> bool:
    command = getattr(args, "command", None)
    if command in LOCAL_RESOURCE_COMMANDS:
        return True
    return command == "generate" and not bool(getattr(args, "server", None))


def require_managed_launch(args: Any) -> Any:
    if not command_requires_rm(args):
        return None
    try:
        from jester_rm_client import require_lease

        return require_lease()
    except ImportError as exc:
        raise ConfigurationError(
            "RM_LEASE_REQUIRED: install jester-rm-client and launch MI through jester-rm."
        ) from exc
    except Exception as exc:
        code = getattr(exc, "code", "RM_LEASE_REQUIRED")
        raise ConfigurationError(f"{code}: {exc}") from exc


def managed_device(requested: str) -> str:
    try:
        from jester_rm_client import current_lease
    except ImportError as exc:
        if requested == "managed":
            raise ConfigurationError(
                "baseModel.device=managed requires the jester-rm-client package"
            ) from exc
        return requested
    try:
        lease = current_lease(required=None)
    except Exception as exc:
        raise ConfigurationError(f"invalid_rm_lease: {exc}") from exc
    if lease is None:
        if requested == "managed":
            raise ConfigurationError("baseModel.device=managed requires an active RM lease")
        return requested
    import os
    backend = os.environ.get("JESTER_RM_DEVICE_BACKEND", "cpu")
    target = "cuda:0" if backend == "torch.cuda" else "xla" if backend == "torch_xla" else "cpu"
    if backend not in {"cpu", "torch.cuda", "torch_xla"}:
        raise ConfigurationError(f"invalid_rm_lease: managed backend {backend!r} is not supported by MI")
    normalized = requested.lower()
    accepted = {"auto", "managed", target}
    if target == "cuda:0":
        accepted.update({"cuda", "cuda:0"})
    if normalized not in accepted:
        raise ConfigurationError(
            f"baseModel.device={requested!r} conflicts with RM binding {target!r}"
        )
    return target


def apply_memory_budget(torch: Any, device: Any) -> None:
    try:
        from jester_rm_client import current_lease
        lease = current_lease(required=None)
        if lease is None or not str(device).startswith("cuda"):
            return
        for index, binding in enumerate(item for item in lease.accelerators if item.get("runtime", {}).get("backend") == "torch.cuda"):
            budget = int(binding.get("runtime", {}).get("memoryBudgetBytes") or 0)
            if budget:
                total = int(torch.cuda.get_device_properties(index).total_memory)
                torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.01, budget / total)), index)
    except Exception as exc:
        raise ConfigurationError(f"Cannot apply RM accelerator memory budget: {exc}") from exc


def residency_budget() -> dict[str, int]:
    """Read explicit MI cache limits from the admitted RM lease/environment.

    This does not discover hardware capacity. Unmanaged processes may remain
    unbounded; a managed process requires the exact RM memory binding.
    RMA starts services under a reserved lease and activates it after health
    succeeds. Reading its budget during bootstrap does not authorize operations.
    """
    import os

    result: dict[str, int] = {}
    env_ram = _optional_env_int("JESTER_MI_RAM_BUDGET_BYTES", positive=True)
    env_margin = _optional_env_int(
        "JESTER_MI_OPERATIONAL_MARGIN_BYTES", positive=False
    )
    if env_margin is not None:
        result["operationalMarginBytes"] = env_margin
    if env_ram is not None:
        result["ramBytes"] = env_ram
    try:
        from jester_rm_client import current_lease

        lease = current_lease(required=None)
        if lease is not None:
            if lease.lease.get("state") not in {"reserved", "active"}:
                raise ConfigurationError(
                    "invalid_rm_lease: MI budget requires a reserved or active RM lease"
                )
            memory = [
                binding
                for binding in lease.bindings
                if "ramBytes" in binding.get("runtime", {})
            ]
            if len(memory) != 1:
                raise ConfigurationError(
                    "invalid_rm_lease: managed MI requires exactly one memory binding"
                )
            lease_ram = _positive_int(
                memory[0]["runtime"]["ramBytes"],
                "RM memory runtime.ramBytes",
            )
            if env_ram is not None and env_ram > lease_ram:
                raise ConfigurationError(
                    "JESTER_MI_RAM_BUDGET_BYTES cannot exceed the active RM memory lease"
                )
            result["ramBytes"] = env_ram if env_ram is not None else lease_ram
            accelerator = sum(
                int(item.get("runtime", {}).get("memoryBudgetBytes") or 0)
                for item in lease.accelerators
                if item.get("runtime", {}).get("backend") == "torch.cuda"
            )
            if accelerator:
                result["acceleratorBytes"] = accelerator
    except ImportError:
        pass
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid_rm_lease: {exc}") from exc
    return result


def _optional_env_int(name: str, *, positive: bool) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ConfigurationError(f"{name} must be a {qualifier} integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _require_active_lease(lease: Any) -> None:
    payload = getattr(lease, "lease", None)
    if not isinstance(payload, dict) or payload.get("state") != "active":
        raise ConfigurationError(
            "invalid_rm_lease: managed MI requires an active RM lease"
        )


def is_managed() -> bool:
    try:
        from jester_rm_client import current_lease

        return current_lease(required=False) is not None
    except Exception:
        return False
