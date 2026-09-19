from __future__ import annotations

from typing import Any

from .core.errors import ConfigurationError


LOCAL_RESOURCE_COMMANDS = {"serve", "inspect-base", "worker"}


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


def start_telemetry() -> Any:
    try:
        from jester_rm_client import start_default_telemetry

        return start_default_telemetry()
    except Exception:
        return None


def is_managed() -> bool:
    try:
        from jester_rm_client import current_lease

        return current_lease(required=False) is not None
    except Exception:
        return False
