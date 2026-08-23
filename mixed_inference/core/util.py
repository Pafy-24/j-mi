from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"JSON file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def resolve_relative(path: str | Path, base: str | Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or base is None:
        return candidate
    return Path(base) / candidate


def import_symbol(path: str) -> Any:
    module_name, separator, symbol_name = path.rpartition(".")
    if not separator:
        raise ConfigurationError(
            f"Custom class path must include a module and symbol name: {path!r}"
        )
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(f"Cannot import custom symbol {path!r}: {exc}") from exc


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a JSON object")
    return value


def optional_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(require_mapping(value, name))


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{name} must be a JSON array")
    return value


def string_list(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = require_list(value, name)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ConfigurationError(f"{name} must contain only non-empty strings")
    return tuple(item.strip() for item in items)
