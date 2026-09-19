"""Safe, allocation-free safetensors header accounting for bounded MI admission."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}


def safetensors_tensor_bytes(path: Path, *, target_element_bytes: int) -> tuple[int, int]:
    """Return serialized tensor bytes and target tensor bytes without loading tensors."""
    if target_element_bytes <= 0:
        raise ConfigurationError("target tensor element size must be positive")
    try:
        with path.open("rb") as stream:
            length_raw = stream.read(8)
            if len(length_raw) != 8:
                raise ValueError("missing header length")
            header_length = struct.unpack("<Q", length_raw)[0]
            if header_length > 16 * 1024 * 1024:
                raise ValueError("header exceeds 16 MiB")
            raw = stream.read(header_length)
        if len(raw) != header_length:
            raise ValueError("truncated header")
        header = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, struct.error) as exc:
        raise ConfigurationError(f"invalid safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ConfigurationError(f"invalid safetensors header {path}: root must be an object")
    source_total = 0
    target_total = 0
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ConfigurationError(f"invalid safetensors tensor metadata {name!r}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ConfigurationError(f"invalid safetensors tensor metadata {name!r}")
        count = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise ConfigurationError(f"invalid safetensors tensor shape {name!r}")
            count *= dimension
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ConfigurationError(f"invalid safetensors tensor offsets {name!r}")
        source_bytes = count * _DTYPE_BYTES[dtype]
        if end - start != source_bytes:
            raise ConfigurationError(f"safetensors offsets do not match tensor shape {name!r}")
        source_total += source_bytes
        target_total += count * target_element_bytes
    if source_total == 0:
        raise ConfigurationError(f"safetensors header {path} declares no tensors")
    return source_total, target_total
