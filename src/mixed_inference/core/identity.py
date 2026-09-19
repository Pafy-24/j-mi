"""Verified local artifact and tokenizer identities for MI compatibility."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import CompatibilityError, ConfigurationError
from .util import sha256_file, sha256_json


@dataclass(frozen=True)
class Identity:
    sha256: str = ""
    status: str = "unverified"
    reason: str = "identity was not supplied as a verifiable local manifest"

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def canonical_artifact_manifest(files: Iterable[dict[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ConfigurationError("baseModel.artifactFiles entries must be objects")
        path = str(item.get("path", "")).replace("\\", "/")
        digest = str(item.get("sha256", "")).lower()
        size = item.get("sizeBytes")
        _relative_path(path)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ConfigurationError("baseModel.artifactFiles.sha256 must be a SHA-256 hex digest")
        if not isinstance(size, int) or size < 0:
            raise ConfigurationError("baseModel.artifactFiles.sizeBytes must be a non-negative integer")
        normalized.append({"path": path, "sha256": digest, "sizeBytes": size})
    if not normalized:
        raise ConfigurationError("baseModel.artifactFiles must contain at least one file")
    paths = [item["path"] for item in normalized]
    if len(set(paths)) != len(paths):
        raise ConfigurationError("baseModel.artifactFiles contains duplicate paths")
    return sha256_json({"schemaVersion": "mi-artifact-files-v1", "files": sorted(normalized, key=lambda item: item["path"])})


def verify_local_artifact(
    root: Path,
    files: Iterable[dict[str, Any]],
    expected_sha256: str,
    *,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> Identity:
    if not root.is_dir():
        raise ConfigurationError("verified local artifact identity requires a model directory")
    entries = list(files)
    root = root.resolve()
    declared = {str(item["path"]).replace("\\", "/") for item in entries}
    required = _loader_contract_files(
        root,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs or {},
        tokenizer_kwargs=tokenizer_kwargs or {},
    )
    missing = sorted(required - declared)
    if missing:
        raise CompatibilityError(
            "base artifact manifest omits loader input(s): " + ", ".join(missing[:5])
        )
    extras = sorted(declared - required)
    if extras:
        raise ConfigurationError(
            "base artifact manifest includes non-loader input(s): " + ", ".join(extras[:5])
        )
    digest = canonical_artifact_manifest(entries)
    if expected_sha256 and digest != expected_sha256.lower():
        raise CompatibilityError("base artifact manifest digest differs from baseModel.artifactSha256")
    for item in entries:
        relative = _relative_path(str(item["path"]).replace("\\", "/"))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ConfigurationError("baseModel.artifactFiles path escapes the model directory") from exc
        if not candidate.is_file():
            raise CompatibilityError(f"base artifact file is missing: {relative.as_posix()}")
        if candidate.stat().st_size != int(item["sizeBytes"]):
            raise CompatibilityError(f"base artifact size differs: {relative.as_posix()}")
        if sha256_file(candidate) != str(item["sha256"]).lower():
            raise CompatibilityError(f"base artifact hash differs: {relative.as_posix()}")
    return Identity(sha256=digest, status="verified", reason="local artifact file manifest verified")


def _loader_contract_files(
    root: Path,
    *,
    trust_remote_code: bool,
    model_kwargs: dict[str, Any],
    tokenizer_kwargs: dict[str, Any],
) -> set[str]:
    """Known local AutoModel/AutoTokenizer inputs; never bless an arbitrary subset."""
    if trust_remote_code:
        raise ConfigurationError(
            "verified local artifact identity does not support trustRemoteCode without an RM-approved code attestation"
        )
    unsupported_model_options = {"use_safetensors", "variant", "subfolder", "from_tf", "from_flax", "state_dict"}
    unsupported_tokenizer_options = {"subfolder", "vocab_file", "merges_file", "tokenizer_file"}
    if unsupported_model_options.intersection(model_kwargs) or unsupported_tokenizer_options.intersection(tokenizer_kwargs):
        raise ConfigurationError(
            "verified local artifact identity does not support loader-selection overrides"
        )
    config = root / "config.json"
    if not config.is_file():
        raise ConfigurationError("verified local artifact identity requires config.json")
    required: set[str] = {"config.json"}
    index = root / "model.safetensors.index.json"
    if index.is_file():
        required.add(index.name)
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            weights = payload.get("weight_map", {})
            if not isinstance(weights, dict) or not all(isinstance(value, str) for value in weights.values()):
                raise ValueError("weight_map must map tensor names to shard names")
            required.update(str(value).replace("\\", "/") for value in weights.values())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ConfigurationError(f"invalid safetensors index {index}: {exc}") from exc
    elif (root / "model.safetensors").is_file():
        required.add("model.safetensors")
    else:
        raise ConfigurationError(
            "verified local artifact identity supports only model.safetensors or model.safetensors.index.json"
        )
    if (root / "generation_config.json").is_file():
        required.add("generation_config.json")
    tokenizer_json = root / "tokenizer.json"
    legacy_tokenizer_inputs = ("vocab.json", "merges.txt", "vocab.txt", "spiece.model")
    if tokenizer_json.is_file():
        required.add("tokenizer.json")
        for name in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
            if (root / name).is_file():
                required.add(name)
    elif any((root / name).is_file() for name in legacy_tokenizer_inputs):
        raise ConfigurationError(
            "verified local artifact identity does not support legacy tokenizer file selection"
        )
    return required


def tokenizer_identity(tokenizer: Any) -> Identity:
    """Fingerprint the serializable tokenizer pipeline, not its vocabulary alone."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialize = getattr(backend, "to_str", None)
    if not callable(serialize):
        return Identity(reason="tokenizer has no serializable backend pipeline")
    try:
        pipeline = serialize()
        if not isinstance(pipeline, str) or not pipeline:
            raise ValueError("empty tokenizer pipeline")
        payload = {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "pipeline": pipeline,
            "specialTokens": _json_value(getattr(tokenizer, "special_tokens_map", {})),
            "addedVocabulary": _json_value(getattr(tokenizer, "get_added_vocab", lambda: {})()),
            "initKwargs": _json_value(_safe_init_kwargs(tokenizer)),
        }
    except (TypeError, ValueError, AttributeError) as exc:
        return Identity(reason=f"tokenizer serialization is unsupported: {exc}")
    return Identity(sha256=sha256_json(payload), status="verified", reason="serialized tokenizer pipeline verified")


def _relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ConfigurationError("baseModel.artifactFiles.path must be a confined relative path")
    return path


def _safe_init_kwargs(tokenizer: Any) -> dict[str, Any]:
    result = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    result.pop("name_or_path", None)
    result.pop("_name_or_path", None)
    return result


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        raise ValueError("unordered tokenizer state is not canonical")
    state = getattr(value, "__getstate__", None)
    if callable(state):
        try:
            return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "state": _json_value(state())}
        except (TypeError, ValueError):
            pass
    raise ValueError(
        f"unsupported noncanonical tokenizer value {type(value).__module__}.{type(value).__qualname__}"
    )
