from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


from .api.server import serve
from .args import build_parser
from .core.errors import ConfigurationError, MIError
from .core.modeling import load_base_model
from .core.registry import FacetRegistry
from .core.specs import BaseModelRequest
from .core.util import atomic_write_json, load_json
from .generate.runtime import MixedInferenceRuntime
from .rm_integration import require_managed_launch
from .worker.server import serve_worker
from .worker.service import MIWorkerService


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.PUs:
        print(
            "MI error [RM_LEASE_REQUIRED]: --PUs was replaced by "
            "`jester-rm resources list`.",
            file=sys.stderr,
        )
        return 2
    if not args.command:
        parser.print_help()
        return 0
    try:
        require_managed_launch(args)
        return _dispatch(args)
    except MIError as exc:
        print(f"MI error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


def _dispatch(args: Any) -> int:
    if args.command == "migrate-rm-config":
        source = Path(args.input).resolve()
        if args.in_place and args.output:
            raise ConfigurationError("migrate-rm-config accepts output or --in-place, not both")
        if not args.in_place and not args.output:
            raise ConfigurationError("migrate-rm-config requires output or --in-place")
        payload = load_json(source)
        if not isinstance(payload, dict):
            raise ConfigurationError("MI configuration root must be a JSON object")
        base = payload.get("baseModel")
        if not isinstance(base, dict):
            raise ConfigurationError("MI configuration has no baseModel object")
        previous = base.get("device")
        base["device"] = "managed"
        destination = source if args.in_place else Path(args.output).resolve()
        atomic_write_json(destination, payload)
        _print_json(
            {
                "migrated": True,
                "path": str(destination),
                "previousDevice": previous,
                "device": "managed",
            }
        )
        return 0

    if args.command == "train":
        raise MIError(
            "MI no longer owns training jobs. Run `j-train migrate-mi-config` and submit a Trainer campaign.",
            code="trainer_required",
            status=409,
        )

    if args.command == "generate":
        request = _generation_request(args)
        if args.server:
            result = request_json(args.server, "/generate", request)
        else:
            if not args.base_model:
                raise ConfigurationError("Local generate requires --base-model, or set --server")
            runtime = MixedInferenceRuntime(args.registry)
            runtime.load_base(base_request(args))
            try:
                result = runtime.generate(request)
            finally:
                runtime.unload_base()
        if args.json:
            _print_json(result)
        else:
            print(result["output"]["text"])
        return 0

    if args.command == "serve":
        request = base_request(args) if args.base_model else None
        serve(
            host=args.host,
            port=args.port,
            registry=args.registry,
            base_model=request,
            workers=args.workers,
            max_request_bytes=args.max_request_bytes,
        )
        return 0

    if args.command == "worker":
        if args.worker_command == "serve":
            serve_worker(args.host, args.port, root=args.root, require_rm=True)
            return 0
        if args.worker_command == "probe":
            _print_json(MIWorkerService(require_rm=True).capabilities())
            return 0
        raise ConfigurationError(f"Unknown worker command: {args.worker_command}")

    if args.command == "plan":
        if args.server:
            result = request_json(args.server, "/compose", {"facets": args.facet})
        else:
            plan, _ = FacetRegistry(args.registry).resolve(args.facet)
            result = plan.to_dict()
        _print_json(result)
        return 0

    if args.command == "facets":
        if args.server:
            result = request_json(args.server, "/facets", method="GET")
        else:
            registry = FacetRegistry(args.registry)
            result = {"registry": str(registry.root), "facets": registry.list()}
        _print_json(result)
        return 0

    if args.command == "inspect-base":
        base = load_base_model(base_request(args), load_tokenizer=False)
        _print_json(base.summary())
        return 0

    if args.command == "health":
        _print_json(request_json(args.server, "/health", method="GET"))
        return 0

    if args.command == "base-load":
        _print_json(request_json(args.server, "/base/load", base_request(args).to_dict()))
        return 0

    if args.command == "base-unload":
        _print_json(request_json(args.server, "/base/unload", {}))
        return 0

    if args.command == "facet-load":
        _print_json(
            request_json(
                args.server,
                "/facets/load",
                {
                    "facets": args.facet,
                    "allowModelIdMismatch": args.allow_model_id_mismatch,
                    "allowUnsafeLegacyBaseBinding": args.allow_unsafe_legacy_base_binding,
                },
            )
        )
        return 0

    if args.command == "facet-unload":
        if not args.all and not args.facet:
            raise ConfigurationError("facet-unload requires --facet or --all")
        _print_json(
            request_json(
                args.server,
                "/facets/unload",
                {"facets": args.facet or [], "all": args.all},
            )
        )
        return 0

    if args.command in {"train-submit", "jobs", "job"}:
        raise MIError(
            "MI training job management was removed; use j-train campaign/job commands.",
            code="trainer_required",
            status=409,
        )

    if args.command == "registry-refresh":
        _print_json(request_json(args.server, "/registry/refresh", {}))
        return 0
    raise ConfigurationError(f"Unknown command: {args.command}")


def base_request(args: Any) -> BaseModelRequest:
    artifact_files = []
    if args.artifact_files:
        artifact_files = load_json(args.artifact_files)
        if not isinstance(artifact_files, list):
            raise ConfigurationError("--artifact-files JSON root must be an array")
    return BaseModelRequest.from_dict(
        {
            "id": args.base_model,
            "canonicalId": args.base_identity,
            "revision": args.revision,
            "layerPath": args.layer_path,
            "dtype": args.dtype,
            "device": args.device,
            "allowDownload": args.online,
            "trustRemoteCode": args.trust_remote_code,
            "modelKwargs": _json_object(args.model_kwargs, "--model-kwargs"),
            "tokenizerKwargs": _json_object(args.tokenizer_kwargs, "--tokenizer-kwargs"),
            "artifactSha256": args.artifact_sha256 or "",
            "artifactFiles": artifact_files,
        }
    )


def _generation_request(args: Any) -> dict[str, Any]:
    if args.request:
        value = load_json(args.request)
        if not isinstance(value, dict):
            raise ConfigurationError("--request JSON root must be an object")
        request = value
    else:
        request = {"schemaVersion": "mi-generate-v1"}
        if args.messages:
            messages = _json_value(args.messages)
            if not isinstance(messages, list):
                raise ConfigurationError("--messages must resolve to a JSON array")
            request["messages"] = messages
        else:
            prompt = " ".join(args.prompt).strip()
            if not prompt:
                raise ConfigurationError("generate requires prompt text, --messages, or --request")
            request["prompt"] = prompt
        request["facets"] = args.facet
        generation: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
        if args.temperature is not None:
            generation["temperature"] = args.temperature
        if args.top_p is not None:
            generation["top_p"] = args.top_p
        if args.do_sample:
            generation["do_sample"] = True
        request["generation"] = generation
        if args.seed is not None:
            request["seed"] = args.seed
        if args.return_full_text:
            request["returnFullText"] = True
    return request


def request_json(
    server: str,
    path: str,
    payload: Any | None = None,
    *,
    method: str = "POST",
    timeout: float = 300.0,
) -> Any:
    url = server.rstrip("/") + path
    body = None if method == "GET" else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode("utf-8")).get("error", {})
            message = str(error.get("message") or exc.reason)
            code = str(error.get("code") or "http_error")
        except Exception:
            message, code = str(exc.reason), "http_error"
        raise MIError(message, code=code, status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise MIError(
            f"Cannot reach MI server {server}: {exc.reason}",
            code="server_unavailable",
            status=503,
        ) from exc


def processing_units() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return [{"index": "cpu", "device": "cpu", "name": "CPU", "available": True}]
    units: list[dict[str, Any]] = [
        {"index": "cpu", "device": "cpu", "name": "CPU", "available": True}
    ]
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        units.append(
            {
                "index": index,
                "device": f"cuda:{index}",
                "name": properties.name,
                "memoryBytes": int(properties.total_memory),
                "available": True,
            }
        )
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        units.append({"index": "mps", "device": "mps", "name": "Apple MPS", "available": True})
    return units


def _json_object(value: str | None, name: str) -> dict[str, Any]:
    if not value:
        return {}
    result = _json_value(value)
    if not isinstance(result, dict):
        raise ConfigurationError(f"{name} must be a JSON object or path")
    return result


def _json_value(value: str) -> Any:
    candidate = Path(value)
    if candidate.exists():
        return load_json(candidate)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid inline JSON or file path {value!r}: {exc.msg}") from exc


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
