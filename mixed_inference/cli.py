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
from .training.trainer import train_from_config
from .rm_integration import require_managed_launch, start_telemetry


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
    telemetry = None
    try:
        require_managed_launch(args)
        telemetry = start_telemetry()
        return _dispatch(args)
    except MIError as exc:
        print(f"MI error [{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        if telemetry is not None:
            telemetry.close()


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
        def progress(event: dict[str, Any]) -> None:
            if not args.quiet:
                print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)

        result = train_from_config(args.config, progress_callback=progress)
        if args.json:
            _print_json(result)
        else:
            print(result["outputDirectory"])
        return 0

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
            jobs_directory=args.jobs_directory,
            workers=args.workers,
            training_workers=args.training_workers,
            max_request_bytes=args.max_request_bytes,
            allow_custom_training_classes=args.allow_custom_training_classes,
        )
        return 0

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

    if args.command == "train-submit":
        path = Path(args.config).resolve()
        payload = {"config": load_json(path)} if args.inline else {"configPath": str(path)}
        _print_json(request_json(args.server, "/training/jobs", payload))
        return 0

    if args.command == "jobs":
        _print_json(request_json(args.server, "/training/jobs", method="GET"))
        return 0

    if args.command == "job":
        suffix = f"/training/jobs/{args.job_id}"
        if args.cancel:
            suffix += "/cancel"
            result = request_json(args.server, suffix, {})
        else:
            result = request_json(args.server, suffix, method="GET")
        _print_json(result)
        return 0

    if args.command == "registry-refresh":
        _print_json(request_json(args.server, "/registry/refresh", {}))
        return 0
    raise ConfigurationError(f"Unknown command: {args.command}")


def base_request(args: Any) -> BaseModelRequest:
    return BaseModelRequest.from_dict(
        {
            "id": args.base_model,
            "revision": args.revision,
            "layerPath": args.layer_path,
            "dtype": args.dtype,
            "device": args.device,
            "allowDownload": args.online,
            "trustRemoteCode": args.trust_remote_code,
            "modelKwargs": _json_object(args.model_kwargs, "--model-kwargs"),
            "tokenizerKwargs": _json_object(args.tokenizer_kwargs, "--tokenizer-kwargs"),
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
