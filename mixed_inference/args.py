from __future__ import annotations

import argparse


DEFAULT_REGISTRY = "artifacts/facets"
DEFAULT_SERVER = "http://127.0.0.1:8120"


def add_base_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--base-model", required=required, help="Hugging Face model ID or local path.")
    parser.add_argument("--revision", help="Pinned model revision/commit.")
    parser.add_argument("--layer-path", help="Explicit decoder ModuleList path, e.g. model.layers.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--device", default="auto", help="Torch device, e.g. cpu, cuda, cuda:1.")
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face downloads.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-kwargs", help="JSON object with extra from_pretrained model kwargs.")
    parser.add_argument("--tokenizer-kwargs", help="JSON object with tokenizer kwargs.")


def add_registry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help=f"Facet registry. Default: {DEFAULT_REGISTRY}")


def add_server(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"MI API URL. Default: {DEFAULT_SERVER}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jester Mixed Inference (MI), component 2/4: train and compose expansion facets."
    )
    parser.add_argument("--PUs", action="store_true", help="List processing units visible to PyTorch.")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train one facet locally from mi-training-v1 JSON.")
    train.add_argument("config", help="Training configuration JSON path.")
    train.add_argument("--quiet", action="store_true", help="Suppress progress events.")
    train.add_argument("--json", action="store_true", help="Print the complete result JSON.")

    generate = subparsers.add_parser("generate", help="Generate locally or through a running MI server.")
    generate.add_argument("prompt", nargs="*", help="Prompt text.")
    generate.add_argument("--request", help="Complete mi-generate-v1-compatible request JSON.")
    generate.add_argument("--messages", help="Inline JSON or path to a chat messages JSON array.")
    generate.add_argument("--facet", "-f", action="append", default=[], help="Facet ID[@version], repeatable.")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--top-p", type=float)
    generate.add_argument("--do-sample", action="store_true")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--return-full-text", action="store_true")
    generate.add_argument("--json", action="store_true")
    generate.add_argument("--server", help="Use this MI API instead of loading a local model.")
    add_registry(generate)
    add_base_options(generate, required=False)

    serve = subparsers.add_parser("serve", help="Start the persistent MI HTTP API.")
    add_registry(serve)
    add_base_options(serve, required=False)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8120)
    serve.add_argument("--workers", type=int, default=4)
    serve.add_argument("--training-workers", type=int, default=1)
    serve.add_argument("--jobs-directory", default="artifacts/training-jobs")
    serve.add_argument("--max-request-bytes", type=int, default=2 * 1024 * 1024)
    serve.add_argument(
        "--allow-custom-training-classes",
        action="store_true",
        help="Allow configs to import custom trainer/optimizer/scheduler Python classes.",
    )

    plan = subparsers.add_parser("plan", help="Resolve a facet DAG without generating.")
    add_registry(plan)
    plan.add_argument("--facet", "-f", action="append", required=True)
    plan.add_argument("--server", help="Resolve through a running MI API.")

    facets = subparsers.add_parser("facets", help="List registered facet artifacts.")
    add_registry(facets)
    facets.add_argument("--server", help="Read from a running MI API.")

    inspect = subparsers.add_parser("inspect-base", help="Load and inspect a base model locally.")
    add_base_options(inspect, required=True)

    health = subparsers.add_parser("health", help="Inspect a running MI service.")
    add_server(health)

    load_base = subparsers.add_parser("base-load", help="Load/replace the server's base model.")
    add_server(load_base)
    add_base_options(load_base, required=True)

    unload_base = subparsers.add_parser("base-unload", help="Unload the server's base model.")
    add_server(unload_base)

    load_facet = subparsers.add_parser("facet-load", help="Preload a resolved facet tree on the server.")
    add_server(load_facet)
    load_facet.add_argument("--facet", "-f", action="append", required=True)
    load_facet.add_argument("--allow-model-id-mismatch", action="store_true")

    unload_facet = subparsers.add_parser("facet-unload", help="Unload server-side facet weights.")
    add_server(unload_facet)
    unload_facet.add_argument("--facet", "-f", action="append")
    unload_facet.add_argument("--all", action="store_true")

    submit = subparsers.add_parser("train-submit", help="Submit an asynchronous server training job.")
    add_server(submit)
    submit.add_argument("config", help="Training JSON path visible to the server.")
    submit.add_argument(
        "--inline",
        action="store_true",
        help="Send config contents; all paths inside it should be absolute for a remote server.",
    )

    jobs = subparsers.add_parser("jobs", help="List server-side training jobs.")
    add_server(jobs)

    job = subparsers.add_parser("job", help="Read or cancel one server-side training job.")
    add_server(job)
    job.add_argument("job_id")
    job.add_argument("--cancel", action="store_true")

    refresh = subparsers.add_parser("registry-refresh", help="Rescan the server facet registry.")
    add_server(refresh)
    return parser
