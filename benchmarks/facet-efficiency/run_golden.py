from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import random
import re
import statistics
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "jester-facet-efficiency-v1"
HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
DESIGN_FIELDS = (
    "contract",
    "requirements",
    "components",
    "data_model",
    "invariants",
    "control_flow",
    "pseudocode",
    "errors",
    "complexity",
    "test_strategy",
    "assumptions",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    token: str | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def extract_root_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL
    )
    candidate = fenced.group(1).strip() if fenced else stripped
    start = candidate.find("{")
    if start < 0:
        return None
    try:
        value, end = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError:
        return None
    if candidate[start + end :].strip() or not isinstance(value, dict):
        return None
    return value


def score_design(record: dict[str, Any], output: str) -> dict[str, Any]:
    parsed = extract_root_json(output)
    field_coverage = sum(
        1 for field in DESIGN_FIELDS if isinstance(parsed, dict) and field in parsed
    ) / len(DESIGN_FIELDS)
    canonical = (
        json.dumps(parsed, ensure_ascii=False, sort_keys=True).lower()
        if parsed is not None
        else output.lower()
    )
    groups = list(record["checks"].get("semanticGroups", []))
    group_results = {
        str(group["id"]): any(str(term).lower() in canonical for term in group["anyOf"])
        for group in groups
    }
    semantic_coverage = (
        sum(group_results.values()) / len(group_results) if group_results else 1.0
    )
    invariants = parsed.get("invariants") if isinstance(parsed, dict) else None
    flow = parsed.get("control_flow") if isinstance(parsed, dict) else None
    tests = parsed.get("test_strategy") if isinstance(parsed, dict) else None
    invariants_ok = isinstance(invariants, list) and len(invariants) >= int(
        record["checks"].get("minInvariants", 0)
    )
    flow_ok = isinstance(flow, list) and len(flow) >= int(
        record["checks"].get("minControlFlowSteps", 0)
    )
    tests_ok = isinstance(tests, list) and len(tests) >= int(
        record["checks"].get("minTests", 0)
    )
    contract_ok = isinstance(parsed, dict) and parsed.get("contract") == "code-design-v3"
    strict = bool(
        parsed is not None
        and contract_ok
        and field_coverage == 1.0
        and semantic_coverage == 1.0
        and invariants_ok
        and flow_ok
        and tests_ok
    )
    return {
        "validRootJson": parsed is not None,
        "contractExact": contract_ok,
        "fieldCoverage": round(field_coverage, 6),
        "semanticCoverage": round(semantic_coverage, 6),
        "semanticGroups": group_results,
        "invariantsMinimum": invariants_ok,
        "controlFlowMinimum": flow_ok,
        "testsMinimum": tests_ok,
        "strictPass": strict,
    }


def _code_candidate(text: str, entry_point: str) -> str:
    parsed = extract_root_json(text)
    if parsed is not None and isinstance(parsed.get("files"), dict):
        files = [str(value) for value in parsed["files"].values()]
        matching = [value for value in files if f"def {entry_point}" in value]
        if matching:
            return matching[0]
    fences = re.findall(
        r"```(?:python)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fences:
        matching = [value for value in fences if f"def {entry_point}" in value]
        return (matching or fences)[0]
    return text.strip()


def normalize_humaneval_completion(text: str, entry_point: str) -> str:
    candidate = _code_candidate(text, entry_point).replace("\r\n", "\n").strip()
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        tree = None
    if tree is not None:
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == entry_point
            ),
            None,
        )
        if function is not None:
            lines = candidate.splitlines()
            imports = [
                node
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            body = list(function.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            selected: list[str] = []
            for node in [*imports, *body]:
                segment = lines[node.lineno - 1 : (node.end_lineno or node.lineno)]
                selected.extend(textwrap.dedent("\n".join(segment)).splitlines())
            candidate = "\n".join(selected).strip("\n")
    candidate = re.split(r"\nif\s+__name__\s*==", candidate, maxsplit=1)[0].rstrip()
    candidate = textwrap.dedent(candidate).strip("\n")
    lines = candidate.splitlines() or ["pass"]
    lines = [("    " + line) if line.strip() else line for line in lines]
    return "\n".join(lines).rstrip() + "\n"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return round(statistics.fmean(items), 6) if items else None


def aggregate_design(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if item.get("scores")]
    strict = sum(bool(item["scores"]["strictPass"]) for item in completed)
    generation_ms = sum(float(item.get("timingMs", {}).get("generation", 0)) for item in completed)
    tokens = sum(int(item.get("usage", {}).get("outputTokens", 0)) for item in completed)
    return {
        "attempted": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "strictPassRate": round(strict / len(completed), 6) if completed else None,
        "strictPassWilson95": wilson(strict, len(completed)),
        "validRootJsonRate": mean(
            1.0 if item["scores"]["validRootJson"] else 0.0 for item in completed
        ),
        "fieldCoverage": mean(item["scores"]["fieldCoverage"] for item in completed),
        "semanticCoverage": mean(
            item["scores"]["semanticCoverage"] for item in completed
        ),
        "generatedTokens": tokens,
        "generationSeconds": round(generation_ms / 1000, 6),
        "tokensPerSecond": round(tokens / (generation_ms / 1000), 6)
        if generation_ms
        else None,
    }


def _token(path: str | None, root: Path) -> str | None:
    if not path:
        return None
    resolved = (root / path).resolve() if not Path(path).is_absolute() else Path(path)
    return resolved.read_text(encoding="utf-8").strip()


def control_plane_snapshot(config: dict[str, Any], root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    rm = config.get("rm") or {}
    if rm.get("url") and rm.get("topologyId"):
        try:
            result["rmTopologyMetrics"] = request_json(
                "GET",
                f"{str(rm['url']).rstrip('/')}/v1/topologies/{rm['topologyId']}/metrics",
                token=_token(rm.get("tokenFile"), root),
                timeout=60,
            )
        except Exception as exc:
            result["rmError"] = f"{type(exc).__name__}: {exc}"
    trainer = config.get("trainer") or {}
    if trainer.get("url") and trainer.get("campaignId"):
        try:
            result["trainerCampaignMetrics"] = request_json(
                "GET",
                f"{str(trainer['url']).rstrip('/')}/v1/campaigns/{trainer['campaignId']}/metrics",
                token=_token(trainer.get("tokenFile"), root),
                timeout=60,
            )
        except Exception as exc:
            result["trainerError"] = f"{type(exc).__name__}: {exc}"
    return result


def run(config_path: Path, output: Path, design_limit: int, humaneval_limit: int) -> None:
    config_path = config_path.resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design_path = (root / config["designSuite"]).resolve()
    humaneval_path = (root / config["humanEvalSuite"]).resolve()
    if sha256_file(humaneval_path) != HUMANEVAL_SHA256:
        raise RuntimeError("HumanEval digest mismatch; refusing a mutable golden suite")
    designs = load_jsonl(design_path)
    problems = load_jsonl(humaneval_path)
    if design_limit:
        designs = designs[:design_limit]
    if humaneval_limit:
        problems = problems[:humaneval_limit]
    states = list(config["states"])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_record: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "running",
        "startedAtUnix": started,
        "configSha256": sha256_file(config_path),
        "suites": {
            "coderDesign": {"sha256": sha256_file(design_path), "records": len(designs)},
            "humanEval": {"sha256": sha256_file(humaneval_path), "records": len(problems)},
        },
        "controlPlaneBefore": control_plane_snapshot(config, root),
        "states": [],
    }
    atomic_json(output / "run.json", run_record)
    mi_url = str(config["miUrl"]).rstrip("/")
    request_json("GET", f"{mi_url}/health", timeout=60)
    for state in states:
        state_id = str(state["id"])
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", state_id)
        design_output = output / f"design.{safe_id}.jsonl"
        samples_output = output / f"humaneval-samples.{safe_id}.jsonl"
        if design_output.exists() or samples_output.exists():
            raise RuntimeError(f"Output for state {state_id} already exists; use a new run directory")
        design_records: list[dict[str, Any]] = []
        for item in designs:
            payload = {
                "requestId": f"golden:{state_id}:design:{item['id']}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a software architect. Obey the requested JSON contract exactly and do not use Markdown.",
                    },
                    {"role": "user", "content": item["prompt"]},
                ],
                "facets": list(state.get("facets", [])),
                "generation": config.get("designGeneration", {}),
                "seed": int(config.get("seed", 42)),
            }
            try:
                response = request_json("POST", f"{mi_url}/generate", payload)
                text = str(response["output"]["text"])
                record = {
                    "id": item["id"],
                    "language": item.get("language"),
                    "output": text,
                    "scores": score_design(item, text),
                    "usage": response.get("usage", {}),
                    "timingMs": response.get("timingMs", {}),
                    "resolvedFacets": response.get("resolvedFacets", []),
                }
            except Exception as exc:
                record = {"id": item["id"], "error": f"{type(exc).__name__}: {exc}"}
            append_jsonl(design_output, record)
            design_records.append(record)
        syntax_passes = 0
        human_errors = 0
        for item in problems:
            payload = {
                "requestId": f"golden:{state_id}:{item['task_id']}",
                "messages": [
                    {
                        "role": "system",
                        "content": "Complete the Python function. Return only executable Python, without Markdown or explanation.",
                    },
                    {"role": "user", "content": str(item["prompt"])},
                ],
                "facets": list(state.get("facets", [])),
                "generation": config.get("humanEvalGeneration", {}),
                "seed": int(config.get("seed", 42)),
            }
            try:
                response = request_json("POST", f"{mi_url}/generate", payload)
                completion = normalize_humaneval_completion(
                    str(response["output"]["text"]), str(item["entry_point"])
                )
                try:
                    compile(str(item["prompt"]) + completion, str(item["task_id"]), "exec")
                    syntax_passes += 1
                except SyntaxError:
                    pass
                sample = {"task_id": item["task_id"], "completion": completion}
            except Exception as exc:
                human_errors += 1
                sample = {
                    "task_id": item["task_id"],
                    "completion": "    raise RuntimeError('generation failed')\n",
                    "jesterError": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(samples_output, sample)
        run_record["states"].append(
            {
                "id": state_id,
                "facets": state.get("facets", []),
                "design": aggregate_design(design_records),
                "humanEval": {
                    "attempted": len(problems),
                    "generationErrors": human_errors,
                    "syntaxPassRate": round(syntax_passes / len(problems), 6)
                    if problems
                    else None,
                    "functionalPassAt1": "pending-container-evaluation",
                    "samples": samples_output.name,
                },
            }
        )
        atomic_json(output / "run.json", run_record)
    run_record["status"] = "generated"
    run_record["finishedAtUnix"] = time.time()
    run_record["wallSeconds"] = round(run_record["finishedAtUnix"] - started, 6)
    run_record["controlPlaneAfter"] = control_plane_snapshot(config, root)
    atomic_json(output / "run.json", run_record)


def bootstrap_delta(
    baseline: dict[str, bool], candidate: dict[str, bool], *, samples: int = 10000
) -> dict[str, Any]:
    common = sorted(set(baseline).intersection(candidate))
    if not common:
        return {"pairedRecords": 0, "delta": None, "bootstrap95": None}
    values = [int(candidate[key]) - int(baseline[key]) for key in common]
    rng = random.Random(42)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return {
        "pairedRecords": len(common),
        "delta": round(statistics.fmean(values), 6),
        "bootstrap95": [
            round(estimates[int(samples * 0.025)], 6),
            round(estimates[int(samples * 0.975) - 1], 6),
        ],
        "improved": sum(value > 0 for value in values),
        "regressed": sum(value < 0 for value in values),
        "unchanged": sum(value == 0 for value in values),
    }


def summarize(run_directory: Path) -> None:
    run_directory = run_directory.resolve()
    record_path = run_directory / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    passes: dict[str, dict[str, bool]] = {}
    for state in record["states"]:
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(state["id"]))
        candidates = [
            run_directory / f"humaneval-samples.{safe_id}.jsonl_results.jsonl",
            run_directory / f"humaneval-samples.{safe_id}.jsonl_results.jsonl.gz",
        ]
        result_path = next((path for path in candidates if path.is_file()), None)
        if result_path is None:
            raise RuntimeError(f"Missing HumanEval evaluator results for {state['id']}")
        rows = load_jsonl(result_path)
        by_task = {str(item["task_id"]): bool(item["passed"]) for item in rows}
        passes[str(state["id"])] = by_task
        succeeded = sum(by_task.values())
        state["humanEval"].update(
            {
                "functionalPassAt1": round(succeeded / len(by_task), 6),
                "functionalPassWilson95": wilson(succeeded, len(by_task)),
                "evaluated": len(by_task),
                "passed": succeeded,
                "resultsSha256": sha256_file(result_path),
            }
        )
    baseline_id = str(record["states"][0]["id"])
    record["pairedHumanEval"] = {
        state_id: bootstrap_delta(passes[baseline_id], values)
        for state_id, values in passes.items()
        if state_id != baseline_id
    }
    record["status"] = "evaluated"
    atomic_json(record_path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jester facet-efficiency golden harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--design-limit", type=int, default=0)
    execute.add_argument("--humaneval-limit", type=int, default=0)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.config, args.output, args.design_limit, args.humaneval_limit)
    else:
        summarize(args.run)


if __name__ == "__main__":
    main()
