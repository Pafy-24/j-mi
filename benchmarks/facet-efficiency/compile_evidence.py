from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "jester-facet-evidence-v1"
STATE_ORDER = (
    "base",
    "base+coder",
    "base+parent+child",
    "base+lsr+mi",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve(base: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _state_by_id(run: dict[str, Any], state_id: str) -> dict[str, Any] | None:
    for state in run.get("states", []):
        if state.get("id") == state_id:
            return state
    return None


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def _nested(source: dict[str, Any], *keys: str) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _ratio(value: Any, baseline: Any) -> float | None:
    numerator = _number(value)
    denominator = _number(baseline)
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def _round(value: Any, digits: int = 6) -> float | int | None:
    number = _number(value)
    if number is None:
        return None
    if isinstance(number, int):
        return number
    return round(number, digits)


def _load_state_source(
    manifest_dir: Path,
    run_entry: dict[str, Any],
    state_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    source_raw = run_entry.get("stateSources", {}).get(state_id, run_entry["run"])
    source_dir = _resolve(manifest_dir, source_raw)
    run_path = source_dir / "run.json"
    if not run_path.is_file():
        return source_dir, {}, None
    source_run = _read_json(run_path)
    return source_dir, source_run, _state_by_id(source_run, state_id)


def _quality_row(
    manifest_dir: Path,
    run_entry: dict[str, Any],
    state_id: str,
) -> dict[str, Any]:
    source_dir, source_run, state = _load_state_source(
        manifest_dir, run_entry, state_id
    )
    if state is None:
        return {
            "state": state_id,
            "status": "not-measured",
            "sourceRun": source_dir.name,
        }

    suite_sources: dict[str, str] = run_entry.get("suiteSources", {}).get(
        state_id, {}
    )
    source_runs = {
        "design": source_dir.name,
        "humanEval": source_dir.name,
        "generalMmlu": source_dir.name,
        "routing": source_dir.name,
        "health": source_dir.name,
    }
    source_documents: dict[str, dict[str, Any]] = {
        "design": source_run,
        "humanEval": source_run,
        "generalMmlu": source_run,
        "routing": source_run,
        "health": source_run,
    }

    def suite_state(suite: str) -> dict[str, Any]:
        override = suite_sources.get(suite)
        if not override:
            return state
        override_dir = _resolve(manifest_dir, override)
        override_run = _read_json(override_dir / "run.json")
        override_state = _state_by_id(override_run, state_id)
        if override_state is None:
            raise ValueError(
                f"State {state_id!r} is missing from suite override {override_dir}"
            )
        source_runs[suite] = override_dir.name
        source_documents[suite] = override_run
        return override_state

    design = suite_state("design").get("design") or {}
    human = suite_state("humanEval").get("humanEval") or {}
    general = suite_state("generalMmlu").get("generalMmlu") or {}
    routing = suite_state("routing").get("routing") or {}
    health = suite_state("health").get("miHealthAfter") or {}
    memory = health.get("memory") or {}
    evaluated = bool(human.get("evaluated"))
    general_complete = general.get("completed") == general.get("attempted")
    completed = (
        design.get("completed") == design.get("attempted")
        and human.get("completed") == human.get("attempted")
        and evaluated
        and general_complete
    )
    return {
        "state": state_id,
        "status": "measured" if completed else "partial",
        "sourceRun": source_dir.name,
        "sourceRuns": source_runs,
        "sourceRunStatus": source_run.get("status"),
        "facets": state.get("facets", []),
        "quality": {
            "generalMmluAttempted": general.get("attempted"),
            "generalMmluCompleted": general.get("completed"),
            "generalMmluErrors": general.get("errors"),
            "generalMmluAccuracy": _round(general.get("accuracy")),
            "generalMmluWilson95": general.get("wilson95"),
            "generalMmluCategories": general.get("categories"),
            "codingAttempted": design.get("attempted"),
            "codingCompleted": design.get("completed"),
            "codingErrors": design.get("errors"),
            "codingSemanticCoverage": _round(design.get("semanticCoverage")),
            "codingValidJsonRate": _round(design.get("validRootJsonRate")),
            "codingFieldCoverage": _round(design.get("fieldCoverage")),
            "pythonHumanEvalAttempted": human.get("attempted"),
            "pythonHumanEvalCompleted": human.get("completed"),
            "pythonHumanEvalErrors": human.get("errors"),
            "pythonHumanEvalPassAt1": _round(human.get("functionalPassAt1")),
            "pythonHumanEvalWilson95": human.get("functionalPassWilson95"),
            "pythonHumanEvalPassed": human.get("passed"),
            "pythonHumanEvalEvaluated": human.get("evaluated"),
        },
        "qualityDeltaVsBase": {
            "generalMmlu": _nested(
                source_documents["generalMmlu"],
                "pairedGeneralMmlu",
                state_id,
            ),
            "pythonHumanEval": _nested(
                source_documents["humanEval"],
                "pairedHumanEval",
                state_id,
            ),
        }
        if state_id != "base"
        else None,
        "performance": {
            "stateWallSeconds": _round(state.get("wallSeconds")),
            "designTokensPerSecond": _round(design.get("tokensPerSecond")),
            "humanEvalTokensPerSecond": _round(human.get("tokensPerSecond")),
            "designLatencyMs": design.get("latencyMs"),
            "humanEvalLatencyMs": human.get("latencyMs"),
        },
        "resources": {
            "cudaAllocatedBytes": memory.get("cudaAllocatedBytes"),
            "cudaReservedBytes": memory.get("cudaReservedBytes"),
        },
        "routing": {
            "expectedDecisionRate": _round(routing.get("expectedDecisionRate")),
            "expectedRecall": _round(routing.get("expectedRecall")),
            "expectedAbstentionRate": _round(
                routing.get("expectedAbstentionRate")
            ),
            "latencyMs": routing.get("latencyMs"),
        }
        if state.get("routeWithLsr")
        else None,
    }


def _attach_costs(rows: list[dict[str, Any]]) -> None:
    baseline = next((row for row in rows if row["state"] == "base"), None)
    if not baseline or baseline.get("status") == "not-measured":
        return
    for row in rows:
        if row.get("status") == "not-measured":
            continue
        if row["state"] != "base":
            coding = _number(
                _nested(row, "quality", "codingSemanticCoverage")
            )
            base_coding = _number(
                _nested(baseline, "quality", "codingSemanticCoverage")
            )
            if coding is not None and base_coding is not None:
                row.setdefault("qualityDeltaVsBase", {})[
                    "codingSemanticCoverage"
                ] = _round(float(coding) - float(base_coding))
        row["costVsBase"] = {
            "cudaAllocated": _ratio(
                _nested(row, "resources", "cudaAllocatedBytes"),
                _nested(baseline, "resources", "cudaAllocatedBytes"),
            ),
            "cudaReserved": _ratio(
                _nested(row, "resources", "cudaReservedBytes"),
                _nested(baseline, "resources", "cudaReservedBytes"),
            ),
            "humanEvalThroughput": _ratio(
                _nested(row, "performance", "humanEvalTokensPerSecond"),
                _nested(baseline, "performance", "humanEvalTokensPerSecond"),
            ),
            "humanEvalP50Latency": _ratio(
                _nested(
                    row,
                    "performance",
                    "humanEvalLatencyMs",
                    "generationP50",
                ),
                _nested(
                    baseline,
                    "performance",
                    "humanEvalLatencyMs",
                    "generationP50",
                ),
            ),
        }


def compile_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schemaVersion") != "jester-facet-evidence-manifest-v1":
        raise ValueError("Unsupported evidence manifest schemaVersion")
    manifest_dir = manifest_path.parent.resolve()
    compiled_runs: list[dict[str, Any]] = []
    for run_entry in manifest.get("runs", []):
        primary_dir = _resolve(manifest_dir, run_entry["run"])
        primary_path = primary_dir / "run.json"
        primary_run = _read_json(primary_path) if primary_path.is_file() else {}
        preflight = primary_run.get("preflight") or {}
        mi_health = preflight.get("miHealth") or {}
        registry = preflight.get("miRegistry") or {}
        files = preflight.get("files") or []
        weights_by_reference = {
            item.get("reference"): item
            for item in files
            if item.get("role") == "facet-weights" and item.get("reference")
        }
        facets = []
        for facet in registry.get("facets", []):
            weights = weights_by_reference.get(facet.get("reference"), {})
            facets.append(
                {
                    "reference": facet.get("reference"),
                    "requires": facet.get("requires", []),
                    "slots": facet.get("slots"),
                    "blocks": facet.get("blocks"),
                    "artifactSha256": _nested(facet, "artifact", "sha256"),
                    "parameterCount": _nested(
                        weights, "safetensors", "parameterCount"
                    ),
                    "tensorCount": _nested(weights, "safetensors", "tensorCount"),
                }
            )
        rows = [
            _quality_row(manifest_dir, run_entry, state_id)
            for state_id in STATE_ORDER
        ]
        _attach_costs(rows)
        compiled_runs.append(
            {
                "id": run_entry["id"],
                "model": run_entry["model"],
                "trainingHardware": run_entry.get("trainingHardware"),
                "trainingRounds": run_entry.get("trainingRounds"),
                "inferenceHardware": run_entry["inferenceHardware"],
                "seed": run_entry.get("seed"),
                "provenance": {
                    "runStatus": primary_run.get("status"),
                    "configSha256": primary_run.get("configSha256"),
                    "suites": primary_run.get("suites"),
                    "base": {
                        "modelId": _nested(mi_health, "base", "modelId"),
                        "modelType": _nested(mi_health, "base", "modelType"),
                        "configSha256": _nested(
                            mi_health, "base", "configSha256"
                        ),
                        "baseLayerCount": _nested(
                            mi_health, "base", "baseLayerCount"
                        ),
                        "parameterCount": _nested(
                            mi_health, "base", "parameters", "total"
                        ),
                    },
                    "facets": facets,
                },
                "states": rows,
                "notes": run_entry.get("notes", []),
            }
        )

    training: list[dict[str, Any]] = []
    training_raw = manifest.get("trainingEvidence")
    if training_raw:
        training_path = _resolve(manifest_dir, training_raw)
        training_document = _read_json(training_path)
        training = list(training_document.get("runs", []))

    return {
        "schemaVersion": SCHEMA_VERSION,
        "manifest": manifest_path.name,
        "methodology": manifest.get("methodology", {}),
        "runs": compiled_runs,
        "training": training,
        "limitations": manifest.get("limitations", []),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    number = _number(value)
    if number is None:
        return "n/m"
    return f"{float(number):.{digits}f}"


def _fmt_ratio(value: Any) -> str:
    number = _number(value)
    return "n/m" if number is None else f"{float(number):.3f}x"


def _fmt_gib(value: Any) -> str:
    number = _number(value)
    return "n/m" if number is None else f"{float(number) / (1024**3):.3f}"


def _fmt_delta(value: Any) -> str:
    if not isinstance(value, dict):
        return "n/m"
    delta = _number(value.get("delta"))
    interval = value.get("bootstrap95")
    if delta is None or not isinstance(interval, list) or len(interval) != 2:
        return "n/m"
    return (
        f"{float(delta):+.4f} "
        f"[{float(interval[0]):+.4f}, {float(interval[1]):+.4f}]"
    )


def _fmt_score_ci(score: Any, interval: Any) -> str:
    number = _number(score)
    if number is None:
        return "n/m"
    if not isinstance(interval, list) or len(interval) != 2:
        return f"{float(number):.4f}"
    return (
        f"{float(number):.4f} "
        f"[{float(interval[0]):.4f}, {float(interval[1]):.4f}]"
    )


def render_markdown(compiled: dict[str, Any]) -> str:
    lines = [
        "# Jester facet evidence",
        "",
        "> Generated from immutable benchmark artifacts. `n/m` means not measured; it is never an inferred value.",
        "",
        "## Quality and inference cost",
        "",
        "| Model | Trained on | Inference | Configuration | General MMLU [Wilson 95%] | Coding semantic | Python HumanEval pass@1 [Wilson 95%] | VRAM allocated | Throughput | p50 latency | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in compiled["runs"]:
        for row in run["states"]:
            quality = row.get("quality", {})
            costs = row.get("costVsBase", {})
            lines.append(
                "| {model} | {trained} | {inference} | `{state}` | {general} | {coding} | {python} | {vram} | {throughput} | {latency} | {status} |".format(
                    model=run["model"],
                    trained=run.get("trainingHardware") or "n/m",
                    inference=run["inferenceHardware"],
                    state=row["state"],
                    general=_fmt_score_ci(
                        quality.get("generalMmluAccuracy"),
                        quality.get("generalMmluWilson95"),
                    ),
                    coding=_fmt(quality.get("codingSemanticCoverage")),
                    python=_fmt_score_ci(
                        quality.get("pythonHumanEvalPassAt1"),
                        quality.get("pythonHumanEvalWilson95"),
                    ),
                    vram=_fmt_ratio(costs.get("cudaAllocated")),
                    throughput=_fmt_ratio(costs.get("humanEvalThroughput")),
                    latency=_fmt_ratio(costs.get("humanEvalP50Latency")),
                    status=row["status"],
                )
            )

    lines.extend(
        [
            "",
            "## Paired quality deltas versus base",
            "",
            "| Model | Inference | Configuration | Coding semantic delta | General MMLU delta [bootstrap 95%] | MMLU improved/regressed | HumanEval delta [bootstrap 95%] | HumanEval improved/regressed |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in compiled["runs"]:
        for row in run["states"]:
            if row["state"] == "base" or row.get("status") == "not-measured":
                continue
            deltas = row.get("qualityDeltaVsBase") or {}
            general_delta = deltas.get("generalMmlu")
            human_delta = deltas.get("pythonHumanEval")
            lines.append(
                f"| {run['model']} | {run['inferenceHardware']} | `{row['state']}` | "
                f"{_fmt(deltas.get('codingSemanticCoverage'))} | "
                f"{_fmt_delta(general_delta)} | "
                f"{general_delta.get('improved', 'n/m') if isinstance(general_delta, dict) else 'n/m'}/"
                f"{general_delta.get('regressed', 'n/m') if isinstance(general_delta, dict) else 'n/m'} | "
                f"{_fmt_delta(human_delta)} | "
                f"{human_delta.get('improved', 'n/m') if isinstance(human_delta, dict) else 'n/m'}/"
                f"{human_delta.get('regressed', 'n/m') if isinstance(human_delta, dict) else 'n/m'} |"
            )

    lines.extend(
        [
            "",
            "## Absolute inference performance and allocator memory",
            "",
            "| Model | Inference | Configuration | Allocated GiB | Reserved GiB | Design tok/s | HumanEval tok/s | HumanEval p50 ms | HumanEval p95 ms | State wall s |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in compiled["runs"]:
        for row in run["states"]:
            performance = row.get("performance", {})
            resources = row.get("resources", {})
            latency = performance.get("humanEvalLatencyMs") or {}
            lines.append(
                f"| {run['model']} | {run['inferenceHardware']} | `{row['state']}` | "
                f"{_fmt_gib(resources.get('cudaAllocatedBytes'))} | "
                f"{_fmt_gib(resources.get('cudaReservedBytes'))} | "
                f"{_fmt(performance.get('designTokensPerSecond'), 2)} | "
                f"{_fmt(performance.get('humanEvalTokensPerSecond'), 2)} | "
                f"{_fmt(latency.get('generationP50'), 2)} | "
                f"{_fmt(latency.get('generationP95'), 2)} | "
                f"{_fmt(performance.get('stateWallSeconds'), 2)} |"
            )

    lines.extend(
        [
            "",
            "## LSR routing",
            "",
            "| Model | Inference | Expected decision | Selection recall | Expected abstention | p50 route latency (ms) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for run in compiled["runs"]:
        routed = next(
            (row for row in run["states"] if row["state"] == "base+lsr+mi"),
            None,
        )
        routing = routed.get("routing") if routed else None
        if not routing:
            continue
        lines.append(
            f"| {run['model']} | {run['inferenceHardware']} | "
            f"{_fmt(routing.get('expectedDecisionRate'))} | "
            f"{_fmt(routing.get('expectedRecall'))} | "
            f"{_fmt(routing.get('expectedAbstentionRate'))} | "
            f"{_fmt(_nested(routing, 'latencyMs', 'p50'), 2)} |"
        )

    lines.extend(
        [
            "",
            "## Controlled training",
            "",
            "| Run | Hardware | Workers | Rounds | Supervised tokens | Last loss | Compute seconds | tok/s | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not compiled["training"]:
        lines.append("| n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | not measured |")
    for item in compiled["training"]:
        losses = item.get("losses") or []
        lines.append(
            "| {id} | {hardware} | {workers} | {rounds} | {tokens} | {loss} | {seconds} | {throughput} | {status} |".format(
                id=item.get("id", "n/m"),
                hardware=item.get("hardware", "n/m"),
                workers=item.get("workerCount", "n/m"),
                rounds=item.get("rounds", "n/m"),
                tokens=item.get("supervisedTokens", "n/m"),
                loss=_fmt(losses[-1] if losses else item.get("lastTrainingLoss")),
                seconds=_fmt(item.get("elapsedSeconds"), 2),
                throughput=_fmt(item.get("tokensPerSecond"), 2),
                status=item.get("status", "n/m"),
            )
        )

    limitations = compiled.get("limitations") or []
    if limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in limitations)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Jester benchmark evidence")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path, dest="json_output")
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    compiled = compile_manifest(args.manifest.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(compiled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(compiled), encoding="utf-8")


if __name__ == "__main__":
    main()
