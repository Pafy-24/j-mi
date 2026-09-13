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
            "pythonHumanEvalSyntaxPassRate": _round(human.get("syntaxPassRate")),
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


def _resource_evidence_index(
    manifest_dir: Path, manifest: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    raw = manifest.get("resourceEvidence")
    if not raw:
        return {}
    document = _read_json(_resolve(manifest_dir, str(raw)))
    if document.get("schemaVersion") != "jester-rm-benchmark-evidence-v1":
        raise ValueError("Unsupported RM resource evidence schemaVersion")
    return {
        (str(run["id"]), str(state["state"])): state
        for run in document.get("runs", [])
        for state in run.get("states", [])
        if run.get("id") and state.get("state")
    }


def _metric_value(
    metrics: dict[str, Any], metric: str, statistic: str
) -> float | int | None:
    value = _nested(metrics, metric, statistic)
    return _number(value)


def _attach_rm_resources(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    evidence: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for row in rows:
        state = evidence.get((run_id, row["state"]))
        if state is None:
            continue
        summary = state.get("metrics") or {}
        instance_metrics = summary.get("instances") or {}
        per_instance: dict[str, Any] = {}
        rss_peaks: list[float] = []
        cpu_means: list[float] = []
        cpu_peaks: list[float] = []
        storage_peaks: list[float] = []
        sources: set[str] = set()
        confidence: set[str] = set()
        for instance_id, metrics in sorted(instance_metrics.items()):
            if not isinstance(metrics, dict):
                continue
            rss_metric = (
                "component.memory.rss"
                if "component.memory.rss" in metrics
                else "process.memory.rss"
            )
            cpu_metric = (
                "component.cpu.percent"
                if "component.cpu.percent" in metrics
                else "process.cpu.percent"
            )
            values = {
                "peakRssBytes": _metric_value(metrics, rss_metric, "maximum"),
                "meanCpuPercent": _metric_value(metrics, cpu_metric, "mean"),
                "peakCpuPercent": _metric_value(metrics, cpu_metric, "maximum"),
                "peakStorageUsedBytes": _metric_value(
                    metrics, "instance.storage.used", "maximum"
                ),
                "peakAcceleratorAllocatedBytes": _metric_value(
                    metrics, "accelerator.memory.allocated", "maximum"
                ),
                "peakAcceleratorReservedBytes": _metric_value(
                    metrics, "accelerator.memory.reserved", "maximum"
                ),
            }
            per_instance[str(instance_id)] = values
            if values["peakRssBytes"] is not None:
                rss_peaks.append(float(values["peakRssBytes"]))
            if values["meanCpuPercent"] is not None:
                cpu_means.append(float(values["meanCpuPercent"]))
            if values["peakCpuPercent"] is not None:
                cpu_peaks.append(float(values["peakCpuPercent"]))
            if values["peakStorageUsedBytes"] is not None:
                storage_peaks.append(float(values["peakStorageUsedBytes"]))
            for metric in metrics.values():
                if not isinstance(metric, dict):
                    continue
                sources.update(str(item) for item in metric.get("sources", []))
                confidence.update(
                    str(item) for item in metric.get("confidence", [])
                )
        cpu_signal_observed = any(value > 0.0 for value in cpu_peaks)
        row.setdefault("resources", {})["rm"] = {
            "status": state.get("status"),
            "resolution": state.get("resolution"),
            "sampleCount": summary.get("sampleCount"),
            # These are conservative upper bounds: per-instance maxima need not
            # occur in the same sample. The per-instance values remain in JSON.
            "peakRssUpperBoundBytes": sum(rss_peaks) if rss_peaks else None,
            # Historical agents which recreated psutil.Process on every poll
            # emitted an all-zero CPU series.  Preserve those raw values in the
            # evidence JSON, but do not present zero as a measured workload.
            "meanCpuPercentSum": sum(cpu_means)
            if cpu_means and cpu_signal_observed
            else None,
            "peakCpuPercentUpperBound": sum(cpu_peaks)
            if cpu_peaks and cpu_signal_observed
            else None,
            "cpuMeasurementStatus": "captured"
            if cpu_signal_observed
            else "unavailable-all-zero",
            "peakStorageUsedBytes": sum(storage_peaks) if storage_peaks else None,
            "sources": sorted(sources),
            "confidence": sorted(confidence),
            "instances": per_instance,
        }


def _cross_hardware_comparisons(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(str(run["model"]), []).append(run)
    comparisons: list[dict[str, Any]] = []
    for model, model_runs in sorted(by_model.items()):
        rtx = next(
            (
                item
                for item in model_runs
                if "RTX" in str(item.get("inferenceHardware", ""))
            ),
            None,
        )
        a40 = next(
            (
                item
                for item in model_runs
                if "A40" in str(item.get("inferenceHardware", ""))
            ),
            None,
        )
        if not rtx or not a40:
            continue
        rtx_states = {item["state"]: item for item in rtx["states"]}
        a40_states = {item["state"]: item for item in a40["states"]}
        for state_id in STATE_ORDER:
            left = rtx_states.get(state_id)
            right = a40_states.get(state_id)
            if not left or not right:
                continue
            comparisons.append(
                {
                    "model": model,
                    "state": state_id,
                    "status": "measured"
                    if left.get("status") != "not-measured"
                    and right.get("status") != "not-measured"
                    else "not-measured",
                    "a40MinusRtx": {
                        "generalMmlu": _difference(
                            _nested(right, "quality", "generalMmluAccuracy"),
                            _nested(left, "quality", "generalMmluAccuracy"),
                        ),
                        "codingSemantic": _difference(
                            _nested(right, "quality", "codingSemanticCoverage"),
                            _nested(left, "quality", "codingSemanticCoverage"),
                        ),
                        "humanEval": _difference(
                            _nested(right, "quality", "pythonHumanEvalPassAt1"),
                            _nested(left, "quality", "pythonHumanEvalPassAt1"),
                        ),
                    },
                    "a40VsRtxRatio": {
                        "humanEvalThroughput": _ratio(
                            _nested(right, "performance", "humanEvalTokensPerSecond"),
                            _nested(left, "performance", "humanEvalTokensPerSecond"),
                        ),
                        "cudaAllocated": _ratio(
                            _nested(right, "resources", "cudaAllocatedBytes"),
                            _nested(left, "resources", "cudaAllocatedBytes"),
                        ),
                    },
                }
            )
    return comparisons


def _difference(value: Any, baseline: Any) -> float | None:
    left = _number(value)
    right = _number(baseline)
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _parameter_overhead(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in runs:
        model = str(run["model"])
        if model in seen:
            continue
        base = _number(_nested(run, "provenance", "base", "parameterCount"))
        facets = _nested(run, "provenance", "facets") or []
        facet_rows = []
        for facet in facets:
            parameters = _number(facet.get("parameterCount"))
            facet_rows.append(
                {
                    "reference": facet.get("reference"),
                    "requires": facet.get("requires", []),
                    "parameters": parameters,
                    "slots": facet.get("slots"),
                    "blocks": facet.get("blocks"),
                }
            )
        total = sum(
            float(item["parameters"])
            for item in facet_rows
            if item["parameters"] is not None
        )
        result.append(
            {
                "model": model,
                "baseParameters": base,
                "facets": facet_rows,
                "insertedParameters": int(total) if total else None,
                "insertedVsBase": round(total / float(base), 6)
                if total and base not in {None, 0}
                else None,
                "composedParameters": int(float(base) + total)
                if base is not None and total
                else None,
            }
        )
        seen.add(model)
    return result


def compile_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schemaVersion") != "jester-facet-evidence-manifest-v1":
        raise ValueError("Unsupported evidence manifest schemaVersion")
    manifest_dir = manifest_path.parent.resolve()
    resource_evidence = _resource_evidence_index(manifest_dir, manifest)
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
        _attach_rm_resources(
            rows,
            run_id=str(run_entry["id"]),
            evidence=resource_evidence,
        )
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
    training_baseline_id: str | None = None
    training_raw = manifest.get("trainingEvidence")
    if training_raw:
        training_path = _resolve(manifest_dir, training_raw)
        training_document = _read_json(training_path)
        training = list(training_document.get("runs", []))
        training_baseline_id = training_document.get("baselineId")
        baseline = next(
            (item for item in training if item.get("id") == training_baseline_id),
            None,
        )
        if baseline:
            baseline_throughput = baseline.get("tokensPerSecond")
            baseline_elapsed = baseline.get("elapsedSeconds")
            for item in training:
                speedup = _ratio(item.get("tokensPerSecond"), baseline_throughput)
                item["speedupVsBaseline"] = speedup
                item["elapsedVsBaseline"] = _ratio(
                    item.get("elapsedSeconds"), baseline_elapsed
                )
                workers = _number(item.get("workerCount"))
                item["scalingEfficiency"] = (
                    round(float(speedup) / float(workers), 6)
                    if speedup is not None and workers not in {None, 0}
                    else None
                )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "manifest": manifest_path.name,
        "methodology": manifest.get("methodology", {}),
        "runs": compiled_runs,
        "parameterOverhead": _parameter_overhead(compiled_runs),
        "crossHardware": _cross_hardware_comparisons(compiled_runs),
        "training": training,
        "trainingBaselineId": training_baseline_id,
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


def _fmt_int(value: Any) -> str:
    number = _number(value)
    return "n/m" if number is None else f"{int(number):,}"


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
        "## Parameter cost of the composed facets",
        "",
        "| Model | Base parameters | `coder` parameters | `coder.python` parameters | Total inserted | Inserted / base | Composed parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in compiled.get("parameterOverhead", []):
        parent = next(
            (
                facet
                for facet in item.get("facets", [])
                if str(facet.get("reference", "")).startswith("coder@")
            ),
            {},
        )
        child = next(
            (
                facet
                for facet in item.get("facets", [])
                if str(facet.get("reference", "")).startswith("coder.python@")
            ),
            {},
        )
        ratio = _number(item.get("insertedVsBase"))
        lines.append(
            f"| {item['model']} | {_fmt_int(item.get('baseParameters'))} | "
            f"{_fmt_int(parent.get('parameters'))} | "
            f"{_fmt_int(child.get('parameters'))} | "
            f"{_fmt_int(item.get('insertedParameters'))} | "
            f"{float(ratio) * 100:.2f}% | "
            f"{_fmt_int(item.get('composedParameters'))} |"
            if ratio is not None
            else f"| {item['model']} | {_fmt_int(item.get('baseParameters'))} | n/m | n/m | n/m | n/m | n/m |"
        )
    lines.extend(
        [
        "",
        "## Quality and inference cost",
        "",
        "| Model | Trained on | Training rounds | Inference | Configuration | General MMLU [Wilson 95%] | Coding semantic | HE syntax pass | Python HumanEval pass@1 [Wilson 95%] | VRAM allocated | Throughput | p50 latency | Status |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in compiled["runs"]:
        for row in run["states"]:
            quality = row.get("quality", {})
            costs = row.get("costVsBase", {})
            lines.append(
                "| {model} | {trained} | {rounds} | {inference} | `{state}` | {general} | {coding} | {syntax} | {python} | {vram} | {throughput} | {latency} | {status} |".format(
                    model=run["model"],
                    trained=run.get("trainingHardware") or "n/m",
                    rounds=run.get("trainingRounds") or "n/m",
                    inference=run["inferenceHardware"],
                    state=row["state"],
                    general=_fmt_score_ci(
                        quality.get("generalMmluAccuracy"),
                        quality.get("generalMmluWilson95"),
                    ),
                    coding=_fmt(quality.get("codingSemanticCoverage")),
                    syntax=_fmt(quality.get("pythonHumanEvalSyntaxPassRate")),
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
            "## Cross-hardware A40 versus RTX",
            "",
            "> Quality columns are A40 minus RTX. Performance and allocator columns are A40 divided by RTX for the same model, artifacts and state.",
            "",
            "| Model | Configuration | General delta | Coding delta | HumanEval delta | HE throughput ratio | VRAM allocated ratio | Status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in compiled.get("crossHardware", []):
        deltas = item.get("a40MinusRtx") or {}
        ratios = item.get("a40VsRtxRatio") or {}
        lines.append(
            f"| {item['model']} | `{item['state']}` | "
            f"{_fmt(deltas.get('generalMmlu'))} | "
            f"{_fmt(deltas.get('codingSemantic'))} | "
            f"{_fmt(deltas.get('humanEval'))} | "
            f"{_fmt_ratio(ratios.get('humanEvalThroughput'))} | "
            f"{_fmt_ratio(ratios.get('cudaAllocated'))} | "
            f"{item.get('status', 'n/m')} |"
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
            "## RM process-tree resources",
            "",
            "> RSS and CPU totals are conservative sums of per-instance statistics. Peak values are upper bounds because instance maxima may occur in different samples.",
            "> The captured RM intervals cover the design and HumanEval generation windows; the later MMLU pass is not included in these process-tree aggregates.",
            "> Historical all-zero CPU series are rendered `n/m` because they came from a superseded sampler baseline and are not credible zero-utilization measurements.",
            "",
            "| Model | Inference | Configuration | Peak RSS upper bound GiB | Mean CPU sum % | Peak CPU upper bound % | Storage used GiB | RM samples | Resolution | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for run in compiled["runs"]:
        for row in run["states"]:
            rm = _nested(row, "resources", "rm") or {}
            lines.append(
                f"| {run['model']} | {run['inferenceHardware']} | `{row['state']}` | "
                f"{_fmt_gib(rm.get('peakRssUpperBoundBytes'))} | "
                f"{_fmt(rm.get('meanCpuPercentSum'), 2)} | "
                f"{_fmt(rm.get('peakCpuPercentUpperBound'), 2)} | "
                f"{_fmt_gib(rm.get('peakStorageUsedBytes'))} | "
                f"{rm.get('sampleCount', 'n/m')} | "
                f"{rm.get('resolution', 'n/m')} | "
                f"{rm.get('status', 'not captured')} |"
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
            f"> Speedup and efficiency are relative to `{compiled.get('trainingBaselineId') or 'n/m'}` and are shown only for cohort-invariant controlled runs.",
            "> These four-round scaling probes train fresh controlled artifacts; they measure Trainer/RM/MI execution and are not the eight-round facets used in the quality table above.",
            "",
            "| Run | Hardware | Workers | Rounds | Supervised tokens | First loss | Last loss | Loss delta | Compute seconds | tok/s | Speedup | Scaling efficiency | Worker wait s | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not compiled["training"]:
        lines.append("| n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | not measured |")
    for item in compiled["training"]:
        losses = item.get("losses") or []
        loss_delta = (
            float(losses[-1]) - float(losses[0]) if len(losses) >= 2 else None
        )
        lines.append(
            "| {id} | {hardware} | {workers} | {rounds} | {tokens} | {first_loss} | {last_loss} | {loss_delta} | {seconds} | {throughput} | {speedup} | {efficiency} | {wait} | {status} |".format(
                id=item.get("id", "n/m"),
                hardware=item.get("hardware", "n/m"),
                workers=item.get("workerCount", "n/m"),
                rounds=item.get("rounds", "n/m"),
                tokens=item.get("supervisedTokens", "n/m"),
                first_loss=_fmt(losses[0] if losses else None),
                last_loss=_fmt(losses[-1] if losses else item.get("lastTrainingLoss")),
                loss_delta=_fmt(loss_delta),
                seconds=_fmt(item.get("elapsedSeconds"), 2),
                throughput=_fmt(item.get("tokensPerSecond"), 2),
                speedup=_fmt_ratio(item.get("speedupVsBaseline")),
                efficiency=_fmt(item.get("scalingEfficiency"), 3),
                wait=_fmt(item.get("workerWaitSeconds"), 2),
                status=item.get("status", "n/m"),
            )
        )

    lines.extend(
        [
            "",
            "## Controlled training resource snapshot",
            "",
            "> These values are captured immediately after each campaign while the RM topology is still active. They are not labelled as interval peaks.",
            "",
            "| Run | RAM GiB | Accelerator memory GiB | CPU % | Channel sent GiB | Channel received GiB | Source |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not compiled["training"]:
        lines.append("| n/m | n/m | n/m | n/m | n/m | n/m | not measured |")
    for item in compiled["training"]:
        resources = item.get("resources") or {}
        lines.append(
            f"| {item.get('id', 'n/m')} | "
            f"{_fmt_gib(resources.get('ramBytes'))} | "
            f"{_fmt_gib(resources.get('acceleratorMemoryBytes'))} | "
            f"{_fmt(resources.get('cpuPercent'), 2)} | "
            f"{_fmt_gib(resources.get('channelBytesSent'))} | "
            f"{_fmt_gib(resources.get('channelBytesReceived'))} | "
            f"{resources.get('source', 'n/m')} |"
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
