from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "facet-efficiency"
    / "compile_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("compile_evidence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CAPTURE_SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "facet-efficiency"
    / "capture_rm_metrics.py"
)
CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "capture_rm_metrics", CAPTURE_SCRIPT
)
assert CAPTURE_SPEC and CAPTURE_SPEC.loader
CAPTURE_MODULE = importlib.util.module_from_spec(CAPTURE_SPEC)
CAPTURE_SPEC.loader.exec_module(CAPTURE_MODULE)


def _state(state_id: str, *, score: float, memory: int) -> dict:
    return {
        "id": state_id,
        "facets": [],
        "routeWithLsr": state_id == "base+lsr+mi",
        "wallSeconds": 10.0,
        "design": {
            "attempted": 1,
            "completed": 1,
            "semanticCoverage": score,
            "validRootJsonRate": 1.0,
            "fieldCoverage": score,
            "tokensPerSecond": 20.0,
            "latencyMs": {"generationP50": 50.0},
        },
        "humanEval": {
            "attempted": 1,
            "completed": 1,
            "evaluated": 1,
            "passed": int(score > 0.5),
            "functionalPassAt1": score,
            "functionalPassWilson95": [0.0, 1.0],
            "tokensPerSecond": 10.0,
            "latencyMs": {"generationP50": 100.0},
        },
        "generalMmlu": {
            "attempted": 1,
            "completed": 1,
            "accuracy": score,
            "wilson95": [0.0, 1.0],
        },
        "routing": {
            "expectedDecisionRate": score,
            "expectedRecall": score,
            "expectedAbstentionRate": score,
            "latencyMs": {"p50": 2.0},
        },
        "miHealthAfter": {
            "memory": {
                "cudaAllocatedBytes": memory,
                "cudaReservedBytes": memory + 1,
            }
        },
    }


def _write_run(path: Path, states: list[dict]) -> None:
    path.mkdir(parents=True)
    (path / "run.json").write_text(
        json.dumps({"status": "evaluated", "states": states}), encoding="utf-8"
    )


def test_compiler_uses_suite_override_and_never_invents_missing_states(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    general = tmp_path / "general"
    _write_run(
        primary,
        [
            _state("base", score=0.5, memory=100),
            _state("base+lsr+mi", score=0.75, memory=125),
        ],
    )
    override_state = _state("base+lsr+mi", score=0.25, memory=999)
    _write_run(general, [override_state])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": "jester-facet-evidence-manifest-v1",
                "runs": [
                    {
                        "id": "demo",
                        "model": "demo-model",
                        "inferenceHardware": "demo-gpu",
                        "run": "primary",
                        "suiteSources": {
                            "base+lsr+mi": {"generalMmlu": "general"}
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    compiled = MODULE.compile_manifest(manifest)
    rows = {row["state"]: row for row in compiled["runs"][0]["states"]}

    assert rows["base"]["status"] == "measured"
    assert rows["base+coder"]["status"] == "not-measured"
    assert rows["base+lsr+mi"]["quality"]["generalMmluAccuracy"] == 0.25
    assert rows["base+lsr+mi"]["quality"]["codingSemanticCoverage"] == 0.75
    assert rows["base+lsr+mi"]["resources"]["cudaAllocatedBytes"] == 125
    assert rows["base+lsr+mi"]["costVsBase"]["cudaAllocated"] == 1.25
    assert "not-measured" in MODULE.render_markdown(compiled)


def test_rm_sample_summary_preserves_instance_boundaries_and_provenance() -> None:
    samples = [
        {
            "instanceId": "mi-01",
            "metric": "process.memory.rss",
            "value": 100,
            "unit": "bytes",
            "sampledAt": "2026-08-31T10:00:00Z",
            "source": "psutil",
            "confidence": "exact",
        },
        {
            "instanceId": "mi-01",
            "metric": "process.memory.rss",
            "value": 140,
            "unit": "bytes",
            "sampledAt": "2026-08-31T10:00:01Z",
            "source": "psutil",
            "confidence": "exact",
        },
        {
            "instanceId": "lsr-01",
            "metric": "process.memory.rss",
            "value": 30,
            "unit": "bytes",
            "sampledAt": "2026-08-31T10:00:01Z",
            "source": "psutil",
            "confidence": "estimated",
        },
    ]

    summary = CAPTURE_MODULE._summarize_samples(samples)

    assert summary["sampleCount"] == 3
    assert summary["instances"]["mi-01"]["process.memory.rss"]["maximum"] == 140
    assert summary["instances"]["mi-01"]["process.memory.rss"]["delta"] == 40
    assert summary["instances"]["lsr-01"]["process.memory.rss"]["confidence"] == [
        "estimated"
    ]
