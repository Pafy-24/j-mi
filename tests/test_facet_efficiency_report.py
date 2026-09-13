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

TRAINING_CAPTURE_SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "facet-efficiency"
    / "capture_training_evidence.py"
)
TRAINING_CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "capture_training_evidence", TRAINING_CAPTURE_SCRIPT
)
assert TRAINING_CAPTURE_SPEC and TRAINING_CAPTURE_SPEC.loader
TRAINING_CAPTURE_MODULE = importlib.util.module_from_spec(TRAINING_CAPTURE_SPEC)
TRAINING_CAPTURE_SPEC.loader.exec_module(TRAINING_CAPTURE_MODULE)


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
    resource_evidence = tmp_path / "rm-metrics.json"
    resource_evidence.write_text(
        json.dumps(
            {
                "schemaVersion": "jester-rm-benchmark-evidence-v1",
                "runs": [
                    {
                        "id": "demo",
                        "states": [
                            {
                                "state": "base",
                                "status": "captured",
                                "resolution": "minute",
                                "metrics": {
                                    "sampleCount": 2,
                                    "instances": {
                                        "mi-01": {
                                            "component.memory.rss": {
                                                "maximum": 100,
                                                "sources": ["psutil"],
                                                "confidence": ["exact"],
                                            },
                                            "component.cpu.percent": {
                                                "mean": 25,
                                                "maximum": 50,
                                                "sources": ["psutil"],
                                                "confidence": ["exact"],
                                            },
                                        },
                                        "lsr-01": {
                                            "component.memory.rss": {
                                                "maximum": 40,
                                                "sources": ["psutil"],
                                                "confidence": ["estimated"],
                                            }
                                        },
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    training_evidence = tmp_path / "training.json"
    training_evidence.write_text(
        json.dumps(
            {
                "baselineId": "one-gpu",
                "runs": [
                    {
                        "id": "one-gpu",
                        "workerCount": 1,
                        "tokensPerSecond": 10,
                        "elapsedSeconds": 20,
                    },
                    {
                        "id": "two-gpu",
                        "workerCount": 2,
                        "tokensPerSecond": 15,
                        "elapsedSeconds": 14,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": "jester-facet-evidence-manifest-v1",
                "resourceEvidence": "rm-metrics.json",
                "trainingEvidence": "training.json",
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
    assert rows["base"]["resources"]["rm"]["peakRssUpperBoundBytes"] == 140
    assert rows["base"]["resources"]["rm"]["meanCpuPercentSum"] == 25
    assert compiled["training"][1]["speedupVsBaseline"] == 1.5
    assert compiled["training"][1]["scalingEfficiency"] == 0.75
    markdown = MODULE.render_markdown(compiled)
    assert "not-measured" in markdown
    assert "RM process-tree resources" in markdown


def test_all_zero_historical_cpu_series_is_not_reported_as_zero_usage() -> None:
    rows = [{"state": "base", "resources": {}}]
    evidence = {
        ("demo", "base"): {
            "status": "captured",
            "resolution": "minute",
            "metrics": {
                "sampleCount": 2,
                "instances": {
                    "mi": {
                        "component.cpu.percent": {
                            "mean": 0.0,
                            "maximum": 0.0,
                            "sources": ["jester-rm-minute-rollup"],
                            "confidence": ["exact"],
                        }
                    }
                },
            },
        }
    }

    MODULE._attach_rm_resources(rows, run_id="demo", evidence=evidence)

    resources = rows[0]["resources"]["rm"]
    assert resources["meanCpuPercentSum"] is None
    assert resources["peakCpuPercentUpperBound"] is None
    assert resources["cpuMeasurementStatus"] == "unavailable-all-zero"


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


def test_rm_sample_summary_aggregates_minute_rollups_with_weighted_mean() -> None:
    samples = [
        {
            "instanceId": "mi-01",
            "metric": "process.memory.rss",
            "value": 120,
            "unit": "bytes",
            "sampledAt": "2026-08-31T10:00:00Z",
            "source": "jester-rm-minute-rollup",
            "confidence": "exact",
            "statistics": {
                "samples": 3,
                "minimum": 100,
                "maximum": 140,
                "sum": 360,
            },
        },
        {
            "instanceId": "mi-01",
            "metric": "process.memory.rss",
            "value": 180,
            "unit": "bytes",
            "sampledAt": "2026-08-31T10:01:00Z",
            "source": "jester-rm-minute-rollup",
            "confidence": "exact",
            "statistics": {
                "samples": 1,
                "minimum": 180,
                "maximum": 180,
                "sum": 180,
            },
        },
    ]

    metric = CAPTURE_MODULE._summarize_samples(samples)["instances"]["mi-01"][
        "process.memory.rss"
    ]

    assert metric["count"] == 4
    assert metric["minimum"] == 100
    assert metric["maximum"] == 180
    assert metric["mean"] == 135
    assert metric["p50"] is None
    assert metric["p95"] is None


def test_training_capture_uses_only_commits_after_last_failed_attempt() -> None:
    evidence = TRAINING_CAPTURE_MODULE.compile_campaign(
        {
            "id": "controlled",
            "campaignId": "campaign-1",
            "hardware": "two GPUs",
            "workerCount": 2,
            "expectedRounds": 2,
        },
        {"state": "succeeded"},
        [{"jobId": "job-1", "state": "succeeded"}],
        [
            {
                "jobId": "job-1",
                "facetReference": "coder@test",
                "artifactSha256": "abc",
                "metrics": {
                    "rounds": 2,
                    "cohort": ["a", "b"],
                    "supervisedTokens": 100,
                    "elapsedSeconds": 10,
                    "tokensPerSecond": 10,
                    "lastTrainingLoss": 1.5,
                },
            }
        ],
        [
            {
                "sequence": 1,
                "jobId": "job-1",
                "type": "ROUND_COMMITTED",
                "data": {"loss": 9.0},
            },
            {"sequence": 2, "jobId": "job-1", "type": "JOB_ATTEMPT_FAILED"},
            {
                "sequence": 3,
                "jobId": "job-1",
                "type": "ROUND_COMMITTED",
                "data": {"loss": 2.0, "durationSeconds": 4.0},
            },
            {
                "sequence": 4,
                "jobId": "job-1",
                "type": "ROUND_COMMITTED",
                "data": {"loss": 1.5, "durationSeconds": 6.0},
            },
        ],
        {
            "resources": {
                "source": "rm-topology-metrics",
                "topologyMeasured": {"ramBytes": 1024, "acceleratorMemoryBytes": 2048},
                "channels": [
                    {"counters": {"bytesSent": 100, "bytesReceived": 200}}
                ],
            }
        },
    )

    assert evidence["status"] == "measured"
    assert evidence["losses"] == [2.0, 1.5]
    assert evidence["roundSeconds"] == [4.0, 6.0]
    assert evidence["resourceSnapshot"]["source"] == "rm-topology-metrics"
    assert evidence["resources"]["acceleratorMemoryBytes"] == 2048
    assert evidence["resources"]["channelBytesSent"] == 100


def test_cross_hardware_comparison_keeps_quality_delta_separate_from_speed() -> None:
    def run(hardware: str, score: float, throughput: float) -> dict:
        return {
            "model": "demo",
            "inferenceHardware": hardware,
            "states": [
                {
                    "state": "base",
                    "status": "measured",
                    "quality": {
                        "generalMmluAccuracy": score,
                        "codingSemanticCoverage": score,
                        "pythonHumanEvalPassAt1": score,
                    },
                    "performance": {"humanEvalTokensPerSecond": throughput},
                    "resources": {"cudaAllocatedBytes": 100},
                }
            ],
        }

    comparison = MODULE._cross_hardware_comparisons(
        [run("RTX 5070", 0.4, 10), run("NVIDIA A40", 0.5, 15)]
    )[0]

    assert comparison["a40MinusRtx"]["humanEval"] == 0.1
    assert comparison["a40VsRtxRatio"]["humanEvalThroughput"] == 1.5
    assert comparison["a40VsRtxRatio"]["cudaAllocated"] == 1.0
