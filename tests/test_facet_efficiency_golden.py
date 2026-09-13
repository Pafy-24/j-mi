from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


RUNNER = Path(__file__).resolve().parents[1] / "benchmarks" / "facet-efficiency" / "run_golden.py"
SPEC = importlib.util.spec_from_file_location("facet_efficiency_golden", RUNNER)
assert SPEC is not None and SPEC.loader is not None
golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden)


def test_safetensors_statistics_reads_only_header_metadata() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "weights.safetensors"
        save_file(
            {
                "a": torch.zeros((2, 3), dtype=torch.bfloat16),
                "b": torch.zeros((4,), dtype=torch.float32),
            },
            str(path),
        )

        result = golden.safetensors_statistics(path)

    assert result == {
        "tensorCount": 2,
        "parameterCount": 10,
        "dtypeParameterCounts": {"BF16": 6, "F32": 4},
    }


def test_root_json_parser_rejects_trailing_or_truncated_contracts() -> None:
    assert golden.extract_root_json('{"contract":"code-design-v3"}') == {
        "contract": "code-design-v3"
    }
    assert golden.extract_root_json('{"contract":"code-design-v3"} trailing') is None
    assert golden.extract_root_json('{"contract":"code-design-v3"') is None


def test_design_score_requires_schema_semantics_and_minimums() -> None:
    record = {
        "checks": {
            "semanticGroups": [
                {"id": "identity", "anyOf": ["idempotency key"]},
                {"id": "commit", "anyOf": ["atomic commit"]},
            ],
            "minInvariants": 2,
            "minControlFlowSteps": 2,
            "minTests": 2,
        }
    }
    payload = {
        "contract": "code-design-v3",
        "requirements": ["idempotency key"],
        "components": [],
        "data_model": {},
        "invariants": ["one", "two"],
        "control_flow": ["validate", "atomic commit"],
        "pseudocode": [],
        "errors": [],
        "complexity": "O(1)",
        "test_strategy": ["duplicate", "crash"],
        "assumptions": [],
    }
    score = golden.score_design(record, json.dumps(payload))
    assert score["strictPass"] is True
    assert score["fieldCoverage"] == 1.0
    assert score["semanticCoverage"] == 1.0


def test_humaneval_normalizer_extracts_function_body_and_localizes_imports() -> None:
    generated = '''```python
import math

def area(radius):
    """Generated duplicate docstring."""
    if radius < 0:
        raise ValueError("negative")
    return math.pi * radius * radius
```'''
    completion = golden.normalize_humaneval_completion(generated, "area")
    prompt = 'def area(radius):\n    """Return circle area."""\n'
    compile(prompt + completion, "HumanEval/test", "exec")
    assert completion.startswith("    import math")
    assert "Generated duplicate docstring" not in completion
    assert "        raise ValueError" in completion


def test_humaneval_normalizer_accepts_mi_files_contract() -> None:
    generated = json.dumps(
        {
            "contract": "code-implementation-v3",
            "files": {"answer.py": "def increment(value):\n    return value + 1\n"},
        }
    )
    completion = golden.normalize_humaneval_completion(generated, "increment")
    compile("def increment(value):\n" + completion, "HumanEval/test", "exec")
    assert completion == "    return value + 1\n"


def test_paired_bootstrap_reports_direction_and_is_deterministic() -> None:
    baseline = {"a": False, "b": True, "c": False}
    candidate = {"a": True, "b": True, "c": False}
    first = golden.bootstrap_delta(baseline, candidate, samples=1000)
    second = golden.bootstrap_delta(baseline, candidate, samples=1000)
    assert first == second
    assert first["delta"] == 0.333333
    assert first["improved"] == 1
    assert first["regressed"] == 0
    assert first["improvedIds"] == ["a"]
    assert first["regressedIds"] == []


def test_lsr_route_maps_only_direct_ids_to_pinned_versions(monkeypatch) -> None:
    def fake_request(method, url, payload, **_):
        assert method == "POST"
        assert url == "http://127.0.0.1:8765/rank"
        assert payload["top_k"] == 1
        return {
            "schemaVersion": "lsr-route-v2",
            "selectedFacets": [{"facet": "coder.python", "weight": 0.87}],
            "performance": {"timing": {"rankingMs": 4.5}},
        }

    monkeypatch.setattr(golden, "request_json", fake_request)
    facets, route = golden.request_facets(
        {
            "lsr": {
                "url": "http://127.0.0.1:8765",
                "topK": 1,
                "facetReferences": {
                    "coder.python": "coder.python@a40-seed-42"
                },
            }
        },
        {"id": "lsr-routed", "routeWithLsr": True},
        "Implement this Python function",
        "coder.python",
    )

    assert facets == ["coder.python@a40-seed-42"]
    assert route["exactDirectSelection"] is True
    assert route["expectedSelected"] is True
    assert "weight" not in facets[0]
    aggregate = golden.aggregate_routing([{"route": route}])
    assert aggregate["latencyMs"]["mean"] == 4.5
    assert aggregate["expectedDecisionRate"] == 1.0
    assert aggregate["selectedFacetCounts"] == {"coder.python": 1}


def test_lsr_general_route_scores_abstention_as_expected_decision(monkeypatch) -> None:
    monkeypatch.setattr(
        golden,
        "request_json",
        lambda *_args, **_kwargs: {
            "schemaVersion": "lsr-route-v2",
            "selectedFacets": [],
            "performance": {"timing": {"rankingMs": 3.0}},
        },
    )

    facets, route = golden.request_facets(
        {
            "lsr": {
                "url": "http://127.0.0.1:8765",
                "topK": 1,
                "facetReferences": {"coder": "coder@seed-42"},
            }
        },
        {"id": "lsr-routed", "routeWithLsr": True},
        "Which treaty ended the war?",
        None,
    )

    assert facets == []
    assert route["expectedAbstain"] is True
    assert route["expectedDecision"] is True
    aggregate = golden.aggregate_routing([{"route": route}])
    assert aggregate["expectedAbstentionRate"] == 1.0
    assert aggregate["expectedRecall"] is None


def test_mmlu_parser_and_aggregate_keep_format_separate_from_accuracy() -> None:
    assert golden.parse_mmlu_answer("B") == ("B", True)
    assert golden.parse_mmlu_answer("Answer: C") == ("C", False)
    assert golden.parse_mmlu_answer("I do not know") == (None, False)

    aggregate = golden.aggregate_mmlu(
        [
            {
                "category": "humanities",
                "usage": {"outputTokens": 1},
                "timingMs": {"generation": 10, "total": 12},
                "score": {"correct": True, "exactFormat": True},
            },
            {
                "category": "humanities",
                "usage": {"outputTokens": 2},
                "timingMs": {"generation": 20, "total": 22},
                "score": {"correct": False, "exactFormat": False},
            },
            {"category": "stem_non_cs", "error": "failed"},
        ]
    )

    assert aggregate["accuracy"] == 0.333333
    assert aggregate["exactFormatRate"] == 0.5
    assert aggregate["categories"]["humanities"]["accuracy"] == 0.5
    assert aggregate["categories"]["stem_non_cs"]["accuracy"] == 0.0


def test_generation_aggregate_reports_latency_distribution() -> None:
    aggregate = golden.aggregate_generation(
        [
            {
                "usage": {"outputTokens": 10},
                "timingMs": {"generation": 100, "total": 120},
            },
            {
                "usage": {"outputTokens": 20},
                "timingMs": {"generation": 300, "total": 340},
            },
            {"error": "failed"},
        ]
    )

    assert aggregate["completed"] == 2
    assert aggregate["errors"] == 1
    assert aggregate["tokensPerSecond"] == 75.0
    assert aggregate["latencyMs"]["generationP50"] == 200.0
    assert aggregate["latencyMs"]["generationP95"] == 290.0


def test_get_request_retries_a_transient_connection_reset(monkeypatch) -> None:
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"status":"ready"}'

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("relay reset")
        return Response()

    monkeypatch.setattr(golden.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(golden.time, "sleep", lambda _seconds: None)

    assert golden.request_json("GET", "http://127.0.0.1/health") == {
        "status": "ready"
    }
    assert calls == 2


def test_post_request_retries_transport_reset_with_same_payload(monkeypatch) -> None:
    calls: list[bytes | None] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"output":{"text":"ok"}}'

    def fake_urlopen(request, **_kwargs):
        calls.append(request.data)
        if len(calls) == 1:
            raise ConnectionResetError("relay reset")
        return Response()

    monkeypatch.setattr(golden.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(golden.time, "sleep", lambda _seconds: None)

    result = golden.request_json(
        "POST",
        "http://127.0.0.1/generate",
        {"requestId": "stable", "seed": 42},
    )

    assert result == {"output": {"text": "ok"}}
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_recover_completed_state_rebuilds_metadata_without_regeneration(
    tmp_path: Path,
) -> None:
    design_path = tmp_path / "design.jsonl"
    samples_path = tmp_path / "samples.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"
    design_path.write_text(
        json.dumps(
            {
                "id": "design-1",
                "scores": {
                    "strictPass": True,
                    "validRootJson": True,
                    "fieldCoverage": 1.0,
                    "semanticCoverage": 1.0,
                },
                "usage": {"outputTokens": 4},
                "timingMs": {"generation": 10, "total": 12},
                "route": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples_path.write_text(
        json.dumps({"task_id": "HumanEval/0", "completion": "    return 1\n"})
        + "\n",
        encoding="utf-8",
    )
    telemetry_path.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "syntaxPassed": True,
                "usage": {"outputTokens": 3},
                "timingMs": {"generation": 20, "total": 22},
                "route": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recovered = golden.recover_completed_state(
        state={"id": "base+coder", "facets": ["coder@test"]},
        designs=[{"id": "design-1"}],
        problems=[{"task_id": "HumanEval/0"}],
        design_output=design_path,
        samples_output=samples_path,
        telemetry_output=telemetry_path,
        previous_state={"finishedAtUnix": 1.0, "miHealthAfter": {"status": "ready"}},
        health_after={"status": "ready", "loadedFacets": ["coder@test"]},
    )

    assert recovered["recoveredFromCompleteOutputs"] is True
    assert recovered["design"]["strictPassRate"] == 1.0
    assert recovered["humanEval"]["syntaxPassRate"] == 1.0
    assert recovered["humanEval"]["functionalPassAt1"] == "pending-container-evaluation"


def test_failed_general_attempt_is_archived_without_losing_evidence(
    tmp_path: Path,
) -> None:
    record = {
        "status": "evaluated",
        "suites": {"generalMmlu": {"records": 2}},
        "generalRetention": {"status": "generated"},
        "pairedGeneralMmlu": {"base+coder": {"delta": 0.0}},
        "states": [
            {"id": "base", "generalMmlu": {"errors": 2}},
            {"id": "base+coder", "generalMmlu": {"errors": 2}},
        ],
    }
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "general-mmlu-problems.jsonl").write_text(
        '{"id":"one"}\n{"id":"two"}\n', encoding="utf-8"
    )
    for state in ("base", "base_coder"):
        (tmp_path / f"general-mmlu.{state}.jsonl").write_text(
            '{"id":"one","error":"connection refused"}\n'
            '{"id":"two","error":"connection refused"}\n',
            encoding="utf-8",
        )

    recovered = golden.archive_failed_general_attempt(tmp_path, record)

    assert recovered["status"] == "generated"
    assert "generalMmlu" not in recovered["suites"]
    assert "pairedGeneralMmlu" not in recovered
    assert all("generalMmlu" not in state for state in recovered["states"])
    archive = tmp_path / recovered["invalidGeneralAttempts"][0]["archiveDirectory"]
    assert (archive / "general-mmlu.base.jsonl").is_file()
    assert (archive / "general-mmlu.base_coder.jsonl").is_file()
    assert not (tmp_path / "general-mmlu.base.jsonl").exists()


def test_failed_state_retry_archives_outputs_and_preserves_general_results(
    tmp_path: Path,
) -> None:
    state_id = "base+parent+child"
    safe_id = "base_parent_child"
    state = {
        "id": state_id,
        "design": {"attempted": 24},
        "humanEval": {"attempted": 164, "generationErrors": 8},
        "generalMmlu": {"accuracy": 0.0},
    }
    record = {
        "status": "evaluated",
        "states": [{"id": "base"}, state, {"id": "lsr-routed"}],
        "pairedHumanEval": {state_id: {"delta": -0.3}},
        "pairedDesign": {state_id: {"delta": -0.6}},
        "pairedGeneralMmlu": {state_id: {"delta": -0.2}},
    }
    generated = [
        tmp_path / f"design.{safe_id}.jsonl",
        tmp_path / f"humaneval-samples.{safe_id}.jsonl",
        tmp_path / f"humaneval-telemetry.{safe_id}.jsonl",
        tmp_path / f"humaneval-samples.{safe_id}.jsonl_results.jsonl",
    ]
    for path in generated:
        path.write_text('{"evidence":true}\n', encoding="utf-8")
    general = tmp_path / f"general-mmlu.{safe_id}.jsonl"
    general.write_text('{"id":"general-1","score":{"correct":false}}\n', encoding="utf-8")

    recovered = golden.archive_failed_states(tmp_path, record, [state_id])

    assert [item["id"] for item in recovered["states"]] == ["base", "lsr-routed"]
    assert state_id not in recovered["pairedHumanEval"]
    assert state_id not in recovered["pairedDesign"]
    assert state_id not in recovered["pairedGeneralMmlu"]
    archive = tmp_path / recovered["invalidStateAttempts"][0]["archiveDirectory"]
    assert (archive / "attempt.json").is_file()
    assert all((archive / path.name).is_file() for path in generated)
    assert all(not path.exists() for path in generated)
    assert general.is_file()


def test_failed_state_retry_can_archive_complete_uncommitted_outputs(
    tmp_path: Path,
) -> None:
    state_id = "base+parent+child"
    safe_id = "base_parent_child"
    record = {"status": "running", "states": [{"id": "base"}]}
    generated = [
        tmp_path / f"design.{safe_id}.jsonl",
        tmp_path / f"humaneval-samples.{safe_id}.jsonl",
        tmp_path / f"humaneval-telemetry.{safe_id}.jsonl",
    ]
    for path in generated:
        path.write_text('{"error":"relay reset"}\n', encoding="utf-8")

    recovered = golden.archive_failed_states(tmp_path, record, [state_id])

    attempt = recovered["invalidStateAttempts"][0]
    assert recovered["states"] == [{"id": "base"}]
    assert attempt["stateId"] == state_id
    archive = tmp_path / attempt["archiveDirectory"]
    metadata = json.loads((archive / "attempt.json").read_text(encoding="utf-8"))
    assert metadata["wasCommitted"] is False
    assert metadata["previousState"] is None


def test_reconcile_unreferenced_invalid_state_archive(tmp_path: Path) -> None:
    archive = tmp_path / "invalid-states" / "base_coder-attempt-20260903T000000Z"
    archive.mkdir(parents=True)
    files = [
        "design.base_coder.jsonl",
        "humaneval-samples.base_coder.jsonl",
        "humaneval-telemetry.base_coder.jsonl",
    ]
    for name in files:
        (archive / name).write_text("{}\n", encoding="utf-8")
    (archive / "attempt.json").write_text(
        json.dumps({"stateId": "base+coder", "files": files}),
        encoding="utf-8",
    )

    recovered = golden.reconcile_invalid_state_attempts(
        tmp_path, {"states": [], "invalidStateAttempts": []}
    )

    assert recovered["invalidStateAttempts"] == [
        {
            "stateId": "base+coder",
            "archiveDirectory": (
                "invalid-states/base_coder-attempt-20260903T000000Z"
            ),
            "files": files,
            "recoveredFromArchiveManifest": True,
        }
    ]
