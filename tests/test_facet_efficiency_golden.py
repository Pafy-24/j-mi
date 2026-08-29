from __future__ import annotations

import importlib.util
import json
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "benchmarks" / "facet-efficiency" / "run_golden.py"
SPEC = importlib.util.spec_from_file_location("facet_efficiency_golden", RUNNER)
assert SPEC is not None and SPEC.loader is not None
golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden)


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
            "performance": {"totalMs": 4.5},
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
