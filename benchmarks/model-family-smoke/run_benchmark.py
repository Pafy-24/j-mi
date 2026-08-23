from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parent
MI_ROOT = BENCHMARK_ROOT.parents[1]
WORKSPACE = MI_ROOT.parent
if str(MI_ROOT) not in sys.path:
    sys.path.insert(0, str(MI_ROOT))

from mixed_inference.core.graph import CompositionPlan, PlannedBlock
from mixed_inference.core.modeling import ActivatedComposition
from mixed_inference.generate.runtime import MixedInferenceRuntime
from mixed_inference.training.config import DataConfig
from mixed_inference.training.data import CausalCollator, read_records, tokenize_record
from mixed_inference.training.trainer import MixedInferenceTrainer


VERSION = "smoke-2026-08-23"
RESULTS = BENCHMARK_ROOT / "results"
MODELS: dict[str, dict[str, Any]] = {
    "qwen": {
        "base": WORKSPACE / "base-models/downloaded/qwen/Qwen2.5-1.5B-Instruct",
        "registry": WORKSPACE / "facets/mi-benchmark/qwen",
        "configs": [
            BENCHMARK_ROOT / "configs/qwen-coder.json",
            BENCHMARK_ROOT / "configs/qwen-coder-python.json",
        ],
    },
    "gemma": {
        "base": WORKSPACE / "base-models/downloaded/gemma/gemma-3-1b-it",
        "registry": WORKSPACE / "facets/mi-benchmark/gemma",
        "configs": [
            BENCHMARK_ROOT / "configs/gemma-coder.json",
            BENCHMARK_ROOT / "configs/gemma-coder-python.json",
        ],
    },
    "llama": {
        "base": WORKSPACE / "base-models/downloaded/llama/Llama-3.2-1B-Instruct",
        "registry": WORKSPACE / "facets/mi-benchmark/llama",
        "configs": [
            BENCHMARK_ROOT / "configs/llama-coder.json",
            BENCHMARK_ROOT / "configs/llama-coder-python.json",
        ],
    },
}
HOLDOUTS = {
    "coder": WORKSPACE / "datasets/facet-infrastructure/facets/coder/training.jsonl",
    "coder.python": WORKSPACE
    / "datasets/facet-infrastructure/facets/coder.python/training.jsonl",
}
REQUIRED_FIELDS = {
    "coder": (
        "requirements",
        "invariants",
        "algorithm",
        "pseudocode",
        "edge_cases",
        "complexity",
        "test_strategy",
    ),
    "coder.python": ("files", "compatibility", "assumptions"),
}
STATES = (
    ("base", []),
    ("base+facet", [f"coder@{VERSION}"]),
    ("base+parent+child", [f"coder.python@{VERSION}"]),
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def clean_gpu(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def progress(family: str, role: str):
    def callback(event: dict[str, Any]) -> None:
        fields = [f"event={event.get('event')}"]
        for name in ("step", "totalSteps", "loss", "gradientNorm", "passed"):
            if event.get(name) is not None:
                fields.append(f"{name}={event[name]}")
        print(f"[{family}/{role}] " + " ".join(fields), flush=True)

    return callback


def train_family(family: str, spec: dict[str, Any], torch: Any) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for config_path in spec["configs"]:
        role = "child" if "python" in config_path.stem else "parent"
        clean_gpu(torch)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer = MixedInferenceTrainer(
            config_path,
            progress_callback=progress(family, role),
        )
        result = trainer.run()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        output = Path(result["outputDirectory"])
        artifact = output / "facet.safetensors"
        runs.append(
            {
                "role": role,
                "config": str(config_path),
                "wallSeconds": round(elapsed, 6),
                "peakCudaAllocatedBytes": peak,
                "artifactBytes": artifact.stat().st_size,
                "result": result,
            }
        )
        del trainer
        clean_gpu(torch)
    return runs


def heldout_records() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for facet, path in HOLDOUTS.items():
        records = list(read_records(path))
        if len(records) < 6:
            raise RuntimeError(f"Expected at least six SFT records in {path}, found {len(records)}")
        result[facet] = records[4:6]
    return result


def target_text(record: dict[str, Any]) -> str:
    messages = (record.get("target") or {}).get("messages") or []
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    data = record.get("input") or {}
    return list(data.get("conditioningMessages") or []) + list(data.get("messages") or [])


def lexical_f1(candidate: str, reference: str) -> float:
    token_pattern = re.compile(r"[\w]+", re.UNICODE)
    candidate_tokens = token_pattern.findall(candidate.lower())
    reference_tokens = token_pattern.findall(reference.lower())
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(candidate_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(stripped)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def python_parse_rate(value: dict[str, Any] | None, text: str) -> float:
    code_files: list[str] = []
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        code_files.extend(
            str(content)
            for name, content in value["files"].items()
            if str(name).lower().endswith(".py")
        )
    if not code_files:
        code_files.extend(
            match.group(1).strip()
            for match in re.finditer(
                r"```python\s*(.*?)(?:```|\Z)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match.group(1).strip()
        )
    if not code_files:
        return 0.0
    valid = 0
    for content in code_files:
        try:
            ast.parse(content)
            valid += 1
        except SyntaxError:
            pass
    return valid / len(code_files)


def score_generation(facet: str, text: str, reference: str) -> dict[str, Any]:
    parsed = extract_json_object(text)
    fields = REQUIRED_FIELDS[facet]
    coverage = sum(1 for name in fields if isinstance(parsed, dict) and name in parsed) / len(fields)
    lowered = text.lower()
    mention_coverage = sum(
        1
        for name in fields
        if re.search(re.escape(name).replace("_", r"[\s_-]+"), lowered)
    ) / len(fields)
    syntax = python_parse_rate(parsed, text) if facet == "coder.python" else None
    strict_components = [1.0 if parsed is not None else 0.0, coverage]
    structure_components = [1.0 if parsed is not None else 0.0, mention_coverage]
    if syntax is not None:
        strict_components.append(syntax)
        structure_components.append(syntax)
    return {
        "validContractJson": parsed is not None,
        "requiredFieldCoverage": round(coverage, 6),
        "requiredFieldMentionCoverage": round(mention_coverage, 6),
        "pythonParseRate": round(syntax, 6) if syntax is not None else None,
        "lexicalF1": round(lexical_f1(text, reference), 6),
        "strictContractScore": round(sum(strict_components) / len(strict_components), 6),
        "structureScore": round(sum(structure_components) / len(structure_components), 6),
    }


def mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def aggregate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    def values(name: str) -> list[float]:
        return [float(item["scores"][name]) for item in examples if item["scores"].get(name) is not None]

    generated_tokens = sum(int(item["usage"]["outputTokens"]) for item in examples)
    generation_ms = sum(float(item["timingMs"]["generation"]) for item in examples)
    return {
        "examples": len(examples),
        "validContractJsonRate": mean([1.0 if item["scores"]["validContractJson"] else 0.0 for item in examples]),
        "requiredFieldCoverage": mean(values("requiredFieldCoverage")),
        "requiredFieldMentionCoverage": mean(values("requiredFieldMentionCoverage")),
        "pythonParseRate": mean(values("pythonParseRate")),
        "lexicalF1": mean(values("lexicalF1")),
        "strictContractScore": mean(values("strictContractScore")),
        "structureScore": mean(values("structureScore")),
        "generatedTokens": generated_tokens,
        "generationSeconds": round(generation_ms / 1000.0, 6),
        "tokensPerSecond": round(generated_tokens / (generation_ms / 1000.0), 6) if generation_ms else None,
    }


def evaluate_loss(
    runtime: MixedInferenceRuntime,
    plan: CompositionPlan,
    loaded: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    torch: Any,
) -> dict[str, Any]:
    assert runtime.base is not None
    base = runtime.base
    data_config = DataConfig(train=(), max_sequence_length=512, mask_prompt=True)
    collator = CausalCollator(base.tokenizer.pad_token_id)

    def provide(block: PlannedBlock) -> Any:
        return loaded[block.facet].block(block.slot_id, block.block_index)

    result: dict[str, Any] = {}
    previous = base.model.training
    base.model.eval()
    with ActivatedComposition(base, plan, provide), torch.inference_mode():
        for facet, items in records.items():
            total_nll = 0.0
            total_tokens = 0
            per_record = []
            for record in items:
                tokenized = tokenize_record(base.tokenizer, record, data_config)
                batch = collator(
                    [
                        {
                            "input_ids": tokenized.input_ids,
                            "labels": tokenized.labels,
                            "record_id": tokenized.record_id,
                        }
                    ]
                )
                batch = {name: tensor.to(base.device) for name, tensor in batch.items()}
                output = base.model(**batch, use_cache=False)
                valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                loss = float(output.loss.detach().float().item())
                total_nll += loss * valid_tokens
                total_tokens += valid_tokens
                per_record.append(
                    {
                        "id": tokenized.record_id,
                        "loss": round(loss, 8),
                        "targetTokens": valid_tokens,
                        "sequenceTokens": len(tokenized.input_ids),
                        "hitMaxSequenceLength": len(tokenized.input_ids) == data_config.max_sequence_length,
                    }
                )
            average = total_nll / max(1, total_tokens)
            result[facet] = {
                "loss": round(average, 8),
                "perplexity": round(math.exp(min(average, 80.0)), 8),
                "targetTokens": total_tokens,
                "records": per_record,
            }
    base.model.train(previous)
    return result


def evaluate_state(
    runtime: MixedInferenceRuntime,
    state: str,
    requested: list[str],
    records: dict[str, list[dict[str, Any]]],
    torch: Any,
) -> dict[str, Any]:
    assert runtime.base is not None
    runtime.registry.unload(all_facets=True)
    clean_gpu(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if requested:
        plan, loaded = runtime.registry.load_resolved(requested, runtime.base)
    else:
        plan, loaded = CompositionPlan(requested=(), resolved=(), blocks=()), {}
    active_parameters = sum(
        parameter.numel()
        for facet in loaded.values()
        for parameter in facet.parameters()
    )
    loss = evaluate_loss(runtime, plan, loaded, records, torch)
    generated: dict[str, list[dict[str, Any]]] = {}
    for facet, items in records.items():
        generated[facet] = []
        for record in items:
            response = runtime.generate(
                {
                    "messages": prompt_messages(record),
                    "facets": requested,
                    "generation": {"max_new_tokens": 256, "do_sample": False},
                    "seed": 42,
                }
            )
            text = str(response["output"]["text"])
            generated[facet].append(
                {
                    "id": record.get("id"),
                    "language": (record.get("coverage") or {}).get("language"),
                    "output": text,
                    "reference": target_text(record),
                    "scores": score_generation(facet, text, target_text(record)),
                    "usage": response["usage"],
                    "timingMs": response["timingMs"],
                }
            )
    aggregates = {facet: aggregate_examples(items) for facet, items in generated.items()}
    all_examples = [item for items in generated.values() for item in items]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "state": state,
        "requestedFacets": requested,
        "resolvedFacets": list(plan.resolved),
        "expandedBlockCount": len(plan.blocks),
        "activeFacetParameters": active_parameters,
        "loss": loss,
        "generation": {"aggregate": {**aggregates, "all": aggregate_examples(all_examples)}, "examples": generated},
        "cudaAllocatedBytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
        "peakCudaAllocatedBytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }


def evaluate_family(family: str, spec: dict[str, Any], torch: Any) -> dict[str, Any]:
    runtime = MixedInferenceRuntime(spec["registry"])
    base_summary = runtime.load_base(
        {
            "id": str(Path(spec["base"]).resolve()),
            "revision": None,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "allowDownload": False,
        }
    )
    assert runtime.base is not None
    base_parameters = sum(parameter.numel() for parameter in runtime.base.model.parameters())
    records = heldout_records()
    states = []
    for state, requested in STATES:
        print(f"[{family}/evaluation] state={state}", flush=True)
        states.append(evaluate_state(runtime, state, requested, records, torch))
    runtime.unload_base()
    clean_gpu(torch)
    return {
        "base": base_summary,
        "baseParameters": base_parameters,
        "holdoutPolicy": "last two SFT records per facet; never included in training configs",
        "states": states,
    }


def hardware(torch: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        value["cuda"] = {
            "torchCuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "totalMemoryBytes": int(props.total_memory),
            "computeCapability": f"{props.major}.{props.minor}",
            "bfloat16Supported": bool(torch.cuda.is_bf16_supported()),
        }
    return value


def run_family(family: str, *, do_training: bool, do_evaluation: bool, torch: Any) -> dict[str, Any]:
    spec = MODELS[family]
    model_file = Path(spec["base"]) / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_file}")
    result: dict[str, Any] = {
        "schemaVersion": "mi-model-family-benchmark-v1",
        "benchmark": "model-family-smoke",
        "family": family,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(torch),
        "baseModelDirectory": str(Path(spec["base"]).resolve()),
        "baseModelBytes": model_file.stat().st_size,
        "facetVersion": VERSION,
        "trainingData": {
            "continuedPretrainingPerFacet": 10,
            "sftTrainingPerFacet": 4,
            "heldoutSftPerFacet": 2,
            "warning": "integration smoke benchmark; dataset is too small for quality claims",
        },
    }
    output_path = RESULTS / f"{family}.json"
    if do_training:
        result["training"] = train_family(family, spec, torch)
        atomic_json(output_path, result)
    elif output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("training"):
            result["training"] = previous["training"]
    if do_evaluation:
        result["evaluation"] = evaluate_family(family, spec, torch)
    atomic_json(output_path, result)
    return result


def write_summary() -> None:
    available = []
    for family in MODELS:
        path = RESULTS / f"{family}.json"
        if path.exists():
            available.append(json.loads(path.read_text(encoding="utf-8")))
    atomic_json(
        RESULTS / "summary.json",
        {
            "schemaVersion": "mi-model-family-benchmark-summary-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "families": available,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and compare MI facets on Qwen, Gemma and Llama.")
    parser.add_argument("--model", action="append", choices=tuple(MODELS), help="Family to run; repeatable. Default: all.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.skip_training and args.skip_evaluation:
        parser.error("At least one of training or evaluation must remain enabled")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark configuration requires CUDA; use the licenta conda environment")
    families = args.model or list(MODELS)
    for family in families:
        print(f"[benchmark] family={family}", flush=True)
        run_family(
            family,
            do_training=not args.skip_training,
            do_evaluation=not args.skip_evaluation,
            torch=torch,
        )
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
