from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..core.errors import ConfigurationError
from .config import DataConfig, DatasetSource


@dataclass(frozen=True)
class TokenizedRecord:
    input_ids: list[int]
    labels: list[int]
    record_id: str


class FacetTokenDataset:
    def __init__(self, records: list[TokenizedRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "input_ids": record.input_ids,
            "labels": record.labels,
            "record_id": record.record_id,
        }


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        maximum = max(len(item["input_ids"]) for item in items)
        input_ids = []
        labels = []
        attention = []
        for item in items:
            padding = maximum - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def build_datasets(tokenizer: Any, config: DataConfig) -> tuple[FacetTokenDataset, FacetTokenDataset | None]:
    train = tokenize_sources(tokenizer, config.train, config)
    validation = tokenize_sources(tokenizer, config.validation, config) if config.validation else []
    if not train:
        raise ConfigurationError("No training examples remain after stage filtering/tokenization")
    return FacetTokenDataset(train), FacetTokenDataset(validation) if validation else None


def tokenize_sources(
    tokenizer: Any,
    sources: Iterable[DatasetSource],
    config: DataConfig,
) -> list[TokenizedRecord]:
    tokenized: list[TokenizedRecord] = []
    for source in sources:
        raw_records = list(read_records(source.path))
        if source.stages:
            raw_records = [item for item in raw_records if str(item.get("stage", "")) in source.stages]
        if source.limit is not None:
            raw_records = raw_records[: source.limit]
        source_records = [tokenize_record(tokenizer, item, config) for item in raw_records]
        for _ in range(source.repeat):
            tokenized.extend(source_records)
    if config.packing:
        tokenized = pack_records(tokenized, config.max_sequence_length)
    return tokenized


def read_records(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise ConfigurationError(f"Dataset does not exist: {source}")
    if source.is_dir():
        files = sorted(
            item
            for item in source.iterdir()
            if item.suffix.lower() in {".json", ".jsonl"}
            and item.name.lower()
            not in {
                "test.json",
                "test.jsonl",
                "facet.json",
                "facet.mi.json",
                "manifest.json",
                "training-result.json",
            }
            and not item.name.lower().endswith(".schema.json")
        )
        for item in files:
            yield from read_records(item)
        return
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(
                        f"Invalid JSONL in {source}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ConfigurationError(f"Dataset item {source}:{line_number} is not an object")
                yield value
        return
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {source}: {exc.msg}") from exc
    if isinstance(value, dict):
        for key in ("records", "items", "data"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            yield value
            return
    if not isinstance(value, list):
        raise ConfigurationError(f"Dataset root in {source} must be an object or array")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigurationError(f"Dataset item {source}[{index}] is not an object")
        yield item


def tokenize_record(tokenizer: Any, record: dict[str, Any], config: DataConfig) -> TokenizedRecord:
    record_id = str(record.get("id") or record.get("record_id") or "unknown")
    stage = str(record.get("stage", ""))
    if record.get("schemaVersion") == "facet-record-v3":
        if stage == "continued-pretraining":
            content = (record.get("input") or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise ConfigurationError(f"Record {record_id} has no input.content")
            return tokenize_causal_text(tokenizer, content, config.max_sequence_length, record_id)
        return tokenize_facet_sft(tokenizer, record, config, record_id)
    if isinstance(record.get("text"), str):
        return tokenize_causal_text(
            tokenizer, str(record["text"]), config.max_sequence_length, record_id
        )
    if isinstance(record.get("prompt"), str) and isinstance(record.get("completion"), str):
        prompt = str(record["prompt"])
        completion = str(record["completion"])
        return tokenize_prompt_completion(tokenizer, prompt, completion, config, record_id)
    messages = record.get("messages")
    if isinstance(messages, list):
        if config.mask_prompt:
            assistant_indices = [
                index
                for index, message in enumerate(messages)
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if assistant_indices:
                target_start = assistant_indices[-1]
                return tokenize_chat(
                    tokenizer,
                    messages[:target_start],
                    messages[target_start:],
                    config,
                    record_id,
                )
        return tokenize_chat(tokenizer, [], messages, config, record_id)
    raise ConfigurationError(f"Cannot infer training format for record {record_id}")


def tokenize_causal_text(tokenizer: Any, text: str, maximum: int, record_id: str) -> TokenizedRecord:
    encoded = tokenizer(text, truncation=True, max_length=maximum, add_special_tokens=True)
    ids = _ids(encoded)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and len(ids) < maximum and (not ids or ids[-1] != int(eos)):
        ids.append(int(eos))
    if len(ids) < 2:
        raise ConfigurationError(f"Record {record_id} tokenizes to fewer than two tokens")
    return TokenizedRecord(input_ids=ids, labels=list(ids), record_id=record_id)


def tokenize_facet_sft(
    tokenizer: Any,
    record: dict[str, Any],
    config: DataConfig,
    record_id: str,
) -> TokenizedRecord:
    input_data = record.get("input") or {}
    target_data = record.get("target") or {}
    prompt_messages = list(input_data.get("conditioningMessages") or []) + list(
        input_data.get("messages") or []
    )
    target_messages = list(target_data.get("messages") or [])
    if not prompt_messages or not target_messages:
        raise ConfigurationError(
            f"SFT record {record_id} requires input messages and target.messages"
        )
    return tokenize_chat(tokenizer, prompt_messages, target_messages, config, record_id)


def tokenize_chat(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    target_messages: list[dict[str, Any]],
    config: DataConfig,
    record_id: str,
) -> TokenizedRecord:
    all_messages = prompt_messages + target_messages
    if getattr(tokenizer, "chat_template", None):
        full = tokenizer.apply_chat_template(
            all_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        full_ids = _ids(full)[: config.max_sequence_length]
        if config.mask_prompt and prompt_messages:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = _ids(prompt)
            prompt_length = common_prefix_length(full_ids, prompt_ids)
        else:
            prompt_length = 0
    else:
        prompt_text = render_plain_messages(prompt_messages)
        target_text = render_plain_messages(target_messages)
        return tokenize_prompt_completion(tokenizer, prompt_text, target_text, config, record_id)
    labels = list(full_ids)
    for index in range(min(prompt_length, len(labels))):
        labels[index] = -100
    if not any(label != -100 for label in labels):
        raise ConfigurationError(
            f"Record {record_id} has no supervised target tokens after truncation; "
            "increase data.maxSequenceLength"
        )
    return TokenizedRecord(input_ids=full_ids, labels=labels, record_id=record_id)


def tokenize_prompt_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    config: DataConfig,
    record_id: str,
) -> TokenizedRecord:
    prompt_ids = _ids(tokenizer(prompt, add_special_tokens=True))
    completion_ids = _ids(tokenizer(completion, add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not completion_ids or completion_ids[-1] != eos):
        completion_ids.append(int(eos))
    full = (prompt_ids + completion_ids)[: config.max_sequence_length]
    labels = list(full)
    if config.mask_prompt:
        for index in range(min(len(prompt_ids), len(labels))):
            labels[index] = -100
    if not any(label != -100 for label in labels):
        raise ConfigurationError(
            f"Record {record_id} target was truncated; increase data.maxSequenceLength"
        )
    return TokenizedRecord(input_ids=full, labels=labels, record_id=record_id)


def render_plain_messages(messages: list[dict[str, Any]]) -> str:
    return "".join(
        f"{str(message.get('role', 'user')).upper()}: {str(message.get('content', ''))}\n"
        for message in messages
    )


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def pack_records(records: list[TokenizedRecord], maximum: int) -> list[TokenizedRecord]:
    packed: list[TokenizedRecord] = []
    ids: list[int] = []
    labels: list[int] = []
    source_ids: list[str] = []
    for record in records:
        cursor = 0
        while cursor < len(record.input_ids):
            available = maximum - len(ids)
            take = min(available, len(record.input_ids) - cursor)
            ids.extend(record.input_ids[cursor : cursor + take])
            labels.extend(record.labels[cursor : cursor + take])
            source_ids.append(record.record_id)
            cursor += take
            if len(ids) == maximum:
                packed.append(
                    TokenizedRecord(
                        input_ids=ids,
                        labels=labels,
                        record_id="packed:" + ",".join(dict.fromkeys(source_ids)),
                    )
                )
                ids, labels, source_ids = [], [], []
    if len(ids) >= 2:
        packed.append(
            TokenizedRecord(
                input_ids=ids,
                labels=labels,
                record_id="packed:" + ",".join(dict.fromkeys(source_ids)),
            )
        )
    return packed


def _ids(encoded: Any) -> list[int]:
    value = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]
