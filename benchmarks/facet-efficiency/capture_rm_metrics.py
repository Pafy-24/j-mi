from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


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
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _state(run: dict[str, Any], state_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in run.get("states", []) if item.get("id") == state_id),
        None,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        value = sample.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        instance_id = str(sample.get("instanceId") or "unknown")
        metric = str(sample.get("metric") or "unknown")
        grouped.setdefault((instance_id, metric), []).append(sample)

    instances: dict[str, dict[str, Any]] = {}
    for (instance_id, metric), records in sorted(grouped.items()):
        records.sort(key=lambda item: str(item.get("sampledAt") or ""))
        values = sorted(float(item["value"]) for item in records)
        chronological = [float(item["value"]) for item in records]
        instances.setdefault(instance_id, {})[metric] = {
            "count": len(values),
            "unit": records[0].get("unit"),
            "minimum": values[0],
            "maximum": values[-1],
            "mean": statistics.fmean(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "first": chronological[0],
            "last": chronological[-1],
            "delta": chronological[-1] - chronological[0],
            "sources": sorted(
                {str(item.get("source")) for item in records if item.get("source")}
            ),
            "confidence": sorted(
                {
                    str(item.get("confidence"))
                    for item in records
                    if item.get("confidence")
                }
            ),
        }

    sampled_at = sorted(
        str(item.get("sampledAt")) for item in samples if item.get("sampledAt")
    )
    return {
        "sampleCount": len(samples),
        "firstSampleAt": sampled_at[0] if sampled_at else None,
        "lastSampleAt": sampled_at[-1] if sampled_at else None,
        "instances": instances,
    }


def _iso_timestamp(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _request_json(url: str, token: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return value


def _config_for_run(run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    snapshot = run.get("configSnapshot")
    if isinstance(snapshot, str) and (run_dir / snapshot).is_file():
        return _read_json(run_dir / snapshot)
    candidates = sorted(run_dir.glob("experiment-config*.json"))
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify experiment config in {run_dir}")
    return _read_json(candidates[0])


def capture(
    manifest_path: Path,
    *,
    rm_url: str,
    token: str,
    limit: int,
    timeout: float,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    base = manifest_path.parent.resolve()
    captured: list[dict[str, Any]] = []
    for run_entry in manifest.get("runs", []):
        states: list[dict[str, Any]] = []
        for state_id in STATE_ORDER:
            source_raw = run_entry.get("stateSources", {}).get(
                state_id, run_entry["run"]
            )
            run_dir = _resolve(base, source_raw)
            run_path = run_dir / "run.json"
            if not run_path.is_file():
                states.append({"state": state_id, "status": "not-measured"})
                continue
            run = _read_json(run_path)
            state = _state(run, state_id)
            if state is None or not state.get("startedAtUnix") or not state.get(
                "finishedAtUnix"
            ):
                states.append({"state": state_id, "status": "not-measured"})
                continue
            config = _config_for_run(run_dir, run)
            topology_id = config.get("rm", {}).get("topologyId")
            if not topology_id:
                states.append(
                    {
                        "state": state_id,
                        "status": "unavailable",
                        "error": "topologyId missing from experiment config",
                    }
                )
                continue
            since = _iso_timestamp(float(state["startedAtUnix"]))
            until = _iso_timestamp(float(state["finishedAtUnix"]))
            query = urlencode({"limit": limit, "since": since, "until": until})
            url = (
                rm_url.rstrip("/")
                + f"/v1/topologies/{quote(str(topology_id), safe='')}/metrics?{query}"
            )
            try:
                response = _request_json(url, token, timeout)
                samples = response.get("samples") or []
                summary = _summarize_samples(samples)
                states.append(
                    {
                        "state": state_id,
                        "status": "captured" if samples else "unavailable",
                        "sourceRun": run_dir.name,
                        "topologyId": topology_id,
                        "interval": {"since": since, "until": until},
                        "truncated": len(samples) >= limit,
                        "metrics": summary,
                    }
                )
            except Exception as exc:  # evidence captures retain per-state failures
                states.append(
                    {
                        "state": state_id,
                        "status": "unavailable",
                        "sourceRun": run_dir.name,
                        "topologyId": topology_id,
                        "interval": {"since": since, "until": until},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        captured.append(
            {
                "id": run_entry["id"],
                "model": run_entry["model"],
                "inferenceHardware": run_entry["inferenceHardware"],
                "states": states,
            }
        )
    return {
        "schemaVersion": "jester-rm-benchmark-evidence-v1",
        "manifest": manifest_path.name,
        "runs": captured,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RM metrics by benchmark state")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rm-url", default="http://127.0.0.1:8790")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100_000:
        parser.error("--limit must be between 1 and 100000")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        parser.error("token file is empty")
    result = capture(
        args.manifest.resolve(),
        rm_url=args.rm_url,
        token=token,
        limit=args.limit,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
