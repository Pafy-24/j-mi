from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


SCHEMA_VERSION = "jester-controlled-training-evidence-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _request_json(base: str, token: str, path: str) -> dict[str, Any]:
    request = Request(
        base.rstrip("/") + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object from Trainer endpoint {path}")
    return value


def compile_campaign(
    entry: dict[str, Any],
    campaign: dict[str, Any],
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metrics_snapshot: dict[str, Any],
) -> dict[str, Any]:
    succeeded_jobs = [item for item in jobs if item.get("state") == "succeeded"]
    result = results[0] if len(results) == 1 else None
    metrics = (result or {}).get("metrics") or {}
    job_id = str((result or {}).get("jobId") or "")
    job_events = [item for item in events if str(item.get("jobId") or "") == job_id]
    last_failed_sequence = max(
        (
            int(item.get("sequence", 0))
            for item in job_events
            if item.get("type") == "JOB_ATTEMPT_FAILED"
        ),
        default=-1,
    )
    committed = [
        item
        for item in job_events
        if item.get("type") == "ROUND_COMMITTED"
        and int(item.get("sequence", 0)) > last_failed_sequence
    ]
    committed.sort(key=lambda item: int(item.get("sequence", 0)))
    expected_rounds = int(entry.get("expectedRounds", 0))
    observed_rounds = int(metrics.get("rounds", 0))
    measured_workers = list(metrics.get("cohort") or [])
    resources = metrics_snapshot.get("resources") or {}
    topology_measured = resources.get("topologyMeasured") or {}
    channels = resources.get("channels") or []
    complete = (
        campaign.get("state") == "succeeded"
        and len(succeeded_jobs) == len(jobs) == 1
        and result is not None
        and observed_rounds == expected_rounds
        and len(measured_workers) == int(entry.get("workerCount", 0))
    )
    return {
        "id": entry["id"],
        "campaignId": entry["campaignId"],
        "hardware": entry["hardware"],
        "workerCount": len(measured_workers) or entry.get("workerCount"),
        "workers": measured_workers,
        "rounds": observed_rounds,
        "expectedRounds": expected_rounds,
        "supervisedTokens": metrics.get("supervisedTokens"),
        "losses": [item.get("data", {}).get("loss") for item in committed],
        "roundSeconds": [
            item.get("data", {}).get("durationSeconds") for item in committed
        ],
        "gradientNorms": [
            item.get("data", {}).get("gradientNorm") for item in committed
        ],
        "meanTrainingLoss": metrics.get("meanTrainingLoss"),
        "lastTrainingLoss": metrics.get("lastTrainingLoss"),
        "elapsedSeconds": metrics.get("elapsedSeconds"),
        "tokensPerSecond": metrics.get("tokensPerSecond"),
        "meanRoundSeconds": metrics.get("meanRoundSeconds"),
        "workerWaitSeconds": metrics.get("workerWaitSeconds"),
        "artifact": {
            "facetReference": (result or {}).get("facetReference"),
            "sha256": (result or {}).get("artifactSha256"),
            "digest": (result or {}).get("artifactDigest"),
        },
        "campaignState": campaign.get("state"),
        "jobStates": {str(item.get("jobId")): item.get("state") for item in jobs},
        "resources": {
            "capturedAt": "post-campaign",
            "cpuPercent": topology_measured.get("cpuPercent"),
            "ramBytes": topology_measured.get("ramBytes"),
            "acceleratorMemoryBytes": topology_measured.get(
                "acceleratorMemoryBytes"
            ),
            "channelBytesSent": sum(
                int((item.get("counters") or {}).get("bytesSent", 0))
                for item in channels
            ),
            "channelBytesReceived": sum(
                int((item.get("counters") or {}).get("bytesReceived", 0))
                for item in channels
            ),
            "source": resources.get("source"),
            "trainerIncludedByRM": resources.get("trainerIncludedByRM"),
        },
        "resourceSnapshot": metrics_snapshot.get("resources"),
        "status": "measured" if complete else "partial",
    }


def capture(
    manifest: dict[str, Any],
    *,
    base: str,
    token: str,
    campaign_ids: set[str] | None = None,
) -> dict[str, Any]:
    if manifest.get("schemaVersion") != "jester-training-evidence-manifest-v1":
        raise ValueError("Unsupported controlled-training manifest schemaVersion")
    runs = []
    for entry in manifest.get("campaigns", []):
        campaign_id = str(entry["campaignId"])
        if campaign_ids and campaign_id not in campaign_ids:
            continue
        encoded = quote(campaign_id, safe="")
        campaign = _request_json(base, token, f"/v1/campaigns/{encoded}")
        jobs = _request_json(base, token, f"/v1/campaigns/{encoded}/jobs").get(
            "items", []
        )
        results = _request_json(
            base, token, f"/v1/campaigns/{encoded}/results"
        ).get("items", [])
        events = _request_json(
            base, token, f"/v1/campaigns/{encoded}/events?afterSequence=0"
        ).get("items", [])
        metrics = _request_json(base, token, f"/v1/campaigns/{encoded}/metrics")
        runs.append(
            compile_campaign(entry, campaign, jobs, results, events, metrics)
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceManifest": manifest.get("name"),
        "baselineId": manifest.get("baselineId"),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture controlled Trainer campaign evidence through its REST API"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--trainer-url", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--campaign-id", action="append", dest="campaign_ids")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        parser.error("Trainer token file is empty")
    evidence = capture(
        _read_json(args.manifest.resolve()),
        base=args.trainer_url,
        token=token,
        campaign_ids=set(args.campaign_ids) if args.campaign_ids else None,
    )
    if args.merge and args.output.is_file():
        previous = _read_json(args.output)
        if previous.get("schemaVersion") != SCHEMA_VERSION:
            parser.error("existing output has an incompatible schemaVersion")
        by_id = {str(item["id"]): item for item in previous.get("runs", [])}
        by_id.update({str(item["id"]): item for item in evidence["runs"]})
        evidence["runs"] = [by_id[key] for key in sorted(by_id)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
