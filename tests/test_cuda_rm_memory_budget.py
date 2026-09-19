from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from jester_rm_client import current_lease
from mixed_inference.rm_integration import apply_memory_budget, managed_device


def test_managed_cuda_binding_allocates_on_visible_gpu(tmp_path: Path, monkeypatch) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        pytest.skip("NVIDIA driver is required")
    uuid = subprocess.check_output(
        [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).splitlines()[0].strip()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch and a GPU are required")
    instance = {"metadata": {"id": "cuda-smoke"}}
    config_hash = hashlib.sha256(
        json.dumps(instance, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    effective = tmp_path / "config" / "effective.json"
    effective.parent.mkdir()
    effective.write_text(
        json.dumps(
            {
                "schemaVersion": "jester-rm-effective-v1",
                "instance": instance,
                "lease": {
                    "apiVersion": "jester.rm/v1",
                    "kind": "Lease",
                    "leaseId": "cuda-lease",
                    "instanceId": "cuda-smoke",
                    "hostId": "hardware-host",
                    "state": "active",
                    "configSha256": config_hash,
                    "telemetryToken": "test-token",
                    "bindings": [
                        {
                            "resourceId": f"gpu-nvidia-{uuid.lower()}",
                            "runtime": {
                                "kind": "gpu",
                                "backend": "torch.cuda",
                                "cudaUuid": uuid,
                                "device": "cuda:0",
                                "memoryBudgetBytes": 512 * 1024**2,
                            },
                        }
                    ],
                    "paths": {"run": str(tmp_path / "run")},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JESTER_RM_REQUIRED", "1")
    monkeypatch.setenv("JESTER_RM_LEASE_FILE", str(effective))
    monkeypatch.setenv("JESTER_RM_LEASE_ID", "cuda-lease")
    monkeypatch.setenv("JESTER_RM_INSTANCE_ID", "cuda-smoke")
    monkeypatch.setenv("JESTER_RM_HOST_ID", "hardware-host")
    monkeypatch.setenv("JESTER_RM_DEVICE_BACKEND", "torch.cuda")
    monkeypatch.setenv("JESTER_RM_DEVICE_ID", f"gpu-nvidia-{uuid.lower()}")
    current_lease(required=True, refresh=True)
    device = torch.device(managed_device("managed"))
    apply_torch_memory_budget(torch, device)
    tensor = torch.ones((256, 256), device=device)
    assert tensor.device.type == "cuda"
    del tensor
    torch.cuda.empty_cache()
