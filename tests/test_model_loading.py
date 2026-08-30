from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mixed_inference.core.modeling import load_base_model
from mixed_inference.core.specs import BaseModelRequest


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


class _Config:
    is_encoder_decoder = False
    model_type = "test"
    num_hidden_layers = 1

    @staticmethod
    def to_dict() -> dict[str, object]:
        return {"model_type": "test", "num_hidden_layers": 1}


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.config = _Config()
        self.moved_to: torch.device | None = None

    def to(self, device: torch.device) -> "_Model":
        self.moved_to = device
        return self


def test_cuda_load_materializes_directly_on_the_rm_device() -> None:
    model = _Model()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    request = BaseModelRequest.from_dict(
        {
            "id": "portable-model",
            "device": "managed",
            "dtype": "bfloat16",
            "allowDownload": False,
        }
    )
    assigned = torch.device("cuda:0")

    with (
        patch("mixed_inference.core.modeling.managed_device", return_value="cuda:0"),
        patch("mixed_inference.core.modeling.apply_memory_budget"),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as loader,
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        loaded = load_base_model(request)

    assert loader.call_args.kwargs["device_map"] == {"": assigned}
    assert loader.call_args.kwargs["dtype"] is torch.bfloat16
    assert model.moved_to == assigned
    assert loaded.device == assigned


def test_cuda_host_staged_load_avoids_accelerate_device_map() -> None:
    model = _Model()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    request = BaseModelRequest.from_dict(
        {
            "id": "portable-model",
            "device": "managed",
            "dtype": "bfloat16",
            "allowDownload": False,
        }
    )
    assigned = torch.device("cuda:0")

    with (
        patch.dict(
            os.environ,
            {"JESTER_MI_CUDA_LOAD_STRATEGY": "host-staged"},
        ),
        patch("mixed_inference.core.modeling.managed_device", return_value="cuda:0"),
        patch("mixed_inference.core.modeling.apply_memory_budget"),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as loader,
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        loaded = load_base_model(request)

    assert "device_map" not in loader.call_args.kwargs
    assert loader.call_args.kwargs["dtype"] is torch.bfloat16
    assert model.moved_to == assigned
    assert loaded.device == assigned
