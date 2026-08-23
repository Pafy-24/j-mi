from __future__ import annotations

import unittest

import torch

from mixed_inference.core.graph import CompositionPlan, PlannedBlock
from mixed_inference.core.initialization import initialize_expansion_block
from mixed_inference.core.modeling import ActivatedComposition, LayerStackHandle, LoadedBaseModel
from mixed_inference.core.specs import BaseBinding, BaseModelRequest, BlockSpec


class TinyBlock(torch.nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.layer_idx = index
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.self_attn.o_proj(hidden) + self.mlp.down_proj(hidden)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(0), TinyBlock(1)])
        self.config = type("Config", (), {"num_hidden_layers": 2})()


class CompositionTests(unittest.TestCase):
    def test_identity_copy_and_reversible_activation(self) -> None:
        model = TinyModel()
        block, report = initialize_expansion_block(
            model.layers[0], BlockSpec(source_layer=0, initializer="identity-copy")
        )
        self.assertIn("self_attn.o_proj.weight", report.zeroed_parameters)
        self.assertIn("mlp.down_proj.weight", report.zeroed_parameters)
        hidden = torch.randn(2, 4)
        self.assertTrue(torch.equal(block(hidden), hidden))

        request = BaseModelRequest(model_id="tiny")
        binding = BaseBinding("tiny", None, "hash", "layers", 2, "tiny")
        base = LoadedBaseModel(
            request=request,
            model=model,
            tokenizer=None,
            stack=LayerStackHandle("layers", model, "layers", model.layers),
            binding=binding,
            device=torch.device("cpu"),
        )
        plan = CompositionPlan(
            requested=("facet@1",),
            resolved=("facet@1",),
            blocks=(PlannedBlock("facet", "1", "slot", 0, 0, 0, 0),),
        )
        original = model.layers
        with ActivatedComposition(base, plan, lambda _: block):
            self.assertEqual(len(model.layers), 3)
            self.assertEqual(model.config.num_hidden_layers, 3)
            self.assertEqual([item.layer_idx for item in model.layers], [0, 1, 2])
        self.assertIs(model.layers, original)
        self.assertEqual(model.config.num_hidden_layers, 2)
        self.assertEqual([item.layer_idx for item in model.layers], [0, 1])


if __name__ == "__main__":
    unittest.main()
