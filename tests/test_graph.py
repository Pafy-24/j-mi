from __future__ import annotations

import unittest

from mixed_inference.core.errors import CompositionError
from mixed_inference.core.graph import CompositionResolver
from mixed_inference.core.specs import (
    BaseBinding,
    BlockSpec,
    DependencySpec,
    FacetManifest,
    SlotSpec,
)


BASE = BaseBinding("base", None, "hash", "model.layers", 4, "test")


def facet(
    name: str,
    *,
    requires: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
    priority: int = 0,
    slot_id: str = "shared-slot",
    slot_order: int = 0,
) -> FacetManifest:
    return FacetManifest(
        facet_id=name,
        version="1",
        base=BASE,
        slots=(
            SlotSpec(
                slot_id=slot_id,
                after_layer=1,
                order=slot_order,
                blocks=(BlockSpec(source_layer=1),),
            ),
        ),
        dependencies=DependencySpec(
            requires=requires,
            conflicts=conflicts,
            priority=priority,
        ),
    )


class CompositionResolverTests(unittest.TestCase):
    def test_multi_parent_tree_is_dependency_ordered(self) -> None:
        manifests = {
            item.facet_id: item
            for item in (
                facet("reasoning", slot_id="reasoning-slot", slot_order=100),
                facet("coder", slot_id="coder-slot", slot_order=100),
                facet(
                    "python",
                    requires=("reasoning@1", "coder@1"),
                    slot_id="python-slot",
                    slot_order=-100,
                ),
            )
        }
        plan, _ = CompositionResolver(lambda ref: manifests[ref.facet_id]).resolve(["python@1"])

        self.assertEqual(plan.resolved[-1], "python@1")
        self.assertEqual([block.facet for block in plan.blocks][-1], "python")
        self.assertEqual(set(plan.resolved[:2]), {"reasoning@1", "coder@1"})

    def test_conflicts_are_rejected(self) -> None:
        manifests = {
            "safe": facet("safe", conflicts=("unsafe",)),
            "unsafe": facet("unsafe"),
        }
        with self.assertRaises(CompositionError):
            CompositionResolver(lambda ref: manifests[ref.facet_id]).resolve(
                ["safe@1", "unsafe@1"]
            )

    def test_two_versions_of_one_facet_are_rejected(self) -> None:
        one = facet("coder")
        two = FacetManifest(**{**one.__dict__, "version": "2"})
        manifests = {"1": one, "2": two}
        with self.assertRaises(CompositionError):
            CompositionResolver(lambda ref: manifests[ref.version or "2"]).resolve(
                ["coder@1", "coder@2"]
            )


if __name__ == "__main__":
    unittest.main()
