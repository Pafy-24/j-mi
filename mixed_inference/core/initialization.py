from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass
from typing import Any

from .errors import CompatibilityError, ConfigurationError
from .specs import BlockSpec


DEFAULT_IDENTITY_TARGETS = (
    "*self_attn.o_proj*",
    "*self_attn.out_proj*",
    "*self_attn.dense*",
    "*self_attention.dense*",
    "*attention.output.dense*",
    "*attention.dense*",
    "*attn.c_proj*",
    "*mlp.down_proj*",
    "*mlp.dense_4h_to_h*",
    "*mlp.fc2*",
    "*mlp.c_proj*",
    "*feed_forward.w2*",
)


@dataclass(frozen=True)
class InitializationReport:
    initializer: str
    zeroed_parameters: tuple[str, ...]


def initialize_expansion_block(source_layer: Any, spec: BlockSpec) -> tuple[Any, InitializationReport]:
    block = copy.deepcopy(source_layer)
    if spec.initializer == "copy":
        return block, InitializationReport(initializer="copy", zeroed_parameters=())
    if spec.initializer == "zero":
        names: list[str] = []
        for name, parameter in block.named_parameters():
            parameter.data.zero_()
            names.append(name)
        return block, InitializationReport(initializer="zero", zeroed_parameters=tuple(names))
    if spec.initializer == "reset":
        reset_count = 0
        for module in block.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset) and not list(module.children()):
                reset()
                reset_count += 1
        if reset_count == 0:
            raise CompatibilityError(
                f"Block class {type(block).__name__} has no resettable leaf modules"
            )
        return block, InitializationReport(initializer="reset", zeroed_parameters=())
    if spec.initializer != "identity-copy":
        raise ConfigurationError(f"Unknown block initializer: {spec.initializer}")

    patterns = spec.identity_targets or DEFAULT_IDENTITY_TARGETS
    matched: list[str] = []
    for name, parameter in block.named_parameters():
        if any(_matches(name, pattern) for pattern in patterns):
            parameter.data.zero_()
            matched.append(name)
    if not matched:
        raise CompatibilityError(
            f"Identity initialization found no output projections in {type(block).__name__}. "
            "Set identityTargets explicitly for this architecture."
        )
    return block, InitializationReport(
        initializer="identity-copy",
        zeroed_parameters=tuple(matched),
    )


def _matches(parameter_name: str, pattern: str) -> bool:
    if any(marker in pattern for marker in "*?["):
        return fnmatch.fnmatchcase(parameter_name, pattern)
    return parameter_name == pattern or parameter_name.startswith(pattern + ".")
