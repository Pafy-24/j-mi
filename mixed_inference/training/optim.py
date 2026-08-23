from __future__ import annotations

import math
from typing import Any, Iterable

from ..core.errors import ConfigurationError
from ..core.util import import_symbol
from .config import OptimizerConfig, SchedulerConfig


def build_optimizer(config: OptimizerConfig, parameters: Iterable[Any]) -> Any:
    import torch

    params = list(parameters)
    if not params:
        raise ConfigurationError("No trainable facet parameters were found")
    common = {"lr": config.learning_rate, "weight_decay": config.weight_decay, **config.kwargs}
    if config.class_path:
        optimizer_class = import_symbol(config.class_path)
        return optimizer_class(params, **common)
    if config.type == "adamw":
        return torch.optim.AdamW(params, betas=config.betas, eps=config.eps, **common)
    if config.type == "adam":
        return torch.optim.Adam(params, betas=config.betas, eps=config.eps, **common)
    if config.type == "sgd":
        return torch.optim.SGD(params, momentum=config.momentum, **common)
    if config.type == "adagrad":
        return torch.optim.Adagrad(params, eps=config.eps, **common)
    if config.type == "rmsprop":
        return torch.optim.RMSprop(params, momentum=config.momentum, eps=config.eps, **common)
    raise ConfigurationError(
        f"Unsupported optimizer.type {config.type!r}; use adamw, adam, sgd, adagrad, rmsprop or classPath"
    )


def build_scheduler(
    config: SchedulerConfig,
    optimizer: Any,
    total_steps: int,
) -> tuple[Any, int]:
    import torch

    warmup = config.warmup_steps
    if warmup <= 0 and config.warmup_ratio > 0:
        warmup = int(total_steps * config.warmup_ratio)
    warmup = min(max(warmup, 0), max(total_steps - 1, 0))
    if config.class_path:
        scheduler_class = import_symbol(config.class_path)
        return scheduler_class(optimizer, **config.kwargs), warmup

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        if config.type == "constant":
            return 1.0
        if config.type == "linear":
            return max(0.0, 1.0 - progress)
        if config.type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ConfigurationError(
            f"Unsupported scheduler.type {config.type!r}; use constant, linear, cosine or classPath"
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup

