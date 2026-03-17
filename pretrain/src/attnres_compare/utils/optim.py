from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from attnres_compare.config import ModelConfig, OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("bias") or "norm" in name or "query" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=torch.cuda.is_available(),
    )


@dataclass
class LRScheduler:
    optimizer: torch.optim.Optimizer
    cfg: OptimConfig

    def get_lr(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * step / max(1, self.cfg.warmup_steps)
        if self.cfg.scheduler == "cosine":
            denom = max(1, self.cfg.max_steps - self.cfg.warmup_steps)
            progress = min(1.0, (step - self.cfg.warmup_steps) / denom)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.cfg.min_lr + coeff * (self.cfg.lr - self.cfg.min_lr)
        if self.cfg.scheduler == "wsd":
            stable_end = self.cfg.warmup_steps + self.cfg.stable_steps
            if step < stable_end:
                return self.cfg.lr
            decay_steps = self.cfg.decay_steps or max(1, self.cfg.max_steps - stable_end)
            progress = min(1.0, (step - stable_end) / decay_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.cfg.min_lr + coeff * (self.cfg.lr - self.cfg.min_lr)
        raise ValueError(f"Unsupported scheduler: {self.cfg.scheduler}")

    def step(self, step: int) -> float:
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
