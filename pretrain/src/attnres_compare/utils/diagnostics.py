from __future__ import annotations

import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def global_grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach().float().pow(2).sum().item()
        total += value
    return total ** 0.5


def block_grad_norms(model: nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    if not hasattr(model, "blocks"):
        return norms
    blocks = getattr(model, "blocks")
    for idx, block in enumerate(blocks):
        total = 0.0
        for param in block.parameters():
            if param.grad is None:
                continue
            total += param.grad.detach().float().pow(2).sum().item()
        norms[f"grad/block_{idx}"] = total ** 0.5
    return norms


def tensor_hist_stats(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}/mean": float(arr.mean()),
        f"{prefix}/std": float(arr.std()),
        f"{prefix}/min": float(arr.min()),
        f"{prefix}/max": float(arr.max()),
    }


def make_heatmap_image(matrix: list[list[float]], title: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Source Index")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    import PIL.Image

    image = np.array(PIL.Image.open(buffer))
    return image
