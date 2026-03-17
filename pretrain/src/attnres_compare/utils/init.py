from __future__ import annotations

import hashlib
import math
from typing import Iterable

import torch
import torch.nn as nn

from attnres_compare.config import ModelConfig


def _seed_from_name(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _normal_(tensor: torch.Tensor, std: float, seed: int) -> None:
    generator = torch.Generator(device=tensor.device.type if tensor.is_cuda else "cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std, generator=generator)


@torch.no_grad()
def initialize_model(model: nn.Module, cfg: ModelConfig, seed: int) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        seed_i = _seed_from_name(seed, name)
        if name.endswith("bias"):
            param.zero_()
        elif "query" in name:
            param.zero_()
        elif name.endswith("norm.weight") or ".norm.weight" in name or "_norm.weight" in name:
            param.fill_(1.0)
        elif name.endswith("attn_norm.weight") or name.endswith("mlp_norm.weight"):
            param.fill_(1.0)
        elif name.endswith("final_norm.weight"):
            param.fill_(1.0)
        elif name.endswith("pos_emb.weight") or name.endswith("tok_emb.weight"):
            _normal_(param, cfg.init_std, seed_i)
        elif name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
            std = cfg.init_std
            if cfg.init_resid_scale:
                std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
            _normal_(param, std, seed_i)
        elif param.ndim >= 2:
            _normal_(param, cfg.init_std, seed_i)
        else:
            param.fill_(1.0)


def shared_parameter_fingerprint(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        if any(tag in name for tag in ["pre_attn_mixer", "pre_mlp_mixer", "output_mixer"]):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()
