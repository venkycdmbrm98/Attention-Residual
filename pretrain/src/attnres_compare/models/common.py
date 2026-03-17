from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, kind: str = "swiglu", bias: bool = False) -> None:
        super().__init__()
        self.kind = kind
        if kind == "none" or d_ff == 0:
            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None
        elif kind == "swiglu":
            self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
            self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
            self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        elif kind == "gelu":
            self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        else:
            raise ValueError(f"Unsupported MLP kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none" or self.down_proj is None and getattr(self, "fc2", None) is None:
            return torch.zeros_like(x)
        if self.kind == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    aux: dict[str, Any] | None = None
