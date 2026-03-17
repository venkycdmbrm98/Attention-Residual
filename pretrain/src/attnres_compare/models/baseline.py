from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from attnres_compare.config import ModelConfig
from attnres_compare.models.common import CausalSelfAttention, ForwardOutput, MLP, RMSNorm, tensor_rms


class BaselineBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, bias=cfg.attn_bias, dropout=cfg.dropout)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, kind=cfg.mlp_type, bias=cfg.attn_bias)

    def forward(self, x: torch.Tensor, collect_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if collect_aux:
            aux["pre_attn_rms"] = tensor_rms(x)
        attn_out = self.attn(self.attn_norm(x))
        if collect_aux:
            aux["attn_out_rms"] = tensor_rms(attn_out)
        x = x + attn_out
        if collect_aux:
            aux["pre_mlp_rms"] = tensor_rms(x)
        mlp_out = self.mlp(self.mlp_norm(x))
        if collect_aux:
            aux["mlp_out_rms"] = tensor_rms(mlp_out)
        x = x + mlp_out
        if collect_aux:
            aux["post_block_rms"] = tensor_rms(x)
        return x, aux


class BaselineGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([BaselineBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> ForwardOutput:
        bsz, seqlen = input_ids.shape
        if seqlen > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds model max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seqlen, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]

        aux: dict[str, Any] | None = None
        if return_aux:
            aux = {
                "pre_attn_rms": [],
                "pre_mlp_rms": [],
                "attn_out_rms": [],
                "mlp_out_rms": [],
                "post_block_rms": [],
            }

        for block in self.blocks:
            x, block_aux = block(x, collect_aux=return_aux)
            if return_aux and aux is not None:
                for key in aux:
                    aux[key].append(float(block_aux[key].detach().cpu().item()))

        if return_aux and aux is not None:
            aux["final_resid_rms"] = float(tensor_rms(x).detach().cpu().item())
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return ForwardOutput(logits=logits, loss=loss, aux=aux)
