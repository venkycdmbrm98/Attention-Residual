from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from attnres_compare.config import ModelConfig
from attnres_compare.models.common import CausalSelfAttention, ForwardOutput, MLP, RMSNorm, tensor_rms


class DepthMixer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.key_norm = RMSNorm(dim, eps=eps)

    def forward(self, sources: list[torch.Tensor], return_stats: bool = False) -> tuple[torch.Tensor, dict[str, Any] | None]:
        v = torch.stack(sources, dim=0)  # [N, B, T, D]
        k = self.key_norm(v)
        logits = torch.einsum("d,nbtd->nbt", self.query, k)
        weights = logits.float().softmax(dim=0)
        h = torch.einsum("nbt,nbtd->btd", weights.to(dtype=v.dtype), v)
        if not return_stats:
            return h, None

        mean_weights = weights.mean(dim=(1, 2)).detach().cpu()
        eps = 1e-12
        entropy = float((-(mean_weights * (mean_weights + eps).log()).sum()).item())
        stats = {
            "mean_weights": mean_weights.tolist(),
            "entropy": entropy,
            "embed_weight": float(mean_weights[0].item()),
            "prev_weight": float(mean_weights[-1].item()),
            "max_weight": float(mean_weights.max().item()),
            "query_norm": float(self.query.detach().float().norm().cpu().item()),
            "source_count": int(len(sources)),
        }
        return h, stats


class AttnResBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.pre_attn_mixer = DepthMixer(cfg.d_model, eps=cfg.norm_eps)
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, bias=cfg.attn_bias, dropout=cfg.dropout)
        self.pre_mlp_mixer = DepthMixer(cfg.d_model, eps=cfg.norm_eps)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, kind=cfg.mlp_type, bias=cfg.attn_bias)


class FullAttnResGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([AttnResBlock(cfg) for _ in range(cfg.n_layers)])
        self.output_mixer = DepthMixer(cfg.d_model, eps=cfg.norm_eps)
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _blank_aux(self) -> dict[str, Any]:
        max_sources = 1 + 2 * self.cfg.n_layers
        return {
            "pre_attn_rms": [],
            "pre_mlp_rms": [],
            "attn_out_rms": [],
            "mlp_out_rms": [],
            "pre_attn_entropy": [],
            "pre_mlp_entropy": [],
            "pre_attn_embed_weight": [],
            "pre_mlp_embed_weight": [],
            "pre_attn_prev_weight": [],
            "pre_mlp_prev_weight": [],
            "pre_attn_query_norm": [],
            "pre_mlp_query_norm": [],
            "pre_attn_weights": [],
            "pre_mlp_weights": [],
            "final_output_weights": [float("nan")] * max_sources,
            "final_output_entropy": float("nan"),
            "final_output_query_norm": float("nan"),
        }

    def _pad_weights(self, weights: list[float]) -> list[float]:
        max_sources = 1 + 2 * self.cfg.n_layers
        return weights + [float("nan")] * (max_sources - len(weights))

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
        x0 = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        sources: list[torch.Tensor] = [x0]

        aux = self._blank_aux() if return_aux else None

        for block in self.blocks:
            h, stats = block.pre_attn_mixer(sources, return_stats=return_aux)
            if return_aux and aux is not None and stats is not None:
                aux["pre_attn_rms"].append(float(tensor_rms(h).detach().cpu().item()))
                aux["pre_attn_entropy"].append(float(stats["entropy"]))
                aux["pre_attn_embed_weight"].append(float(stats["embed_weight"]))
                aux["pre_attn_prev_weight"].append(float(stats["prev_weight"]))
                aux["pre_attn_query_norm"].append(float(stats["query_norm"]))
                aux["pre_attn_weights"].append(self._pad_weights(stats["mean_weights"]))
            attn_out = block.attn(block.attn_norm(h))
            if return_aux and aux is not None:
                aux["attn_out_rms"].append(float(tensor_rms(attn_out).detach().cpu().item()))
            sources.append(attn_out)

            h, stats = block.pre_mlp_mixer(sources, return_stats=return_aux)
            if return_aux and aux is not None and stats is not None:
                aux["pre_mlp_rms"].append(float(tensor_rms(h).detach().cpu().item()))
                aux["pre_mlp_entropy"].append(float(stats["entropy"]))
                aux["pre_mlp_embed_weight"].append(float(stats["embed_weight"]))
                aux["pre_mlp_prev_weight"].append(float(stats["prev_weight"]))
                aux["pre_mlp_query_norm"].append(float(stats["query_norm"]))
                aux["pre_mlp_weights"].append(self._pad_weights(stats["mean_weights"]))
            mlp_out = block.mlp(block.mlp_norm(h))
            if return_aux and aux is not None:
                aux["mlp_out_rms"].append(float(tensor_rms(mlp_out).detach().cpu().item()))
            sources.append(mlp_out)

        h, out_stats = self.output_mixer(sources, return_stats=return_aux)
        if return_aux and aux is not None and out_stats is not None:
            aux["final_output_weights"] = self._pad_weights(out_stats["mean_weights"])
            aux["final_output_entropy"] = float(out_stats["entropy"])
            aux["final_output_query_norm"] = float(out_stats["query_norm"])
        x = self.final_norm(h)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return ForwardOutput(logits=logits, loss=loss, aux=aux)
