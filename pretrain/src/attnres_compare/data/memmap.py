from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class CorpusMeta:
    vocab_size: int
    dtype: str
    num_tokens: int
    tokenizer_name: str | None = None
    extra: dict[str, Any] | None = None


class MemmapCorpus:
    def __init__(self, path: str | Path, meta_path: str | Path) -> None:
        self.path = Path(path)
        self.meta_path = Path(meta_path)
        with self.meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.meta = CorpusMeta(
            vocab_size=int(meta["vocab_size"]),
            dtype=str(meta.get("dtype", "uint32")),
            num_tokens=int(meta["num_tokens"]),
            tokenizer_name=meta.get("tokenizer_name"),
            extra=meta,
        )
        dtype = np.dtype(self.meta.dtype)
        self.data = np.memmap(self.path, dtype=dtype, mode="r")
        if len(self.data) != self.meta.num_tokens:
            raise ValueError(
                f"Token count mismatch for {self.path}: data has {len(self.data)} tokens, meta says {self.meta.num_tokens}"
            )

    def get_batch(
        self,
        offsets: np.ndarray,
        seq_len: int,
        device: torch.device,
        non_blocking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(offsets.shape[0])
        x = np.empty((batch_size, seq_len), dtype=np.int64)
        y = np.empty((batch_size, seq_len), dtype=np.int64)
        for i, start in enumerate(offsets.tolist()):
            end = start + seq_len
            x[i] = self.data[start:end]
            y[i] = self.data[start + 1 : end + 1]
        x_t = torch.from_numpy(x).to(device=device, dtype=torch.long, non_blocking=non_blocking)
        y_t = torch.from_numpy(y).to(device=device, dtype=torch.long, non_blocking=non_blocking)
        return x_t, y_t


class BatchPlan:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets = np.load(self.path)
        if self.offsets.ndim != 2:
            raise ValueError(f"Batch plan must be rank-2 [num_micro_batches, global_micro_batch_size], got {self.offsets.shape}")

    @property
    def num_micro_batches(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def global_micro_batch_size(self) -> int:
        return int(self.offsets.shape[1])

    def get_rank_offsets(self, micro_batch_index: int, rank: int, world_size: int, local_micro_batch_size: int) -> np.ndarray:
        row = self.offsets[micro_batch_index]
        expected = world_size * local_micro_batch_size
        if row.shape[0] != expected:
            raise ValueError(
                f"Batch plan row width {row.shape[0]} does not match world_size * micro_batch_size = {expected}"
            )
        start = rank * local_micro_batch_size
        end = start + local_micro_batch_size
        return row[start:end]


def build_evenly_spaced_eval_plan(num_tokens: int, seq_len: int, num_batches: int, batch_size: int) -> np.ndarray:
    max_offset = num_tokens - seq_len - 1
    total = num_batches * batch_size
    if total <= 0:
        raise ValueError("Need at least one evaluation example")
    if max_offset <= 0:
        raise ValueError("Corpus too short for the requested sequence length")
    values = np.linspace(0, max_offset, num=total, dtype=np.int64)
    return values.reshape(num_batches, batch_size)
