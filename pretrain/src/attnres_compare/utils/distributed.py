from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistInfo(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistInfo(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def maybe_all_reduce_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {k: float(v.item()) for k, v in zip(keys, values)}
