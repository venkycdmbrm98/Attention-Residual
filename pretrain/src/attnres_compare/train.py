from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from attnres_compare.config import ExperimentConfig, load_config, save_config
from attnres_compare.data import BatchPlan, MemmapCorpus, build_evenly_spaced_eval_plan
from attnres_compare.models import build_model
from attnres_compare.utils.checkpoint import load_checkpoint, save_checkpoint
from attnres_compare.utils.diagnostics import block_grad_norms, global_grad_norm, make_heatmap_image, tensor_hist_stats
from attnres_compare.utils.distributed import DistInfo, destroy_distributed, init_distributed, maybe_all_reduce_dict
from attnres_compare.utils.init import initialize_model, shared_parameter_fingerprint
from attnres_compare.utils.optim import LRScheduler, build_optimizer
from attnres_compare.utils.seed import seed_everything
from attnres_compare.utils.wandb_utils import init_wandb, log_metrics

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline or Full AttnRes causal LM")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    return parser.parse_args()


class Trainer:
    def __init__(self, cfg: ExperimentConfig, dist_info: DistInfo) -> None:
        self.cfg = cfg
        self.dist = dist_info
        self.output_dir = Path(cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.train.tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = torch.device("cuda", dist_info.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        seed_everything(cfg.train.seed + dist_info.rank, deterministic=cfg.train.deterministic)

        self.train_corpus = MemmapCorpus(cfg.data.train_path, cfg.data.train_meta_path)
        self.val_corpus = MemmapCorpus(cfg.data.val_path, cfg.data.val_meta_path)
        self.cfg.model.vocab_size = self.train_corpus.meta.vocab_size
        self.cfg.model.max_seq_len = self.cfg.data.seq_len

        if cfg.data.train_batch_plan is None:
            raise ValueError("Research-grade paired runs should use an explicit train_batch_plan")
        self.train_plan = BatchPlan(cfg.data.train_batch_plan)
        expected_width = self.dist.world_size * cfg.train.micro_batch_size
        if self.train_plan.global_micro_batch_size != expected_width:
            raise ValueError(
                f"Train batch plan width {self.train_plan.global_micro_batch_size} does not match world_size * micro_batch_size {expected_width}"
            )
        if self.train_plan.num_micro_batches < cfg.optim.max_steps * cfg.train.grad_accum_steps:
            raise ValueError("Train batch plan is too short for max_steps * grad_accum_steps")

        if cfg.data.val_batch_plan is not None:
            val_plan_obj = BatchPlan(cfg.data.val_batch_plan)
            if val_plan_obj.global_micro_batch_size != expected_width:
                raise ValueError(
                    f"Val batch plan width {val_plan_obj.global_micro_batch_size} does not match world_size * micro_batch_size {expected_width}"
                )
            self.val_plan = val_plan_obj.offsets
        else:
            self.val_plan = build_evenly_spaced_eval_plan(
                num_tokens=self.val_corpus.meta.num_tokens,
                seq_len=cfg.data.seq_len,
                num_batches=cfg.data.val_num_batches,
                batch_size=self.dist.world_size * cfg.train.micro_batch_size,
            )

        self.model = build_model(cfg.model).to(self.device)
        initialize_model(self.model, cfg.model, seed=cfg.train.seed)
        self.shared_fingerprint = shared_parameter_fingerprint(self.model)

        if cfg.train.compile:
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(self.model, cfg.optim)
        self.scheduler = LRScheduler(self.optimizer, cfg.optim)
        self.grad_scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.train.dtype == "float16" and self.device.type == "cuda")

        self.start_step = 0
        self.tokens_seen = 0
        if cfg.train.resume_path:
            self._load_resume(cfg.train.resume_path)

        if self.dist.enabled:
            self.model = DDP(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )

        self.model_no_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        self.scheduler.step(self.start_step)

        self.run = init_wandb(cfg, str(self.output_dir), enabled=self.dist.is_main)
        if self.dist.is_main:
            save_config(cfg, self.output_dir / "resolved_config.yaml")
            with (self.output_dir / "run_info.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "num_parameters": self.model_no_ddp.num_parameters(),
                        "shared_parameter_fingerprint": self.shared_fingerprint,
                        "train_tokens": self.train_corpus.meta.num_tokens,
                        "val_tokens": self.val_corpus.meta.num_tokens,
                    },
                    handle,
                    indent=2,
                )

    def _autocast_context(self):
        if self.device.type != "cuda" or self.cfg.train.dtype == "float32":
            return nullcontext()
        if self.cfg.train.dtype == "bfloat16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.cfg.train.dtype == "float16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        raise ValueError(f"Unsupported dtype: {self.cfg.train.dtype}")

    def _load_resume(self, path: str) -> None:
        state = load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("grad_scaler") is not None:
            self.grad_scaler.load_state_dict(state["grad_scaler"])
        self.start_step = int(state.get("step", 0))
        self.tokens_seen = int(state.get("tokens_seen", 0))

    def _save_checkpoint(self, step: int, val_loss: float | None = None) -> None:
        if not self.dist.is_main:
            return
        state = {
            "config": self.cfg.to_dict(),
            "model": self.model_no_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.cfg.train.save_optimizer_state else None,
            "grad_scaler": self.grad_scaler.state_dict() if self.grad_scaler.is_enabled() else None,
            "step": step,
            "tokens_seen": self.tokens_seen,
            "val_loss": val_loss,
            "shared_parameter_fingerprint": self.shared_fingerprint,
        }
        path = self.output_dir / "checkpoints" / f"step_{step:07d}.pt"
        save_checkpoint(path, state)
        latest = self.output_dir / "checkpoints" / "latest.pt"
        save_checkpoint(latest, state)

    def _get_train_batch(self, micro_batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = self.train_plan.get_rank_offsets(
            micro_batch_index,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            local_micro_batch_size=self.cfg.train.micro_batch_size,
        )
        return self.train_corpus.get_batch(offsets, self.cfg.data.seq_len, self.device, non_blocking=self.cfg.data.pin_memory)

    def _get_val_batch(self, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.val_plan[batch_index]
        start = self.dist.rank * self.cfg.train.micro_batch_size
        end = start + self.cfg.train.micro_batch_size
        offsets = row[start:end]
        return self.val_corpus.get_batch(offsets, self.cfg.data.seq_len, self.device, non_blocking=self.cfg.data.pin_memory)

    @torch.no_grad()
    def evaluate(self, step: int) -> tuple[float, dict[str, float], dict[str, Any] | None]:
        self.model.eval()
        losses = []
        aux_snapshot: dict[str, Any] | None = None
        for batch_idx in range(self.cfg.data.val_num_batches):
            x, y = self._get_val_batch(batch_idx)
            with self._autocast_context():
                out = self.model(x, targets=y, return_aux=batch_idx == 0)
            losses.append(float(out.loss.detach().float().item()))
            if batch_idx == 0:
                aux_snapshot = out.aux
        loss_tensor = torch.tensor([sum(losses) / len(losses)], device=self.device)
        if self.dist.enabled:
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            loss_tensor /= self.dist.world_size
        val_loss = float(loss_tensor.item())
        metrics = {
            "val/loss": val_loss,
            "val/perplexity": float(math.exp(min(20.0, val_loss))),
        }
        return val_loss, metrics, aux_snapshot

    def _log_aux(self, aux: dict[str, Any] | None, step: int, prefix: str) -> None:
        if aux is None:
            return
        metrics: dict[str, float] = {}
        for key, value in aux.items():
            if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
                metrics.update(tensor_hist_stats(value, f"{prefix}/{key}"))
            elif isinstance(value, (int, float)):
                metrics[f"{prefix}/{key}"] = float(value)
        log_metrics(self.run, metrics, step)

        if self.dist.is_main and self.run is not None and wandb is not None:
            images = {}
            if "pre_attn_weights" in aux and aux["pre_attn_weights"]:
                images[f"{prefix}/pre_attn_weights"] = wandb.Image(make_heatmap_image(aux["pre_attn_weights"], f"{prefix} pre-attn depth weights"))
            if "pre_mlp_weights" in aux and aux["pre_mlp_weights"]:
                images[f"{prefix}/pre_mlp_weights"] = wandb.Image(make_heatmap_image(aux["pre_mlp_weights"], f"{prefix} pre-mlp depth weights"))
            if images:
                self.run.log(images, step=step)

    def train(self) -> None:
        step = self.start_step
        tokens_per_step = self.cfg.train.micro_batch_size * self.cfg.train.grad_accum_steps * self.dist.world_size * self.cfg.data.seq_len
        running_loss = 0.0
        running_steps = 0
        t0 = time.time()
        iterator = range(step, self.cfg.optim.max_steps)
        if self.dist.is_main:
            iterator = tqdm(iterator, initial=step, total=self.cfg.optim.max_steps, desc="train")

        while step < self.cfg.optim.max_steps:
            self.model.train()
            collect_diag = (step % self.cfg.train.diag_interval == 0)
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            last_aux: dict[str, Any] | None = None

            for micro_idx in range(self.cfg.train.grad_accum_steps):
                micro_batch_index = step * self.cfg.train.grad_accum_steps + micro_idx
                x, y = self._get_train_batch(micro_batch_index)
                sync_context = nullcontext()
                if isinstance(self.model, DDP) and micro_idx < self.cfg.train.grad_accum_steps - 1:
                    sync_context = self.model.no_sync()
                with sync_context:
                    with self._autocast_context():
                        out = self.model(x, targets=y, return_aux=collect_diag and micro_idx == self.cfg.train.grad_accum_steps - 1)
                        loss = out.loss / self.cfg.train.grad_accum_steps
                    if self.grad_scaler.is_enabled():
                        self.grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                step_loss += float(out.loss.detach().float().item())
                if out.aux is not None:
                    last_aux = out.aux

            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            grad_norm_value = float(grad_norm.detach().float().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            step += 1
            self.tokens_seen += tokens_per_step
            lr = self.scheduler.step(step)

            reduced = maybe_all_reduce_dict({"train/loss": step_loss / self.cfg.train.grad_accum_steps}, self.device)
            train_loss = reduced["train/loss"]
            running_loss += train_loss
            running_steps += 1

            if step % self.cfg.train.log_interval == 0:
                elapsed = time.time() - t0
                metrics = {
                    "train/loss": running_loss / max(1, running_steps),
                    "optim/lr": lr,
                    "optim/global_grad_norm": grad_norm_value,
                    "train/tokens_seen": float(self.tokens_seen),
                    "train/tokens_per_sec": float((running_steps * tokens_per_step) / max(elapsed, 1e-6)),
                }
                if self.device.type == "cuda" and self.cfg.train.profile_memory:
                    metrics["sys/max_cuda_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                if collect_diag and last_aux is not None:
                    metrics.update(block_grad_norms(self.model_no_ddp))
                log_metrics(self.run, metrics, step)
                if collect_diag:
                    self._log_aux(last_aux, step, prefix="train_diag")
                running_loss = 0.0
                running_steps = 0
                t0 = time.time()
                if self.dist.is_main and isinstance(iterator, tqdm):
                    iterator.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.2e}")

            if step % self.cfg.train.eval_interval == 0 or step == self.cfg.optim.max_steps:
                val_loss, val_metrics, val_aux = self.evaluate(step)
                log_metrics(self.run, val_metrics, step)
                self._log_aux(val_aux, step, prefix="val_diag")

            if step % self.cfg.train.checkpoint_interval == 0 or step == self.cfg.optim.max_steps:
                self._save_checkpoint(step)

            if self.dist.is_main and isinstance(iterator, tqdm):
                iterator.update(1)
            elif not self.dist.is_main:
                pass

        if self.dist.is_main and isinstance(iterator, tqdm):
            iterator.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dist_info = init_distributed()
    try:
        trainer = Trainer(cfg, dist_info)
        trainer.train()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
