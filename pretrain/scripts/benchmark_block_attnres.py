#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import time
from contextlib import nullcontext

import torch

from attnres_compare.config import load_config
from attnres_compare.data.memmap import MemmapCorpus
from attnres_compare.models import build_model
from attnres_compare.utils.init import initialize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline vs Block AttnRes vs Full AttnRes on a CUDA device"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config, e.g. configs/tinystories_block_attnres.yaml",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "block_attnres", "full_attnres"],
        choices=["baseline", "block_attnres", "full_attnres"],
        help="Model variants to benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override micro batch size. Defaults to config train.micro_batch_size.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override sequence length. Defaults to config data.seq_len.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default=None,
        help="Override compute dtype. Defaults to config train.dtype.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override model.attnres_block_size for block_attnres.",
    )
    return parser.parse_args()


def resolve_vocab_size(cfg) -> int:
    if cfg.model.vocab_size and cfg.model.vocab_size != 50257:
        return cfg.model.vocab_size
    if cfg.data.train_meta_path:
        corpus = MemmapCorpus(cfg.data.train_path, cfg.data.train_meta_path)
        return corpus.meta.vocab_size
    return cfg.model.vocab_size


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    if dtype_name == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def benchmark_variant(
    cfg,
    variant: str,
    batch_size: int,
    seq_len: int,
    dtype_name: str,
    block_size: int | None,
    warmup_iters: int,
    benchmark_iters: int,
) -> dict[str, float]:
    device = torch.device("cuda")
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.variant = variant
    model_cfg.max_seq_len = seq_len
    model_cfg.vocab_size = resolve_vocab_size(cfg)
    if variant == "block_attnres" and block_size is not None:
        model_cfg.attnres_block_size = block_size

    model = build_model(model_cfg).to(device)
    initialize_model(model, model_cfg, seed=cfg.train.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        fused=True,
    )

    x = torch.randint(model_cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(model_cfg.vocab_size, (batch_size, seq_len), device=device)

    for _ in range(warmup_iters):
        with autocast_context(device, dtype_name):
            out = model(x, targets=y)
            loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        with autocast_context(device, dtype_name):
            out = model(x, targets=y)
            loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_iter = batch_size * seq_len
    return {
        "sec_per_iter": elapsed / benchmark_iters,
        "tokens_per_sec": (benchmark_iters * tokens_per_iter) / elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    cfg = load_config(args.config)
    batch_size = args.batch_size or cfg.train.micro_batch_size
    seq_len = args.seq_len or cfg.data.seq_len
    dtype_name = args.dtype or cfg.train.dtype

    print(
        {
            "config": args.config,
            "variants": args.variants,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "dtype": dtype_name,
            "block_size": args.block_size or cfg.model.attnres_block_size,
            "warmup_iters": args.warmup_iters,
            "benchmark_iters": args.benchmark_iters,
            "device": torch.cuda.get_device_name(0),
        }
    )

    for variant in args.variants:
        metrics = benchmark_variant(
            cfg=cfg,
            variant=variant,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_name=dtype_name,
            block_size=args.block_size,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
        )
        print(variant, metrics)