#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic train/val batch plans for paired runs")
    parser.add_argument("--train-num-tokens", type=int, required=True)
    parser.add_argument("--val-num-tokens", type=int, default=None)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--num-val-batches", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_max_offset = args.train_num_tokens - args.seq_len - 2
    val_num_tokens = args.val_num_tokens if args.val_num_tokens is not None else args.train_num_tokens
    val_max_offset = val_num_tokens - args.seq_len - 2
    if train_max_offset <= 0 or val_max_offset <= 0:
        raise ValueError("Corpus is shorter than seq_len + 2")

    global_micro_batch = args.micro_batch_size * args.world_size
    num_micro_batches = args.max_steps * args.grad_accum_steps
    rng = np.random.default_rng(args.seed)
    train_offsets = rng.integers(0, train_max_offset, size=(num_micro_batches, global_micro_batch), dtype=np.int64)
    val_rng = np.random.default_rng(args.seed + 1)
    val_offsets = val_rng.integers(0, val_max_offset, size=(args.num_val_batches, global_micro_batch), dtype=np.int64)

    np.save(output_dir / "train_plan.npy", train_offsets)
    np.save(output_dir / "val_plan.npy", val_offsets)
    with (output_dir / "plan_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_num_tokens": args.train_num_tokens,
                "val_num_tokens": val_num_tokens,
                "seq_len": args.seq_len,
                "micro_batch_size": args.micro_batch_size,
                "world_size": args.world_size,
                "grad_accum_steps": args.grad_accum_steps,
                "max_steps": args.max_steps,
                "num_val_batches": args.num_val_batches,
                "seed": args.seed,
                "train_shape": list(train_offsets.shape),
                "val_shape": list(val_offsets.shape),
            },
            handle,
            indent=2,
        )
