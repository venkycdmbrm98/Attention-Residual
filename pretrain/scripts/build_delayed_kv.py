#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOS = 0
EOS = 1
SEP = 2
Q = 3
ANS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a delayed key-value retrieval corpus for mechanistic interp")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-examples", type=int, default=200000)
    parser.add_argument("--val-examples", type=int, default=10000)
    parser.add_argument("--num-keys", type=int, default=256)
    parser.add_argument("--pairs-per-example", type=int, default=8)
    parser.add_argument("--noise-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_example(rng: np.random.Generator, num_keys: int, pairs_per_example: int, noise_tokens: int) -> list[int]:
    keys = np.arange(num_keys)
    rng.shuffle(keys)
    selected = keys[:pairs_per_example]
    values = rng.permutation(selected)

    noise_base = 5 + 2 * num_keys
    sequence = [BOS]
    for key, value in zip(selected.tolist(), values.tolist()):
        sequence.extend([5 + key, 5 + num_keys + value, SEP])
    for _ in range(noise_tokens):
        sequence.append(noise_base + int(rng.integers(0, num_keys)))
    target_idx = int(rng.integers(0, pairs_per_example))
    query_key = int(selected[target_idx])
    query_value = int(values[target_idx])
    sequence.extend([Q, 5 + query_key, ANS, 5 + num_keys + query_value, EOS])
    return sequence


def build_split(num_examples: int, rng: np.random.Generator, num_keys: int, pairs_per_example: int, noise_tokens: int) -> np.ndarray:
    out = []
    for _ in range(num_examples):
        out.extend(build_example(rng, num_keys, pairs_per_example, noise_tokens))
    return np.asarray(out, dtype=np.uint32)


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train = build_split(args.train_examples, rng, args.num_keys, args.pairs_per_example, args.noise_tokens)
    val = build_split(args.val_examples, np.random.default_rng(args.seed + 1), args.num_keys, args.pairs_per_example, args.noise_tokens)
    train.tofile(output_dir / "train.bin")
    val.tofile(output_dir / "val.bin")
    vocab_size = 5 + 3 * args.num_keys
    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint32",
        "num_tokens": int(train.shape[0]),
        "train_num_tokens": int(train.shape[0]),
        "val_num_tokens": int(val.shape[0]),
        "tokenizer_name": "synthetic_delayed_kv",
        "num_keys": args.num_keys,
        "pairs_per_example": args.pairs_per_example,
        "noise_tokens": args.noise_tokens,
        "special_tokens": {"BOS": BOS, "EOS": EOS, "SEP": SEP, "Q": Q, "ANS": ANS},
    }
    with (output_dir / "train_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({**meta, "num_tokens": int(train.shape[0])}, handle, indent=2)
    with (output_dir / "val_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({**meta, "num_tokens": int(val.shape[0])}, handle, indent=2)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump({**meta, "num_tokens": int(train.shape[0])}, handle, indent=2)
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **{f"K_{i}": 5 + i for i in range(args.num_keys)},
                **{f"V_{i}": 5 + args.num_keys + i for i in range(args.num_keys)},
            },
            handle,
            indent=2,
        )
