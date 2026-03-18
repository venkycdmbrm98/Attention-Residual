#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize a Hugging Face text dataset into contiguous memmaps")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--num-proc", type=int, default=4)
    return parser.parse_args()


def tokenize_split(dataset, tokenizer, text_field: str, append_eos: bool):
    logger.info(f"Starting tokenization | split size={len(dataset)}")

    eos = tokenizer.eos_token_id
    if eos is None and append_eos:
        raise ValueError("Tokenizer has no EOS token but --append-eos was requested")

    def fn(batch):
        texts = batch[text_field]
        output = tokenizer(texts, add_special_tokens=False)
        ids = output["input_ids"]
        if append_eos:
            ids = [x + [eos] for x in ids]
        return {"input_ids": ids}

    start = time.time()

    tokenized = dataset.map(
        fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
        desc="Tokenizing dataset",
    )

    logger.info("Finished HF map() tokenization")

    total = 0
    for i, item in enumerate(tokenized):
        total += len(item["input_ids"])
        if i % 10000 == 0 and i > 0:
            logger.info(f"Counting tokens... processed {i} samples")

    logger.info(f"Total tokens: {total:,}")
    logger.info(f"Tokenization took {time.time() - start:.2f}s")

    return tokenized, total


def write_memmap(tokenized, total_tokens: int, output_path: Path) -> None:
    logger.info(f"Writing memmap → {output_path}")
    logger.info(f"Total tokens to write: {total_tokens:,}")

    start = time.time()

    mmap = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))
    cursor = 0

    for i, item in enumerate(tokenized):
        ids = np.asarray(item["input_ids"], dtype=np.uint32)
        mmap[cursor : cursor + len(ids)] = ids
        cursor += len(ids)

        if i % 10000 == 0 and i > 0:
            logger.info(f"Writing progress... {cursor:,} tokens written")

    mmap.flush()

    logger.info(f"Finished writing memmap ({output_path})")
    logger.info(f"Write took {time.time() - start:.2f}s")


if __name__ == "__main__":
    args = parse_args()

    logger.info("==== STARTING PRETOKENIZATION PIPELINE ====")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    logger.info("Loading datasets...")
    train_ds = load_dataset(args.dataset, args.dataset_config, split=args.train_split)
    val_ds = load_dataset(args.dataset, args.dataset_config, split=args.val_split)

    logger.info(f"Train size: {len(train_ds):,}")
    logger.info(f"Val size: {len(val_ds):,}")

    logger.info("---- TOKENIZING TRAIN ----")
    train_tok, train_total = tokenize_split(train_ds, tokenizer, args.text_field, args.append_eos)

    logger.info("---- TOKENIZING VAL ----")
    val_tok, val_total = tokenize_split(val_ds, tokenizer, args.text_field, args.append_eos)

    logger.info("---- WRITING TRAIN MEMMAP ----")
    write_memmap(train_tok, train_total, output_dir / "train.bin")

    logger.info("---- WRITING VAL MEMMAP ----")
    write_memmap(val_tok, val_total, output_dir / "val.bin")

    logger.info("Saving metadata...")
    meta = {
        "vocab_size": int(tokenizer.vocab_size),
        "dtype": "uint32",
        "num_tokens": train_total,
        "tokenizer_name": args.tokenizer,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "text_field": args.text_field,
        "append_eos": args.append_eos,
        "train_num_tokens": train_total,
        "val_num_tokens": val_total,
    }

    with (output_dir / "train_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({**meta, "num_tokens": train_total}, handle, indent=2)

    with (output_dir / "val_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({**meta, "num_tokens": val_total}, handle, indent=2)

    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump({**meta, "num_tokens": train_total}, handle, indent=2)

    logger.info("==== DONE ====")