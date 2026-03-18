#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Any

from attnres_compare.config import load_config
from attnres_compare.models import build_model
from attnres_compare.utils.init import initialize_model, shared_parameter_fingerprint


def _assert_equal(section: str, baseline: Any, variant: Any) -> None:
    if baseline != variant:
        raise SystemExit(
            f"Paired-run mismatch in {section}:\nbaseline={baseline}\nvariant={variant}"
        )


def _assert_protocol_match(base_cfg, attn_cfg) -> None:
    if hasattr(base_cfg.data, "to_dict"):
        _assert_equal("data", base_cfg.data.to_dict(), attn_cfg.data.to_dict())
    else:
        _assert_equal("data", vars(base_cfg.data), vars(attn_cfg.data))
    _assert_equal("optim", vars(base_cfg.optim), vars(attn_cfg.optim))

    train_keys = [
        "seed",
        "micro_batch_size",
        "grad_accum_steps",
        "dtype",
        "compile",
        "tf32",
        "deterministic",
        "log_interval",
        "eval_interval",
        "checkpoint_interval",
        "diag_interval",
        "save_optimizer_state",
        "profile_memory",
    ]
    for key in train_keys:
        _assert_equal(f"train.{key}", getattr(base_cfg.train, key), getattr(attn_cfg.train, key))

    shared_model_keys = [
        "n_layers",
        "d_model",
        "n_heads",
        "d_ff",
        "vocab_size",
        "max_seq_len",
        "dropout",
        "norm_eps",
        "attn_bias",
        "mlp_type",
        "tie_embeddings",
        "init_std",
        "init_resid_scale",
    ]
    for key in shared_model_keys:
        _assert_equal(f"model.{key}", getattr(base_cfg.model, key), getattr(attn_cfg.model, key))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify paired shared initialization between a baseline config and an AttnRes-family config")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--attnres-config", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.baseline_config)
    attn_cfg = load_config(args.attnres_config)
    _assert_protocol_match(base_cfg, attn_cfg)
    base_model = build_model(base_cfg.model)
    attn_model = build_model(attn_cfg.model)
    initialize_model(base_model, base_cfg.model, seed=base_cfg.train.seed)
    initialize_model(attn_model, attn_cfg.model, seed=attn_cfg.train.seed)
    base_fp = shared_parameter_fingerprint(base_model)
    attn_fp = shared_parameter_fingerprint(attn_model)
    print(f"baseline_shared_fingerprint={base_fp}")
    print(f"attnres_shared_fingerprint={attn_fp}")
    if base_fp != attn_fp:
        raise SystemExit("Shared parameter fingerprints differ")
    print("Paired initialization check passed")