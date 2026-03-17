#!/usr/bin/env python
from __future__ import annotations

import argparse

from attnres_compare.config import load_config
from attnres_compare.models import build_model
from attnres_compare.utils.init import initialize_model, shared_parameter_fingerprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify paired shared initialization between baseline and Full AttnRes configs")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--attnres-config", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.baseline_config)
    attn_cfg = load_config(args.attnres_config)
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
