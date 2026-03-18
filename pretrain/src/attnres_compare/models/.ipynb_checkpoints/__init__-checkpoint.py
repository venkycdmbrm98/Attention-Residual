# from __future__ import annotations

# from attnres_compare.config import ModelConfig
# from attnres_compare.models.attnres import FullAttnResGPT
# from attnres_compare.models.baseline import BaselineGPT


# def build_model(cfg: ModelConfig):
#     if cfg.variant == "baseline":
#         return BaselineGPT(cfg)
#     if cfg.variant == "full_attnres":
#         return FullAttnResGPT(cfg)
#     raise ValueError(f"Unknown model variant: {cfg.variant}")


# __all__ = ["build_model", "BaselineGPT", "FullAttnResGPT"]


from __future__ import annotations

from attnres_compare.config import ModelConfig
from attnres_compare.models.attnres import BlockAttnResGPT, FullAttnResGPT
from attnres_compare.models.baseline import BaselineGPT


def build_model(cfg: ModelConfig):
    if cfg.variant == "baseline":
        return BaselineGPT(cfg)
    if cfg.variant == "full_attnres":
        return FullAttnResGPT(cfg)
    if cfg.variant == "block_attnres":
        return BlockAttnResGPT(cfg)
    raise ValueError(f"Unknown model variant: {cfg.variant}")


__all__ = ["build_model", "BaselineGPT", "FullAttnResGPT", "BlockAttnResGPT"]