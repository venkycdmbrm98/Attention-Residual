from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    variant: str = "baseline"  # baseline | full_attnres
    n_layers: int = 8
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    vocab_size: int = 50257
    max_seq_len: int = 256
    dropout: float = 0.0
    norm_eps: float = 1e-5
    attn_bias: bool = False
    mlp_type: str = "swiglu"  # swiglu | gelu | none
    tie_embeddings: bool = True
    init_std: float = 0.02
    init_resid_scale: bool = True


@dataclass
class DataConfig:
    train_path: str = ""
    val_path: str = ""
    train_meta_path: str = ""
    val_meta_path: str = ""
    seq_len: int = 256
    train_batch_plan: str | None = None
    val_batch_plan: str | None = None
    val_num_batches: int = 64
    pin_memory: bool = True


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    max_steps: int = 10000
    warmup_steps: int = 500
    scheduler: str = "cosine"  # cosine | wsd
    stable_steps: int = 0
    decay_steps: int | None = None


@dataclass
class TrainConfig:
    seed: int = 1337
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    compile: bool = False
    tf32: bool = True
    deterministic: bool = True
    log_interval: int = 10
    eval_interval: int = 250
    checkpoint_interval: int = 1000
    diag_interval: int = 250
    output_dir: str = "outputs/default"
    resume_path: str | None = None
    save_optimizer_state: bool = True
    profile_memory: bool = True


@dataclass
class WandbConfig:
    mode: str = "online"  # online | offline | disabled
    project: str = "attnres-compare"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    job_type: str = "pretrain"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    log_code: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type, payload: dict[str, Any]) -> Any:
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in field_names:
            raise KeyError(f"Unknown config key for {cls.__name__}: {key}")
        field_type = cls.__dataclass_fields__[key].type  # type: ignore[attr-defined]
        default = getattr(cls(), key)
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _construct_dataclass(ExperimentConfig, payload)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
