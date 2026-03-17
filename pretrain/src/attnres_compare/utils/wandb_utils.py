from __future__ import annotations

from pathlib import Path
from typing import Any

from attnres_compare.config import ExperimentConfig

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def init_wandb(config: ExperimentConfig, output_dir: str, enabled: bool) -> Any:
    if not enabled or config.wandb.mode == "disabled" or wandb is None:
        return None
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        mode=config.wandb.mode,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        dir=output_dir,
        config=config.to_dict(),
    )
    if config.wandb.log_code:
        wandb.run.log_code(root=str(Path(output_dir).resolve().parent))
    return run


def log_metrics(run: Any, metrics: dict[str, float], step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)
