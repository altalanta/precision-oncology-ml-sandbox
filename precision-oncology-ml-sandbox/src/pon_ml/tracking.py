from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def init_mlflow(uri: str | None = None, experiment: str = "pon-sandbox") -> None:
    """Initialize MLflow to local file store by default."""
    mlruns_path = Path(uri or "./artifacts/mlruns")
    if not str(mlruns_path).startswith("file:"):
        # Use local file path scheme
        mlflow.set_tracking_uri(f"file:{mlruns_path}")
    else:
        mlflow.set_tracking_uri(str(mlruns_path))
    mlruns_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(experiment)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step)


def log_params(params: Dict[str, Any]) -> None:
    mlflow.log_params({k: str(v) for k, v in params.items()})


__all__ = ["init_mlflow", "log_metrics", "log_params", "mlflow"]

