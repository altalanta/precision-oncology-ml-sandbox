from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pon_ml.data.gen_genomics import synthesize_genomics, save_genomics
from pon_ml.seeds import set_seeds
from pon_ml.tracking import init_mlflow, log_metrics, log_params, mlflow


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    set_seeds(cfg.seed)
    init_mlflow(cfg.mlflow.uri, cfg.mlflow.experiment)

    # Ensure data exists
    data_dir = Path(get_original_cwd()) / "data/genomics"
    csv_path = data_dir / "cfDNA_methylation.csv"
    if not csv_path.exists():
        data = synthesize_genomics(
            samples=cfg.genomics.samples,
            loci=cfg.genomics.loci,
            signal_loci=cfg.genomics.signal_loci,
            age_effect=cfg.genomics.age_effect,
            batch_effect=cfg.genomics.batch_effect,
            seed=cfg.seed,
        )
        save_genomics(data, data_dir)

    df = pd.read_csv(csv_path)
    y = df["label"].values
    groups = df["batch"].values
    X = df.drop(columns=["label", "age", "batch"]).values.astype(np.float32)

    pipe = Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", LogisticRegression(max_iter=200))])
    clf = CalibratedClassifierCV(pipe, method="sigmoid", cv=3) if cfg.genomics.calibrate else pipe

    aucs, aps = [], []
    gkf = GroupKFold(n_splits=cfg.genomics.cv_folds)
    with mlflow.start_run(run_name="genomics_logreg"):
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
            xtr, xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            clf.fit(xtr, ytr)
            prob = clf.predict_proba(xva)[:, 1]
            auc = roc_auc_score(yva, prob)
            ap = average_precision_score(yva, prob)
            aucs.append(auc)
            aps.append(ap)
            log_metrics({"fold_auc": auc, "fold_ap": ap}, step=fold)
        log_metrics({"auc_mean": float(np.mean(aucs)), "ap_mean": float(np.mean(aps))})
        log_params({"loci": X.shape[1], "samples": X.shape[0], "cv_folds": cfg.genomics.cv_folds})


if __name__ == "__main__":
    main()

