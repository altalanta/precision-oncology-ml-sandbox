from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from pon_ml.seeds import set_seeds


@dataclass
class GenomicsData:
    X: np.ndarray  # shape (n_samples, n_loci)
    y: np.ndarray  # binary labels
    age: np.ndarray  # age in years
    batch: np.ndarray  # batch ids
    loci_names: list[str]


def synthesize_genomics(
    samples: int, loci: int, signal_loci: int, age_effect: float, batch_effect: float, seed: int
) -> GenomicsData:
    rng = np.random.default_rng(seed)
    # Base methylation beta values ~ Beta distribution per locus
    base_alpha = rng.uniform(1.5, 3.0, size=loci)
    base_beta = rng.uniform(1.5, 3.0, size=loci)
    X = rng.beta(base_alpha, base_beta, size=(samples, loci)).astype(np.float32)

    # True signal loci
    signal_idx = rng.choice(loci, size=signal_loci, replace=False)
    coef = rng.normal(0.0, 0.8, size=loci)
    coef[signal_idx] += rng.normal(2.0, 0.5, size=signal_loci)  # sparse positive signals

    # Confounders
    age = rng.normal(60, 10, size=samples)
    batch = rng.integers(0, 4, size=samples)  # 4 batches

    # Linear logit with confounding
    logit = X @ coef + age_effect * (age - age.mean()) / age.std() + batch_effect * (batch - 1.5)
    prob = 1 / (1 + np.exp(-logit / np.sqrt(loci)))
    y = (rng.uniform(0, 1, size=samples) < prob).astype(int)

    # Add subtle batch shift per locus
    batch_shift = rng.normal(0, 0.02, size=(4, loci))
    X = np.clip(X + batch_shift[batch], 0, 1)

    loci_names = [f"L{i:05d}" for i in range(loci)]
    return GenomicsData(X=X, y=y, age=age.astype(np.float32), batch=batch.astype(np.int32), loci_names=loci_names)


def save_genomics(data: GenomicsData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data.X, columns=data.loci_names)
    df.insert(0, "label", data.y)
    df.insert(1, "age", data.age)
    df.insert(2, "batch", data.batch)
    df.to_csv(out_dir / "cfDNA_methylation.csv", index=False)
    # Data card
    card = {
        "modality": "genomics",
        "samples": int(data.X.shape[0]),
        "loci": int(data.X.shape[1]),
        "label_balance": float(data.y.mean()),
        "batches": int(np.unique(data.batch).size),
        "notes": "Sparse signal loci with age and batch confounding. Deterministic seed.",
    }
    pd.Series(card).to_json(out_dir / "cfDNA_methylation.datacard.json")


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover - exercised via tests at function level
    set_seeds(cfg.seed)
    gcfg = cfg.genomics
    data = synthesize_genomics(
        samples=gcfg.samples,
        loci=gcfg.loci,
        signal_loci=gcfg.signal_loci,
        age_effect=gcfg.age_effect,
        batch_effect=gcfg.batch_effect,
        seed=cfg.seed,
    )
    out = Path(get_original_cwd()) / "data/genomics"
    save_genomics(data, out)


if __name__ == "__main__":
    main()

