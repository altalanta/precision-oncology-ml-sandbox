from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from pon_ml.data.gen_nlp import synthesize_ehr
from pon_ml.seeds import set_seeds


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    set_seeds(cfg.seed)
    notes = synthesize_ehr(50, seed=cfg.seed + 2)
    out = Path(get_original_cwd()) / "data/nlp"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(notes).to_json(out / "ehr_notes.json", orient="records", lines=True)


if __name__ == "__main__":
    main()

