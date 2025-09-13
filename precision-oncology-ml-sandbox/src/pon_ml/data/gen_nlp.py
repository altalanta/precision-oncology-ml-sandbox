from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from pon_ml.seeds import set_seeds


TOPICS = {
    "peptide_delivery": "Peptide-based ovarian cancer drug delivery",
    "car_t_nk": "CAR-T and NK cell therapies",
    "intraperitoneal": "Intraperitoneal targeting for ovarian metastases",
}


def synthesize_literature(docs_per_topic: int, min_year: int, max_year: int, seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    docs: List[dict] = []
    tid = 0
    for key, topic in TOPICS.items():
        for i in range(docs_per_topic):
            year = int(rng.integers(min_year, max_year + 1))
            title = f"{topic}: study {i+1}"
            # Compose deterministic synthetic text with keywords and negations
            phrases = [
                topic,
                "liposomal nanoparticles",
                "receptor-mediated endocytosis",
                "bioavailability",
                "toxicity",
                "no evidence of disease progression" if i % 3 == 0 else "partial response observed",
            ]
            text = ". ".join(phrases) + "."
            docs.append({"id": f"DOC{tid:04d}", "title": title, "year": year, "text": text, "topic": key})
            tid += 1
    return docs


NEG_TERMS = ["no evidence of", "denies", "not consistent with"]
POS_TERMS = ["metastasis", "progression", "ascites", "platinum", "response", "toxicity"]


def synthesize_ehr(n_notes: int, seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    notes: List[dict] = []
    for i in range(n_notes):
        has_neg = (i % 2) == 0
        neg = rng.choice(NEG_TERMS)
        pos = rng.choice(POS_TERMS)
        text = (
            f"Patient with ovarian carcinoma. {neg} {rng.choice(POS_TERMS)}. "
            f"Chemotherapy with {rng.choice(['carboplatin','paclitaxel','bevacizumab'])}. "
            f"Findings include {pos}."
        ) if has_neg else (
            f"Ovarian cancer follow-up. {rng.choice(POS_TERMS)} present. "
            f"Considering intraperitoneal delivery and CAR-T options."
        )
        notes.append({"id": f"NOTE{i:04d}", "text": text})
    return notes


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    set_seeds(cfg.seed)
    ncfg = cfg.nlp
    lit = synthesize_literature(ncfg.docs_per_topic, ncfg.min_year, ncfg.max_year, seed=cfg.seed)
    ehr = synthesize_ehr(n_notes=50, seed=cfg.seed + 1)
    out = Path(get_original_cwd()) / "data/nlp"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lit).to_json(out / "literature.json", orient="records", lines=True)
    pd.DataFrame(ehr).to_json(out / "ehr_notes.json", orient="records", lines=True)


if __name__ == "__main__":
    main()

