from pathlib import Path

import json
import pandas as pd

from pon_ml.data.gen_genomics import synthesize_genomics
from pon_ml.data.gen_imaging import synthesize_tiles
from pon_ml.data.gen_nlp import synthesize_literature, TOPICS


def test_genomics_shapes_and_balance(tmp_path: Path):
    data = synthesize_genomics(samples=120, loci=200, signal_loci=10, age_effect=0.02, batch_effect=0.5, seed=123)
    assert data.X.shape == (120, 200)
    # label roughly balanced (not degenerate)
    assert 0.1 < data.y.mean() < 0.9


def test_imaging_tiles(tmp_path: Path):
    X, y = synthesize_tiles(size=64, samples_per_class=5, seed=42)
    assert X.shape == (10, 64, 64, 3)
    assert set(y.tolist()) == {0, 1}


def test_literature_contains_topics(tmp_path: Path):
    docs = synthesize_literature(docs_per_topic=3, min_year=2015, max_year=2016, seed=42)
    assert len(docs) == 3 * len(TOPICS)
    # Ensure at least one doc per topic has its keywords
    for key in TOPICS.keys():
        assert any(key in d["topic"] for d in docs)

