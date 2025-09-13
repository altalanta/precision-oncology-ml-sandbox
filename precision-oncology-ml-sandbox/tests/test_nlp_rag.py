from pathlib import Path

import pandas as pd

from pon_ml.data.gen_nlp import synthesize_literature
from pon_ml.nlp.rag import search


def test_rag_returns_relevant_doc(tmp_path: Path):
    docs = synthesize_literature(docs_per_topic=2, min_year=2018, max_year=2019, seed=7)
    query = "CAR-T intraperitoneal delivery"
    results = search(query, top_k=3, corpus=docs)
    assert len(results) >= 1
    assert any("car-t" in (r["title"].lower() + " " + r["text"].lower()) for r in results)

