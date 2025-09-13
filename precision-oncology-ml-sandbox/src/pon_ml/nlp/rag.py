from __future__ import annotations

import json
from pathlib import Path
from typing import List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from pon_ml.nlp.bm25 import BM25
from pon_ml.seeds import set_seeds


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.replace("/", " ").replace("-", " ").split()]


def load_corpus(path: Path) -> List[dict]:
    docs = []
    with path.open() as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def search(query: str, top_k: int, corpus: List[dict]) -> List[dict]:
    docs_tokens = [_tokenize(d["title"] + " " + d["text"]) for d in corpus]
    bm25 = BM25(docs_tokens)
    q = _tokenize(query)
    ranks = bm25.query(q, top_k=top_k)
    return [{**corpus[i], "score": float(s)} for i, s in ranks]


def cli() -> None:  # pragma: no cover - exercised in tests via function calls
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()
    cfg = {"top_k": args.top_k}
    base = Path.cwd()
    corpus = load_corpus(base / "data/nlp/literature.json")
    results = search(args.query, args.top_k, corpus)
    for r in results:
        print(f"{r['id']} {r['year']} {r['title']} (score={r['score']:.3f})")


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    set_seeds(cfg.seed)
    base = Path(get_original_cwd())
    corpus_path = base / "data/nlp/literature.json"
    if not corpus_path.exists():
        from pon_ml.data.gen_nlp import synthesize_literature

        lit = synthesize_literature(
            cfg.nlp.docs_per_topic, cfg.nlp.min_year, cfg.nlp.max_year, seed=cfg.seed
        )
        (base / "data/nlp").mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(lit).to_json(corpus_path, orient="records", lines=True)
    corpus = load_corpus(corpus_path)
    res = search("CAR-T intraperitoneal delivery", cfg.nlp.top_k, corpus)
    out = base / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "nlp_search.json").open("w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()

