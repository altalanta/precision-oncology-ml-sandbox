from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.df: Dict[str, int] = defaultdict(int)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        for d in docs:
            for term in set(d):
                self.df[term] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()}

    def score(self, q: List[str], idx: int) -> float:
        dl = self.doc_len[idx]
        c = Counter(self.docs[idx])
        score = 0.0
        for term in q:
            if term not in c:
                continue
            idf = self.idf.get(term, 0.0)
            tf = c[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def query(self, q: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

