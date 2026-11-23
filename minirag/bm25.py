"""BM25 稀疏检索索引（补图检索丢掉的字面精确匹配能力）。

MiniRAG 的图检索靠向量相似度找块，"普罗旺斯" 和 "布拉格" 在向量空间里很近，
但字面上完全不同。BM25 看的是词是否真的出现过，正好补这一块。

索引缓存在工作目录下的 bm25_index.pkl，块数没变就直接复用，不用重建。
"""

import json
import math
import os
import pickle
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
CACHE_NAME = "bm25_index.pkl"


def tokenize(text: str):
    return TOKEN.findall(str(text).lower())


class BM25Index:
    def __init__(self, chunk_ids, docs, k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = chunk_ids
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.lens = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        self.tf = [Counter(d) for d in self.docs]
        self.n = len(self.docs)
        df = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.idf = {
            w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def search(self, query: str, top_k: int = 10):
        terms = tokenize(query)
        if not terms or not self.n:
            return []
        scores = []
        for i in range(self.n):
            s = 0.0
            tf = self.tf[i]
            dl = self.lens[i] or 1
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                )
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunk_ids[i] for _, i in scores[:top_k]]

    # ---------- 缓存 ----------
    @classmethod
    def load_or_build(cls, working_dir: str):
        chunks_path = os.path.join(working_dir, "kv_store_text_chunks.json")
        cache_path = os.path.join(working_dir, CACHE_NAME)
        if not os.path.exists(chunks_path):
            return None
        with open(chunks_path, encoding="utf-8") as fh:
            chunks = json.load(fh)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    obj = pickle.load(fh)
                if getattr(obj, "n", -1) == len(chunks):
                    return obj
            except Exception:
                pass
        index = cls(list(chunks.keys()), [c["content"] for c in chunks.values()])
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(index, fh)
        except Exception:
            pass
        return index


def rrf_fuse(rank_lists, k: int = 60):
    """Reciprocal Rank Fusion：两路检索各给一个名次，融合成统一排序。"""
    scores = {}
    for ranking in rank_lists:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
