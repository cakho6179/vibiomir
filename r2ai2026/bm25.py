"""BM25 indexes (document level and chunk level) built on ``bm25s``.

Tuning knobs are exposed as data so every run records exactly which index
produced a submission: ``k1``/``b``, tokenizer variant and the id array mapping
row positions back to the organizers' document ids.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np

from r2ai2026.text import tokenize


@dataclass(slots=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    keep_diacritics: bool = False


class BM25Index:
    """Sparse lexical index over arbitrary integer row descriptors."""

    def __init__(self, params: BM25Params | None = None, *, name: str = "bm25") -> None:
        self.params = params or BM25Params()
        self.name = name
        self.rows: np.ndarray = np.zeros(0, dtype=np.int64)
        self._retriever: bm25s.BM25 | None = None

    # ------------------------------------------------------------------ build
    def build(self, rows: Sequence[object] | np.ndarray, texts: Iterable[str], *, show_progress: bool = False) -> int:
        row_array = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64)
        tokenized = [tokenize(text, keep_diacritics=self.params.keep_diacritics) for text in texts]
        keep = [index for index, tokens in enumerate(tokenized) if tokens]
        if len(keep) != len(tokenized):
            tokenized = [tokenized[index] for index in keep]
            row_array = row_array[keep]
        if not tokenized:
            raise ValueError("cannot build an index with no tokens")
        retriever = bm25s.BM25(k1=self.params.k1, b=self.params.b, method=self.params.method)
        retriever.index(tokenized, show_progress=show_progress)
        self._retriever = retriever
        self.rows = np.atleast_2d(row_array)
        if self.rows.shape[0] != len(tokenized):
            raise RuntimeError(f"row/text mismatch: {self.rows.shape[0]} rows vs {len(tokenized)} texts")
        return len(tokenized)

    # ----------------------------------------------------------------- search
    def _ensure_ready(self) -> bm25s.BM25:
        if self._retriever is None:
            raise RuntimeError("index is not built or loaded")
        return self._retriever

    def search(self, query: str, k: int = 100) -> list[tuple[np.ndarray, float]]:
        """Return ``[(row_descriptor, score), ...]`` best first."""
        return self.search_batch([query], k=k)[0]

    def search_batch(self, queries: Sequence[str], k: int = 100, *, n_threads: int = 0) -> list[list[tuple[np.ndarray, float]]]:
        retriever = self._ensure_ready()
        tokenized = [tokenize(query, keep_diacritics=self.params.keep_diacritics) for query in queries]
        tokenized = [tokens if tokens else ["__empty__"] for tokens in tokenized]
        k = max(1, min(int(k), self.rows.shape[0]))
        results, scores = retriever.retrieve(
            tokenized, k=k, show_progress=False, n_threads=n_threads, sorted=True, leave_progress=False
        )
        output: list[list[tuple[np.ndarray, float]]] = []
        for query_index in range(results.shape[0]):
            hits: list[tuple[np.ndarray, float]] = []
            for rank in range(results.shape[1]):
                position = int(results[query_index, rank])
                if position < 0 or position >= self.rows.shape[0]:
                    continue
                hits.append((self.rows[position], float(scores[query_index, rank])))
            output.append(hits)
        return output

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        retriever = self._ensure_ready()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        retriever.save(str(directory / f"{self.name}.bm25"), show_progress=False)
        np.save(directory / f"{self.name}.rows.npy", self.rows)
        (directory / f"{self.name}.params.json").write_text(
            json.dumps(asdict(self.params), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, name: str = "bm25", mmap: bool = True) -> BM25Index:
        directory = Path(directory)
        params_path = directory / f"{name}.params.json"
        params = BM25Params(**json.loads(params_path.read_text(encoding="utf-8"))) if params_path.exists() else BM25Params()
        index = cls(params, name=name)
        index.rows = np.load(directory / f"{name}.rows.npy")
        index._retriever = bm25s.BM25.load(str(directory / f"{name}.bm25"), load_corpus=False, mmap=mmap)
        return index

    @property
    def size(self) -> int:
        return int(self.rows.shape[0])


def rank_ids(hits: Sequence[tuple[np.ndarray, float]]) -> list[int]:
    """Document ids from hits whose descriptors are ``[doc_id]``."""
    return [int(hit[0][0]) for hit in hits]


def rank_rows(hits: Sequence[tuple[np.ndarray, float]]) -> list[tuple[int, ...]]:
    """Raw row descriptors, for indexes that carry more than the document id."""
    return [tuple(int(value) for value in hit[0]) for hit in hits]
