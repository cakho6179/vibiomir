"""Build a persisted BM25 index over docstore sections, then rank the queries once.

    python build_section_index.py --data DIR --out DIR [--k1 1.5] [--b 0.75]

DIR is searched for sections.parquet and part-*.jsonl* (the docstore v1 layout).
The index is a hashed sparse term matrix (no vocabulary file to keep in sync):
  weights.npz   float32 (n_features, n_sections), BM25 weights, transposed for querying
  ids.npy       int64 (n_sections, 2): doc_id, section id
  ranking.parquet  qid, rank, doc_id, score (section score -> best section per doc)
  meta.json     hashing/tokenizer/BM25 parameters and counts
Query time only needs the same HashingVectorizer settings and weights.npz.
Source is ASCII-only (Kaggle decodes kernel source badly otherwise).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

TOKEN_PATTERN = r"(?u)\b\w+\b"
CJK_RANGE = f"{chr(0x4E00)}-{chr(0x9FFF)}"
CJK = re.compile(f"[{CJK_RANGE}]")
CJK_CHAR = re.compile(f"([{CJK_RANGE}])")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def prep(text: str) -> str:
    """NFC lowercase; each CJK character becomes its own token (bounded vocabulary)."""
    text = norm(text)
    return CJK_CHAR.sub(r" \1 ", text) if CJK.search(text) else text


def find_docstore(data: Path) -> Path:
    hits = sorted(glob.glob(f"{data}/**/sections.parquet", recursive=True))
    if not hits:
        raise SystemExit(f"sections.parquet not found under {data}")
    return Path(hits[0]).parent


def corpus_parts(root: Path) -> list[Path]:
    parts = sorted(Path(p) for p in glob.glob(f"{root}/part-*.jsonl*"))
    if not parts:
        parts = sorted(Path(p) for p in glob.glob(f"{root}/**/part-*.jsonl*", recursive=True))
    if not parts:
        raise SystemExit(f"no corpus parts under {root}")
    return parts


def section_map(root: Path) -> dict[int, list[tuple[int, int, int, str]]]:
    sections = pd.read_parquet(root / "sections.parquet", columns=["doc_id", "sid", "start", "end", "heading"])
    out: dict[int, list[tuple[int, int, int, str]]] = {}
    for row in sections.itertuples(index=False):
        out.setdefault(int(row.doc_id), []).append(
            (int(row.sid), int(row.start), int(row.end), str(row.heading or ""))
        )
    for items in out.values():
        items.sort()
    return out


def iter_units(root: Path, max_sections: int = 0):
    """Yield (doc_id, sid, text) in corpus part order."""
    by_doc = section_map(root)
    n, seen_docs, missing = 0, 0, 0
    for part in corpus_parts(root):
        opener = gzip.open if part.suffix == ".gz" else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                did = int(d["id"])
                items = by_doc.pop(did, None)
                if not items:
                    missing += 1
                    continue
                title = str(d.get("t") or "")
                body = str(d.get("x") or "")
                for sid, start, end, heading in items:
                    text = "\n".join(x for x in (title, heading, body[start:end]) if x).strip()
                    yield did, sid, text
                    n += 1
                    if max_sections and n >= max_sections:
                        log(f"stopped at {n:,} sections; {len(by_doc):,} docs not read")
                        return
                seen_docs += 1
                if seen_docs % 100_000 == 0:
                    log(f"read {seen_docs:,} docs / {n:,} sections")
    log(f"done reading: docs {seen_docs:,}, sections {n:,}, docs without sections {missing:,}, left {len(by_doc):,}")


def bm25_weights(counts: sp.csr_matrix, k1: float, b: float) -> sp.csr_matrix:
    counts = counts.tocsr()
    n = counts.shape[0]
    dl = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean())
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm_len = (k1 * (1 - b + b * dl / avgdl)).astype(np.float32)
    denom = norm_len[np.repeat(np.arange(n, dtype=np.int32), np.diff(counts.indptr))]
    data = counts.data.astype(np.float32)
    denom += data
    data *= k1 + 1
    data /= denom
    del denom
    data *= idf[counts.indices]
    return sp.csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--features", type=int, default=1 << 20)
    ap.add_argument("--topk-docs", type=int, default=300)
    ap.add_argument("--topk-sections", type=int, default=2000)
    ap.add_argument("--max-sections", type=int, default=0)
    ap.add_argument("--skip-ranking", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = find_docstore(args.data)
    log(f"docstore {root}")

    vec = HashingVectorizer(n_features=args.features, alternate_sign=False, norm=None, lowercase=False,
                            preprocessor=prep, token_pattern=TOKEN_PATTERN, dtype=np.int32)
    blocks, ids, buf, buf_ids = [], [], [], []
    for did, sid, text in iter_units(root, args.max_sections):
        buf.append(text)
        buf_ids.append((did, sid))
        if len(buf) >= 20_000:
            blocks.append(vec.transform(buf))
            ids.extend(buf_ids)
            buf, buf_ids = [], []
            log(f"vectorized {len(ids):,} sections, nnz={sum(x.nnz for x in blocks):,}")
    if buf:
        blocks.append(vec.transform(buf))
        ids.extend(buf_ids)
    if not blocks:
        raise SystemExit("no sections to index")
    counts = sp.vstack(blocks, format="csr")
    del blocks, buf, buf_ids
    ids_arr = np.asarray(ids, dtype=np.int64)
    log(f"sections {counts.shape[0]:,}, features {counts.shape[1]:,}, nnz {counts.nnz:,}, docs {len(np.unique(ids_arr[:,0])):,}")

    weights = bm25_weights(counts, args.k1, args.b)
    del counts
    weights_t = weights.T.tocsr()
    del weights
    log(f"weights {weights_t.shape}, nnz {weights_t.nnz:,}")
    sp.save_npz(args.out / "weights.npz", weights_t, compressed=True)
    np.save(args.out / "ids.npy", ids_arr)
    meta = {"sections": int(ids_arr.shape[0]), "docs": int(len(np.unique(ids_arr[:, 0]))),
            "features": int(args.features), "nnz": int(weights_t.nnz), "k1": args.k1, "b": args.b,
            "token_pattern": TOKEN_PATTERN, "prep": "NFC lowercase; CJK chars split",
            "sections_file": str(root / "sections.parquet")}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"saved index for {ids_arr.shape[0]:,} sections")

    if not args.skip_ranking:
        qhits = sorted(glob.glob(f"{args.data}/**/query.parquet", recursive=True))
        if not qhits:
            log("query.parquet not found; index saved without ranking")
            return
        queries = pd.read_parquet(qhits[0])
        qtexts = queries["query"].astype(str).tolist()
        q = vec.transform(qtexts).astype(np.float32)
        n_sec = ids_arr.shape[0]
        rows = []
        for lo in range(0, q.shape[0], 16):
            scores = (q[lo:lo + 16] @ weights_t).toarray()
            for j in range(scores.shape[0]):
                s = scores[j]
                take = min(args.topk_sections, n_sec)
                top = np.argpartition(-s, take - 1)[:take]
                top = top[np.argsort(-s[top])]
                qid = int(queries["id"].iloc[lo + j])
                seen = set()
                for idx in top:
                    doc = int(ids_arr[idx, 0])
                    if doc in seen:
                        continue
                    seen.add(doc)
                    rows.append((qid, len(seen) - 1, doc, float(s[idx])))
                    if len(seen) >= args.topk_docs:
                        break
            log(f"ranked {min(lo + 16, q.shape[0]):,}/{q.shape[0]:,} queries")
        pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score"]).to_parquet(
            args.out / "ranking.parquet", index=False)
        meta["queries"] = int(q.shape[0])
        meta["topk_docs"] = args.topk_docs
        (args.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
