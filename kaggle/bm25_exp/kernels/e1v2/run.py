"""ViBioMIR BM25 experiments E1/E2/E3 (Kaggle CPU kernel; also runs locally on a sample).

    python bm25_experiment.py EXP CORPUS_DIR OUT_DIR [--max-docs N] [--max-queries N]

EXP
  e1  document-level BM25, syllable tokens, diacritics KEPT, full text (title + body)
  e2  chunk-level BM25 (~1200-char passages, title on the first one); doc score = max chunk
  e3  e1 with word segmentation: pyvi for Vietnamese, jieba for Chinese

Scoring is Lucene BM25 (k1=1.5, b=0.75) computed directly on a sparse count matrix, so a
1M-document full-text index fits in a 30 GB kernel. Outputs in OUT_DIR:
  ranking.parquet (qid, rank, doc_id, score; top 200), stats.json,
  submission_<exp>_n{28,45,70}.zip  (1 chunk = whole document text, uncapped)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

K1, B = 1.5, 0.75
TOPK = 200
NS = (28, 45, 70)
CHUNK_CHARS = 1200
MAX_CHUNK_TEXT = None  # whole-document chunk, no cap (user decision 2026-10-20)
TOKEN_PATTERN = r"(?u)\b\w+\b"
CJK_RANGE = f"{chr(0x4E00)}-{chr(0x9FFF)}"  # built from code points: keep this file pure ASCII
CJK = re.compile(f"[{CJK_RANGE}]")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def iter_corpus(corpus_dir: Path, max_docs: int | None):
    n = 0
    # every part-* below corpus_dir (base corpus + delta datasets); Kaggle stores uploaded .gz decompressed
    for part in sorted(Path(p) for p in glob.glob(f"{corpus_dir}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.suffix == ".gz" else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield d["id"], d["t"], d["x"]
                n += 1
                if n % 100_000 == 0:
                    log(f"  read {n:,} docs")
                if max_docs and n >= max_docs:
                    return


CJK_CHAR = re.compile(f"([{CJK_RANGE}])")
WORD = re.compile(TOKEN_PATTERN)


def prep_syllables(text: str) -> str:
    """e1/e2: NFC lowercase; each CJK character becomes its own token (bounded vocabulary)."""
    text = norm(text)
    return CJK_CHAR.sub(r" \1 ", text) if CJK.search(text) else text


_SEG = None


def segment(text: str) -> str:
    """e3: pyvi word segmentation (VI) or jieba (ZH); returns space-joined word tokens."""
    global _SEG
    if _SEG is None:
        from pyvi import ViTokenizer
        import jieba
        jieba.setLogLevel(60)
        _SEG = (ViTokenizer.tokenize, jieba.lcut)
    text = norm(text)
    if CJK.search(text[:2000]):
        return " ".join(w for w in _SEG[1](text) if WORD.fullmatch(w))
    return " ".join(WORD.findall(_SEG[0](text)))


def vectorizer(exp: str) -> CountVectorizer:
    if exp == "e3":  # input is pre-segmented by segment()
        return CountVectorizer(lowercase=False, token_pattern=r"\S+", dtype=np.int32)
    return CountVectorizer(preprocessor=prep_syllables, token_pattern=TOKEN_PATTERN, dtype=np.int32)


def parallel_segment(texts, workers: int):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        yield from pool.imap(segment, texts, chunksize=64)


def units(exp: str, corpus_dir: Path, max_docs: int | None, doc_ids: list[int], unit_doc: list[int]):
    """Yield index units (documents or passages); record the owning doc row of each."""
    for row, (doc_id, title, text) in enumerate(iter_corpus(corpus_dir, max_docs)):
        doc_ids.append(doc_id)
        if exp != "e2":
            unit_doc.append(row)
            yield f"{title}\n{text}"
            continue
        buf = title
        for para in text.split("\n"):
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                unit_doc.append(row)
                yield buf
                buf = ""
            buf = f"{buf}\n{para}" if buf else para
        if buf:
            unit_doc.append(row)
            yield buf


def bm25_weights(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Lucene BM25 term weights per (unit, term)."""
    counts = counts.tocsr()
    n = counts.shape[0]
    dl = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean())
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm_len = (K1 * (1 - B + B * dl / avgdl)).astype(np.float32)
    denom = norm_len[np.repeat(np.arange(n, dtype=np.int32), np.diff(counts.indptr))]
    data = counts.data.astype(np.float32)  # tf; computed in place to keep peak memory low
    denom += data
    data *= K1 + 1
    data /= denom
    del denom
    data *= idf[counts.indices]
    return sp.csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp", choices=["e1", "e2", "e3"])
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    corpus_dir = args.corpus_dir / "corpus" if (args.corpus_dir / "corpus").exists() else args.corpus_dir
    qpath = args.queries or next(args.corpus_dir.rglob("query.parquet"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids: list[int] = []
    unit_doc: list[int] = []
    vec = vectorizer(args.exp)
    log(f"{args.exp}: vectorizing ...")
    stream = units(args.exp, corpus_dir, args.max_docs, doc_ids, unit_doc)
    if args.exp == "e3":
        stream = parallel_segment(stream, args.workers)
    counts = vec.fit_transform(stream)
    unit_doc_arr = np.asarray(unit_doc, dtype=np.int32)
    log(f"units={counts.shape[0]:,} docs={len(doc_ids):,} vocab={counts.shape[1]:,} nnz={counts.nnz:,}")
    weights_t = bm25_weights(counts).T.tocsr()  # (vocab, units)
    del counts
    log("weights ready")

    queries = pd.read_parquet(qpath)
    if args.max_queries:
        queries = queries.head(args.max_queries)
    qtexts = queries["query"].tolist()
    if args.exp == "e3":
        qtexts = [segment(q) for q in qtexts]
    q_counts = vec.transform(qtexts).astype(np.float32)
    starts = np.flatnonzero(np.r_[True, unit_doc_arr[1:] != unit_doc_arr[:-1]])
    owners = unit_doc_arr[starts]
    rows = []
    for lo in range(0, q_counts.shape[0], 32):
        scores = (q_counts[lo:lo + 32] @ weights_t).toarray()
        for j, s in enumerate(scores):
            doc_score = np.maximum.reduceat(s, starts) if args.exp == "e2" else s
            top = np.argpartition(-doc_score, min(TOPK, len(doc_score) - 1))[:TOPK]
            top = top[np.argsort(-doc_score[top])]
            qid = int(queries["id"].iloc[lo + j])
            for rank, idx in enumerate(top):
                row = owners[idx] if args.exp == "e2" else idx
                if doc_score[idx] > 0:
                    rows.append((qid, rank, doc_ids[row], float(doc_score[idx])))
        log(f"scored {min(lo + 32, q_counts.shape[0])}/{q_counts.shape[0]} queries")
    ranking = pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score"])
    ranking.to_parquet(args.out_dir / "ranking.parquet", index=False)

    need = set(ranking.loc[ranking["rank"] < max(NS), "doc_id"])
    texts = {}
    for doc_id, title, text in iter_corpus(corpus_dir, args.max_docs):
        if doc_id in need:
            texts[doc_id] = f"{title}\n{text}"[:MAX_CHUNK_TEXT]
    log(f"collected {len(texts):,} document texts for chunks")
    by_q = {q: g.sort_values("rank")["doc_id"].tolist() for q, g in ranking.groupby("qid")}
    stats = {"exp": args.exp, "units": int(len(unit_doc)), "docs": len(doc_ids),
             "vocab": int(weights_t.shape[0]), "k1": K1, "b": B, "seconds": round(time.time() - T0)}
    for n in NS:
        records = []
        for qid in queries["id"].tolist():
            top = by_q.get(int(qid), [])[:n]
            records.append({"id": int(qid), "relevant_docs": [int(d) for d in top],
                            "relevant_chunks": [{"doc_id": int(d), "chunk_text": texts[d]} for d in top]})
        path = args.out_dir / f"submission_{args.exp}_n{n}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("submission.json", json.dumps(records, ensure_ascii=False))
        stats[f"zip_mb_n{n}"] = round(path.stat().st_size / 1e6, 1)
        log(f"wrote {path.name} ({stats[f'zip_mb_n{n}']} MB)")
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log("done " + json.dumps(stats))


if __name__ == "__main__":
    import subprocess
    import sys
    EXP = "e1"
    if EXP == "e3":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyvi", "jieba"], check=True)
    import glob
    query = glob.glob("/kaggle/input/**/query.parquet", recursive=True)[0]
    sys.argv = ["bm25", EXP, "/kaggle/input", "/kaggle/working", "--queries", str(query)]
    main()
