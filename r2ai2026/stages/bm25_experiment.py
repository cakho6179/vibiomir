"""ViBioMIR BM25 experiments (Kaggle CPU kernel; also runs locally on a sample).

    python bm25_experiment.py EXP CORPUS_DIR OUT_DIR [--max-docs N] [--max-queries N]
        [--k1 1.5] [--b 0.75] [--chunk-chars N] [--overlap N] [--seg] [--vi-only] [--ns 28]

EXP is a label; e1/e2/e3 are presets, any other label uses only the flags:
  e1  document-level BM25, syllable tokens, diacritics KEPT, full text (title + body)
  e2  chunk-level BM25 (~1200-char passages, title on the first one); doc score = max chunk
  e3  e1 with word segmentation: pyvi for Vietnamese, jieba for Chinese
Flags: --chunk-chars > 0 indexes overlapping passages (--overlap chars carried over, title on
every passage) with doc score = max passage; --seg uses pyvi/jieba word segmentation;
--vi-only indexes only Vietnamese ids (<= 766441); --clean indexes clean_v4 text; --sections MAX,MIN
indexes heading-aware section chunks (both from r2ai2026/stages/docorg.py; doc score = max chunk).

Scoring is Lucene BM25 (k1=1.5, b=0.75) computed directly on a sparse count matrix, so a
1M-document full-text index fits in a 30 GB kernel. Outputs in OUT_DIR:
  ranking.parquet (qid, rank, doc_id, score; top --topk, default 200), stats.json,
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

from docorg import clean_v4, section_chunks  # sibling file here; the kernel wrapper ships it next to this script

K1, B = 1.5, 0.75
TOPK = 200
NS = (28, 45, 70)
CHUNK_CHARS = 1200
VI_MAX_ID = 766441  # official id blocks: Vietnamese ids come first
MAX_CHUNK_TEXT = None  # whole-document chunk, no cap (user decision 2026-10-20)
TOKEN_PATTERN = r"(?u)\b\w+\b"
CJK_RANGE = f"{chr(0x4E00)}-{chr(0x9FFF)}"  # built from code points: keep this file pure ASCII
CJK = re.compile(f"[{CJK_RANGE}]")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def iter_corpus(corpus_dir: Path, max_docs: int | None, vi_only: bool = False):
    n = 0
    # every part-* below corpus_dir (base corpus + delta datasets); Kaggle stores uploaded .gz decompressed
    for part in sorted(Path(p) for p in glob.glob(f"{corpus_dir}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.suffix == ".gz" else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if vi_only and d["id"] > VI_MAX_ID:
                    continue
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


def vectorizer(seg: bool) -> CountVectorizer:
    if seg:  # input is pre-segmented by segment()
        return CountVectorizer(lowercase=False, token_pattern=r"\S+", dtype=np.int32)
    return CountVectorizer(preprocessor=prep_syllables, token_pattern=TOKEN_PATTERN, dtype=np.int32)


def parallel_segment(texts, workers: int):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        yield from pool.imap(segment, texts, chunksize=64)


def overlapping_passages(title: str, text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-packed passages of <= size chars; each new one starts with the last `overlap`
    chars of the previous (cut at a space), like LangChain's RecursiveCharacterTextSplitter."""
    pieces = []
    for para in text.split("\n"):
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(para[:cut])
            para = para[cut:].lstrip()
        if para.strip():
            pieces.append(para)
    out, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 1 > size:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            sp_ = tail.find(" ")
            buf = tail[sp_ + 1:] if 0 <= sp_ < len(tail) - 1 else tail
        buf = f"{buf}\n{p}" if buf else p
    if buf or not out:
        out.append(buf)
    return [f"{title}\n{p}".strip() for p in out]


def units(args, corpus_dir: Path, doc_ids: list[int], unit_doc: list[int]):
    """Yield index units (documents or passages); record the owning doc row of each."""
    for row, (doc_id, title, text) in enumerate(iter_corpus(corpus_dir, args.max_docs, args.vi_only)):
        doc_ids.append(doc_id)
        if args.clean:  # index side only: submissions keep the original text
            text = clean_v4(text)
        if args.sections:
            for p in section_chunks(title, text, *args.sections):
                unit_doc.append(row)
                yield p
            continue
        if args.chunk_chars > 0:
            for p in overlapping_passages(title, text, args.chunk_chars, args.overlap):
                unit_doc.append(row)
                yield p
            continue
        if args.exp != "e2":
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


def bm25_weights(counts: sp.csr_matrix, k1: float = K1, b: float = B) -> sp.csr_matrix:
    """Lucene BM25 term weights per (unit, term)."""
    counts = counts.tocsr()
    n = counts.shape[0]
    dl = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean())
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm_len = (k1 * (1 - b + b * dl / avgdl)).astype(np.float32)
    denom = norm_len[np.repeat(np.arange(n, dtype=np.int32), np.diff(counts.indptr))]
    data = counts.data.astype(np.float32)  # tf; computed in place to keep peak memory low
    denom += data
    data *= k1 + 1
    data /= denom
    del denom
    data *= idf[counts.indices]
    return sp.csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp", help="label; e1/e2/e3 are presets")
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--topk", type=int, default=TOPK, help="docs kept per query")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k1", type=float, default=K1)
    ap.add_argument("--b", type=float, default=B)
    ap.add_argument("--chunk-chars", type=int, default=0, help="> 0: overlapping passages, doc = max")
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--seg", action="store_true", help="pyvi / jieba word segmentation")
    ap.add_argument("--vi-only", action="store_true")
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--clean", action="store_true", help="index clean_v4 text (docorg.py)")
    ap.add_argument("--sections", default="", help="MAX,MIN chars: heading-aware section chunks (docorg.py)")
    args = ap.parse_args()
    args.seg = args.seg or args.exp == "e3"
    args.sections = tuple(int(x) for x in args.sections.split(",")) if args.sections else ()
    chunked = args.chunk_chars > 0 or args.exp == "e2" or bool(args.sections)
    ns = [int(n) for n in args.ns.split(",") if n]
    corpus_dir = args.corpus_dir / "corpus" if (args.corpus_dir / "corpus").exists() else args.corpus_dir
    qpath = args.queries
    if qpath is None or not qpath.exists():  # a bare file name: find it under corpus_dir (follows symlinks)
        name = qpath.name if qpath else "query.parquet"
        qpath = Path(glob.glob(f"{args.corpus_dir}/**/{name}", recursive=True)[0])
    log(f"queries: {qpath}; k1={args.k1} b={args.b} chunk={args.chunk_chars}/{args.overlap} "
        f"seg={args.seg} vi_only={args.vi_only}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids: list[int] = []
    unit_doc: list[int] = []
    vec = vectorizer(args.seg)
    log(f"{args.exp}: vectorizing ...")
    stream = units(args, corpus_dir, doc_ids, unit_doc)
    if args.seg:
        stream = parallel_segment(stream, args.workers)
    counts = vec.fit_transform(stream)
    unit_doc_arr = np.asarray(unit_doc, dtype=np.int32)
    log(f"units={counts.shape[0]:,} docs={len(doc_ids):,} vocab={counts.shape[1]:,} nnz={counts.nnz:,}")
    weights_t = bm25_weights(counts, args.k1, args.b).T.tocsr()  # (vocab, units)
    del counts
    log("weights ready")

    queries = pd.read_parquet(qpath)
    if args.max_queries:
        queries = queries.head(args.max_queries)
    qtexts = queries["query"].tolist()
    if args.seg:
        qtexts = [segment(q) for q in qtexts]
    q_counts = vec.transform(qtexts).astype(np.float32)
    starts = np.flatnonzero(np.r_[True, unit_doc_arr[1:] != unit_doc_arr[:-1]])
    owners = unit_doc_arr[starts]
    rows = []
    for lo in range(0, q_counts.shape[0], 32):
        scores = (q_counts[lo:lo + 32] @ weights_t).toarray()
        for j, s in enumerate(scores):
            doc_score = np.maximum.reduceat(s, starts) if chunked else s
            top = np.argpartition(-doc_score, min(args.topk, len(doc_score) - 1))[:args.topk]
            top = top[np.argsort(-doc_score[top])]
            qid = int(queries["id"].iloc[lo + j])
            for rank, idx in enumerate(top):
                row = owners[idx] if chunked else idx
                if doc_score[idx] > 0:
                    rows.append((qid, rank, doc_ids[row], float(doc_score[idx])))
        log(f"scored {min(lo + 32, q_counts.shape[0])}/{q_counts.shape[0]} queries")
    ranking = pd.DataFrame(rows, columns=["qid", "rank", "doc_id", "score"])
    ranking.to_parquet(args.out_dir / "ranking.parquet", index=False)

    stats = {"exp": args.exp, "units": int(len(unit_doc)), "docs": len(doc_ids), "vocab": int(weights_t.shape[0]),
             "k1": args.k1, "b": args.b, "chunk_chars": args.chunk_chars, "overlap": args.overlap,
             "seg": args.seg, "vi_only": args.vi_only, "clean": args.clean, "sections": list(args.sections)}
    if ns:
        need = set(ranking.loc[ranking["rank"] < max(ns), "doc_id"])
        texts = {}
        for doc_id, title, text in iter_corpus(corpus_dir, args.max_docs, args.vi_only):
            if doc_id in need:
                texts[doc_id] = f"{title}\n{text}"[:MAX_CHUNK_TEXT]
        log(f"collected {len(texts):,} document texts for chunks")
        by_q = {q: g.sort_values("rank")["doc_id"].tolist() for q, g in ranking.groupby("qid")}
        for n in ns:
            records = []
            for qid in queries["id"].tolist():
                top = by_q.get(int(qid), [])[:n]
                records.append({"id": int(qid), "relevant_docs": [int(d) for d in top],
                                "relevant_chunks": [{"doc_id": int(d), "chunk_text": texts[d]} for d in top]})
            path = args.out_dir / f"submission_{args.exp}_n{n}.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                z.writestr("submission.json", json.dumps(records, ensure_ascii=False))
            stats[f"zip_mb_n{n}"] = round(path.stat().st_size / 1e6, 1)
            log(f"wrote {path.name} ({stats[f'zip_mb_n{n}']} MB)")
    stats["seconds"] = round(time.time() - T0)
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log("done " + json.dumps(stats))


if __name__ == "__main__":
    main()
