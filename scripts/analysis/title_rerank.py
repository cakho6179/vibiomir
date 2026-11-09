"""Rerank title/slug candidates with the passage cross-encoder on Kaggle GPU.

`lex_head2.parquet` is a high-recall, low-precision title/slug head: it reaches documents outside the
1,200-doc production pool, but generic query words ("buoi chieu", "bac si ke thuoc") can outrank the
condition phrase.  This stage keeps that candidate set and scores `(query, title_text)` with
BAAI/bge-reranker-v2-m3, the same cross-encoder used for passages.  The output is a title rank per
query, so the dynamic selector can fuse it with the passage rank as RRF instead of comparing raw logits.

    python title_rerank.py --data /kaggle/input/vibiomir-docstore-v21 \
        --cands /kaggle/input/vibiomir-lex-head2/lex_head2.parquet \
        --out /kaggle/working/title_rerank.parquet
"""
from __future__ import annotations

import argparse
import glob
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
WORD = re.compile(r"[0-9a-zA-Z]+")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("\u0111", "d").replace("\u0110", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def slug_text(url: str) -> str:
    s = str(url).split("?")[0].rstrip("/")
    s = re.sub(r"\.(html?|htm|php|aspx?|shtml)$", "", s, flags=re.I)
    s = s.split("/")[-1]
    return " ".join(WORD.findall(fold(re.sub(r"[-_+.]+", " ", s))))


def find_data(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    hits = sorted(glob.glob("/kaggle/input/**/docs_meta.parquet", recursive=True))
    if not hits:
        raise FileNotFoundError("docs_meta.parquet not found under /kaggle/input")
    return Path(hits[0]).parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--top", type=int, default=200, help="keep candidates with lex_head2 rank < top")
    ap.add_argument("--max-chars", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-queries", type=int, default=0)
    args = ap.parse_args()

    data = find_data(args.data)
    if not args.cands.exists():
        hits = sorted(glob.glob("/kaggle/input/**/lex_head2.parquet", recursive=True))
        if not hits:
            raise FileNotFoundError(f"candidate file not found: {args.cands}")
        args.cands = Path(hits[0])
    log(f"data={data} cands={args.cands}")
    meta = pd.read_parquet(data / "docs_meta.parquet", columns=["id", "url"])
    url_by = dict(zip(meta["id"].astype(int), meta["url"].astype(str)))

    hd: dict[int, list[str]] = {}
    for path in (data / "passages.parquet", data / "sections.parquet"):
        df = pd.read_parquet(path, columns=["doc_id", "heading"])
        s = df["heading"].fillna("").astype(str).str.strip()
        df = df[s != ""]
        for did, h in zip(df["doc_id"].astype(int), df["heading"].astype(str)):
            h = h.strip()
            if h:
                hd.setdefault(did, [])
                if len(hd[did]) < 2 and h not in hd[did]:
                    hd[did].append(h)
    log(f"headings for {len(hd):,} docs")

    cands = pd.read_parquet(args.cands)
    cands = cands[cands["rank"] < args.top].copy()
    q = pd.read_parquet(args.queries or (data / "query.parquet"))
    if args.max_queries:
        q = q.head(args.max_queries).copy()
        cands = cands[cands["qid"].isin(set(q["id"].astype(int)))]
    qtext = dict(zip(q["id"].astype(int), q["query"].astype(str)))

    need = set(cands["doc_id"].astype(int))
    title: dict[int, str] = {}
    for did in need:
        parts = list(hd.get(did, []))
        parts.append(slug_text(url_by.get(did, "")))
        t = " ".join(p for p in parts if p).strip()
        title[did] = t[: args.max_chars] or "_"
    log(f"candidates {len(cands):,}, unique docs {len(need):,}, queries {len(qtext):,}")

    pairs = [(int(qid), int(did), qtext[int(qid)], title[int(did)])
             for qid, did in zip(cands["qid"], cands["doc_id"])
             if int(qid) in qtext and int(did) in title]
    log(f"pairs {len(pairs):,}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    scores = np.empty(len(pairs), dtype=np.float32)
    for lo in range(0, len(pairs), args.batch):
        chunk = pairs[lo:lo + args.batch]
        enc = tok([p[2] for p in chunk], [p[3] for p in chunk], padding=True, truncation="longest_first",
                  max_length=args.max_len, return_tensors="pt").to(device)
        with torch.inference_mode():
            scores[lo:lo + len(chunk)] = model(**enc).logits[:, 0].float().cpu().numpy()
        if (lo // args.batch) % 200 == 0:
            log(f"scored {lo + len(chunk):,}/{len(pairs):,}")

    out = pd.DataFrame({
        "qid": [p[0] for p in pairs],
        "doc_id": [p[1] for p in pairs],
        "title_score": scores,
    })
    out["title_rank"] = out.groupby("qid")["title_score"].rank(ascending=False, method="first").astype(int) - 1
    out = out.sort_values(["qid", "title_rank"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    log(f"wrote {args.out}: {len(out):,} rows, {out.groupby('qid').size().mean():.1f}/query")

    for qid in (1, 33, 100, 137, 500, 844):
        sub = out[out.qid == qid].head(8)
        if sub.empty:
            continue
        print("=" * 100)
        print(f"Q{qid}: {qtext.get(qid, '')[:110]}")
        for _, x in sub.iterrows():
            did = int(x["doc_id"])
            print(f"  {x['title_score']:8.3f} | {slug_text(url_by.get(did, ''))[:70]} | "
                  f"{' / '.join(hd.get(did, []))[:55]}")


if __name__ == "__main__":
    main()
