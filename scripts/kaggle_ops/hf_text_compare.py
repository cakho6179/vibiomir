"""How far does our v2.1 text differ from the tmh1008/ViBioMIR-crawled-corpus text of the same doc id?

    python hf_text_compare.py --input DIR --out DIR [--revision 18cb5a5...] [--save-text]

Motivation: ~22% of the chunks of gold docs fail even when long enough (runs/reports/chunk_geometry); one candidate
cause is that the gold chunk text comes from a different crawl than ours. External data, provenance: Hugging Face
dataset tmh1008/ViBioMIR-crawled-corpus at the pinned revision (public, no license declared; audited in
docs/EXTERNAL_HF_TMH_AUDIT_2026-10-25.md); used for text only, never as labels.

DIR holds the docstore v2.1 parts (part-*.jsonl[.gz]) and cands.parquet (the pool; its doc ids are compared).
Texts are normalised (NFKC, lower case, whitespace collapsed). Per doc: lengths, and the fraction of five 120-char
probes taken evenly from each text that occur verbatim in the other (hf_in_v21: HF probes found in v2.1 text,
v21_in_hf: the reverse). Output compare.parquet (doc_id, lang, len_v21, len_hf, hf_in_v21, v21_in_hf, hf_success)
+ summary.json; --save-text also writes hf_text.parquet (doc_id, title, content) for the pool docs.
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

T0 = time.time()
REPO = "tmh1008/ViBioMIR-crawled-corpus"
VI_MAX, ZH_MAX = 766441, 4392214
WS = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def norm(t: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFKC", t or "").lower()).strip()


def probes(t: str, k: int = 5, n: int = 120) -> list[str]:
    if len(t) <= n:
        return [t] if t else []
    return [t[int(i):int(i) + n] for i in np.linspace(0, len(t) - n, k)]


def found(a: str, b: str) -> float:
    ps = probes(a)
    return sum(p in b for p in ps) / len(ps) if ps else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default="18cb5a544322d8d7b903ca2b6ceb3138628223b4")
    ap.add_argument("--save-text", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    cands = pd.read_parquet(sorted(glob.glob(f"{args.input}/**/cands.parquet", recursive=True))[0], columns=["doc_id"])
    want = set(cands["doc_id"].astype(int))
    log(f"{len(want):,} pool docs")
    root = snapshot_download(REPO, repo_type="dataset", revision=args.revision, allow_patterns=["*.parquet"],
                             local_dir="/tmp/hf_tmh")
    files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    log(f"HF: {len(files)} parquet files")
    hf = {}
    for f in files:
        t = pd.read_parquet(f, columns=["id", "title", "content", "success"])
        t = t[t["id"].isin(want)]
        for i, title, content, ok in zip(t["id"], t["title"], t["content"], t["success"]):
            hf[int(i)] = (title or "", content or "", bool(ok))
    log(f"HF rows for pool docs: {len(hf):,}")
    rows, keep = [], []
    for part in sorted(glob.glob(f"{args.input}/**/part-*.jsonl*", recursive=True)):
        opener = gzip.open if part.endswith(".gz") else open
        with opener(part, "rt", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                i = d["id"]
                if i not in want:
                    continue
                v = norm(d.get("x") or "")
                title, content, ok = hf.get(i, ("", "", False))
                h = norm(content)
                rows.append((i, len(v), len(h), found(h, v), found(v, h), ok))
                if args.save_text and i in hf:
                    keep.append((i, title, content))
        log(f"{part}: {len(rows):,} compared")
    df = pd.DataFrame(rows, columns=["doc_id", "len_v21", "len_hf", "hf_in_v21", "v21_in_hf", "hf_success"])
    df["lang"] = np.where(df["doc_id"] <= VI_MAX, "vi", np.where(df["doc_id"] <= ZH_MAX, "zh", "en"))
    df.to_parquet(args.out / "compare.parquet", index=False)
    if keep:
        pd.DataFrame(keep, columns=["doc_id", "title", "content"]).to_parquet(args.out / "hf_text.parquet", index=False)
    summ = {}
    for lang, g in df.groupby("lang"):
        has = g[g["len_hf"] > 0]
        summ[lang] = {"docs": len(g), "with_hf_text": len(has),
                      "same_text(both>=0.8)": float(((has["hf_in_v21"] >= .8) & (has["v21_in_hf"] >= .8)).mean()),
                      "hf_longer_1.2x": float((has["len_hf"] > 1.2 * has["len_v21"]).mean()),
                      "hf_shorter_0.8x": float((has["len_hf"] < 0.8 * has["len_v21"]).mean()),
                      "median_len_ratio": float((has["len_hf"] / has["len_v21"].clip(lower=1)).median())}
    (args.out / "summary.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
