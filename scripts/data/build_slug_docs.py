"""Pseudo-docs from URL slugs for Vietnamese links with no text in the docstore.

    python scripts/data/build_slug_docs.py [--store data/docstore/v1]

9.9k VI links were never crawled; their slug ("benh-gout-chan-doan-va-dieu-tri") is still a title without
diacritics. Writes slug_docs.parquet (id, url, host, slug) so retrieval can match them (strip diacritics on both
sides) and list them in relevant_docs; they never get chunks (no text). Idea from TranVu2005/r2ai-stage3.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VI_MAX_ID = 766_441


def url_slug(url: str) -> str:
    last = unquote(urlsplit(url).path).rstrip("/").split("/")[-1]
    last = re.sub(r"\.(html?|aspx?|php)$", "", last)
    last = re.sub(r"[-_+]+", " ", last)
    last = re.sub(r"\b(?:\d{5,}|id\d+|[dp]\d+)\b", " ", last)  # numeric/article ids carry no meaning
    return " ".join(last.split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=ROOT / "data" / "docstore" / "v1")
    ap.add_argument("--links", type=Path, default=ROOT / "data" / "raw" / "btc" / "links_corpus_vi_en_zh.parquet")
    args = ap.parse_args()

    links = pd.read_parquet(args.links)
    have = set(pd.read_parquet(args.store / "docs_meta.parquet", columns=["id"])["id"])
    miss = links[(links["id"] <= VI_MAX_ID) & ~links["id"].isin(have)].copy()
    miss["host"] = miss["url"].map(lambda u: urlsplit(u).netloc.lower())
    miss["slug"] = miss["url"].map(url_slug)
    miss = miss[miss["slug"].str.split().str.len() >= 2]
    miss[["id", "url", "host", "slug"]].to_parquet(args.store / "slug_docs.parquet", index=False)

    man_path = args.store / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["slug_docs"] = {"docs": len(miss), "hosts": int(miss["host"].nunique())}
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(man["slug_docs"]))


if __name__ == "__main__":
    main()
