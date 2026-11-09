"""Audit an external crawled corpus against docstore v2 and the official link list.

This is read-only: it never changes the docstore.  It answers the only questions that
matter before merging an external crawl:

  * how many ids does it cover that v2 lacks?
  * are those ids in the official link list (i.e. can they ever be positive)?
  * do the missing rows actually have usable content (success flag, length, junk gate)?
  * how many are exact-content duplicates of v2 or of each other?
  * which hosts / languages would the recovery affect?

The external parquet schema is expected to contain at least:
  id, url, domain, status_code, success, title, content, content_length

Usage:
  python scripts/data/audit_external_corpus.py \
      --corpus /path/to/hf_tmh_conv/default/train \
      --store data/docstore/v2 \
      --out data/external/hf_tmh

Writes:
  <out>/audit.json                 aggregate report
  <out>/candidates.parquet         one row per missing id with usable content metadata
  <out>/candidates.jsonl.gz        id/title/content for the same rows (for a later merge)
  <out>/host_coverage.csv          missing links vs external usable rows per host
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pandas as pd
import pyarrow.parquet as pq

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.corpus.cleaning import junk_reason  # noqa: E402
from r2ai2026.corpus.docstore import host_of  # noqa: E402
from r2ai2026.corpus.processing import content_key, lang_of, prepare_body  # noqa: E402

COLUMNS = ["id", "url", "domain", "status_code", "success", "title", "content", "content_length"]


def log(msg: str) -> None:
    print(msg, flush=True)


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.parquet"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="parquet file or directory (recursive)")
    ap.add_argument("--store", type=Path, default=ROOT / "data" / "docstore" / "v2")
    ap.add_argument("--links", type=Path, default=ROOT / "data" / "raw" / "btc" / "links_corpus_vi_en_zh.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "external" / "hf_tmh")
    ap.add_argument("--batch-size", type=int, default=20_000)
    args = ap.parse_args()

    files = parquet_files(args.corpus)
    if not files:
        raise SystemExit(f"no parquet files under {args.corpus}")
    args.out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.store / "docs_meta.parquet", columns=["id", "content_key", "lang", "n_chars"])
    v2_ids = set(meta["id"].astype("int64"))
    v2_keys = set(meta["content_key"].astype(str))
    v2_nchars = dict(zip(meta["id"].astype("int64"), meta["n_chars"].astype("int64")))
    links = pd.read_parquet(args.links, columns=["id", "url"])
    link_ids = set(links["id"].astype("int64"))
    missing = link_ids - v2_ids
    log(f"v2 docs {len(v2_ids):,}; links {len(link_ids):,}; missing {len(missing):,}; files {len(files):,}")

    total = 0
    seen: set[int] = set()
    duplicate_ids = 0
    not_in_links = 0
    overlap = 0
    missing_rows = 0
    success = Counter()
    missing_success = Counter()
    status = Counter()
    domains = Counter()
    domain_missing = Counter()
    domain_usable = Counter()
    lang_counts = Counter()
    missing_lang = Counter()
    improved = 0
    improved_by_domain = Counter()
    usable_by_flag = Counter()
    junk = Counter()
    candidate_rows: list[dict] = []

    for fi, path in enumerate(files, 1):
        pf = pq.ParquetFile(path)
        cols = [c for c in COLUMNS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            df = batch.to_pandas()
            if "id" not in df.columns:
                continue
            total += len(df)
            for row in df.itertuples(index=False):
                doc_id = int(row.id)
                if doc_id in seen:
                    duplicate_ids += 1
                    continue
                seen.add(doc_id)
                domain = str(getattr(row, "domain", "") or "")
                domains[domain] += 1
                ok = bool(getattr(row, "success", False))
                success[ok] += 1
                status[int(getattr(row, "status_code", 0) or 0)] += 1
                content = str(getattr(row, "content", "") or "")
                title = str(getattr(row, "title", "") or "")
                url = str(getattr(row, "url", "") or "")
                if doc_id not in link_ids:
                    not_in_links += 1
                    continue
                if doc_id in v2_ids:
                    overlap += 1
                    if len(content) > int(v2_nchars.get(doc_id, 0)) * 1.25 and len(content) >= 500:
                        improved += 1
                        improved_by_domain[domain] += 1
                    continue
                missing_rows += 1
                missing_success[ok] += 1
                domain_missing[domain] += 1
                missing_lang[lang_of(doc_id)] += 1
                if not ok or len(content.strip()) < 200:
                    usable_by_flag["not_success_or_short"] += 1
                    continue
                why = junk_reason(content, is_pubmed=False, dup_count=1)
                if why:
                    junk[why] += 1
                    continue
                title2, body = prepare_body(title, content)
                if not body.strip():
                    junk["empty_body"] += 1
                    continue
                key = content_key(body)
                candidate_rows.append({
                    "id": doc_id, "url": url, "host": domain or host_of(url), "lang": lang_of(doc_id),
                    "title": title2, "body": body, "n_chars": len(body), "thin": len(body) < 500,
                    "content_key": key, "status_code": int(getattr(row, "status_code", 0) or 0), "success": ok,
                })
                lang_counts[lang_of(doc_id)] += 1
                domain_usable[domain] += 1
                usable_by_flag["usable"] += 1
        if fi % 50 == 0:
            log(f"  {fi}/{len(files)} files, rows {total:,}, candidates {len(candidate_rows):,}")

    # Deduplicate candidates by id (keep longest body) and content_key (mark later duplicates)
    by_id: dict[int, dict] = {}
    for row in candidate_rows:
        prev = by_id.get(row["id"])
        if prev is None or row["n_chars"] > prev["n_chars"]:
            by_id[row["id"]] = row
    cand = pd.DataFrame(list(by_id.values()))
    if len(cand):
        cand = cand.sort_values("id").reset_index(drop=True)
        first_of: dict[str, int] = {}
        for key, doc_id in zip(cand["content_key"], cand["id"]):
            first_of.setdefault(str(key), int(doc_id))
        cand["dup_of"] = [first_of[str(k)] for k in cand["content_key"]]
        cand["dup_of_v2"] = [str(k) in v2_keys for k in cand["content_key"]]
        cand["is_dup"] = (cand["dup_of"] != cand["id"]) | cand["dup_of_v2"]
    else:
        cand["dup_of"] = []
        cand["dup_of_v2"] = []
        cand["is_dup"] = []

    # host table: official missing links vs external rows (all / usable)
    link_host = links.assign(host=links["url"].map(host_of))
    missing_host = link_host[link_host["id"].isin(missing)]["host"].value_counts()
    host_df = pd.DataFrame({
        "missing_links": missing_host,
        "external_rows_missing": pd.Series(domain_missing),
        "external_usable_missing": pd.Series(domain_usable),
    }).fillna(0).astype(int).sort_values(["external_usable_missing", "missing_links"], ascending=False)
    host_df.index.name = "host"
    host_df.to_csv(args.out / "host_coverage.csv", encoding="utf-8")

    report = {
        "corpus": str(args.corpus), "files": len(files), "rows": total, "unique_ids": len(seen),
        "duplicate_ids": duplicate_ids, "not_in_official_links": not_in_links,
        "v2_overlap": overlap, "v2_missing_links": len(missing),
        "missing_rows_present": missing_rows, "missing_success": dict(missing_success),
        "status": {str(k): v for k, v in status.most_common()},
        "top_domains": dict(domains.most_common(30)),
        "missing_by_lang": dict(missing_lang),
        "usable_by_lang": dict(lang_counts),
        "usable_by_flag": dict(usable_by_flag), "junk": dict(junk),
        "candidates_raw": len(candidate_rows), "candidates_unique_id": len(cand),
        "candidates_unique_content": int(len(cand) - int(cand["is_dup"].sum())) if len(cand) else 0,
        "candidates_dup_v2": int(cand["dup_of_v2"].sum()) if len(cand) else 0,
        "candidates_dup_within": int(((cand["dup_of"] != cand["id"]) & ~cand["dup_of_v2"]).sum()) if len(cand) else 0,
        "v2_overlap_with_longer_content": improved,
        "v2_overlap_longer_top_domains": dict(improved_by_domain.most_common(20)),
        "host_coverage_top": host_df.head(40).reset_index().to_dict(orient="records"),
    }
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if len(cand):
        cand.to_parquet(args.out / "candidates.parquet", index=False)
        # content for a later merge; only the usable missing rows
        with gzip.open(args.out / "candidates.jsonl.gz", "wb", compresslevel=5) as f:
            for row in cand.to_dict(orient="records"):
                f.write(orjson.dumps({k: row[k] for k in
                                      ("id", "url", "host", "lang", "title", "body", "content_key")}) + b"\n")
    log(json.dumps({k: report[k] for k in ("rows", "unique_ids", "missing_rows_present", "candidates_unique_id",
                                            "candidates_unique_content", "candidates_dup_v2",
                                            "candidates_dup_within", "v2_overlap_with_longer_content")},
                   ensure_ascii=False))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
