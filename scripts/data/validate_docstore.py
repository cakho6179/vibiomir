"""Validate a docstore build and write a machine-readable report.

The validator is intentionally independent of the builder: it streams the
corpus back, recomputes the metadata invariants and checks that passages,
sections and furniture tile every body.

Usage::

    python scripts/data/validate_docstore.py --store data/docstore/v2.1
    python scripts/data/validate_docstore.py --store data/docstore/v2.1 --base data/docstore/v2
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

import orjson
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.corpus.docstore import THIN_CHARS  # noqa: E402

DEFAULT_STORE = ROOT / "data" / "docstore" / "v2.1"
DEFAULT_LINKS = ROOT / "data" / "raw" / "btc" / "links_corpus_vi_en_zh.parquet"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--no-corpus", action="store_true", help="skip the full corpus stream")
    args = ap.parse_args()

    store = args.store
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    def err(msg: str) -> None:
        errors.append(msg)

    def warn(msg: str) -> None:
        warnings.append(msg)

    required = ["MANIFEST.json", "docs_meta.parquet", "docs_meta_unique.parquet",
                "passages.parquet", "sections.parquet", "furniture.parquet",
                "templates.json", "corpus"]
    for name in required:
        if not (store / name).exists():
            err(f"missing {name}")
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=1))
        raise SystemExit(1)

    manifest = json.loads((store / "MANIFEST.json").read_text(encoding="utf-8"))
    meta = pd.read_parquet(store / "docs_meta.parquet")
    unique = pd.read_parquet(store / "docs_meta_unique.parquet")
    pas = pd.read_parquet(store / "passages.parquet")
    sec = pd.read_parquet(store / "sections.parquet")
    fur = pd.read_parquet(store / "furniture.parquet")
    log(f"loaded meta={len(meta):,} unique={len(unique):,} pas={len(pas):,} sec={len(sec):,} fur={len(fur):,}")

    # ---- metadata
    ids = set(meta["id"].astype("int64"))
    if meta["id"].duplicated().any():
        err("duplicate ids in docs_meta")
    if meta["content_key"].isna().any():
        err("null content_key")
    if meta["dup_of"].isna().any():
        err("null dup_of")
    if not ids.issuperset(set(meta["dup_of"].astype("int64"))):
        err("dup_of points outside docs_meta")
    key_by_id = dict(zip(meta["id"].astype("int64"), meta["content_key"].astype(str)))
    anchor_by_id = dict(zip(meta["id"].astype("int64"), meta["dup_of"].astype("int64")))
    for doc_id, anchor in anchor_by_id.items():
        if anchor != doc_id:
            if key_by_id.get(doc_id) != key_by_id.get(anchor):
                err(f"duplicate key mismatch id={doc_id} anchor={anchor}")
            if anchor_by_id.get(anchor) != anchor:
                err(f"duplicate anchor is not an anchor id={doc_id} anchor={anchor}")
    if (meta["n_chars"] < 0).any():
        err("negative n_chars")
    if not (meta["thin"].astype(bool) == (meta["n_chars"] < THIN_CHARS)).all():
        err("thin flag mismatch")
    if not meta["lang"].isin(["vi", "en", "zh"]).all():
        err("unknown language")
    if meta["url"].isna().any() or meta["host"].isna().any():
        err("null url/host")
    stats["by_lang"] = {str(k): int(v) for k, v in meta["lang"].value_counts().items()}
    stats["by_source"] = {str(k): int(v) for k, v in meta["source"].value_counts().items()}
    stats["chars"] = int(meta["n_chars"].sum())
    stats["thin"] = int(meta["thin"].sum())
    stats["duplicates"] = int((meta["dup_of"] != meta["id"]).sum())

    # ---- unique view
    expected_unique = meta[meta["dup_of"] == meta["id"]]
    if len(expected_unique) != len(unique):
        err(f"docs_meta_unique length {len(unique)} != {len(expected_unique)}")
    if set(expected_unique["id"].astype("int64")) != set(unique["id"].astype("int64")):
        err("docs_meta_unique ids differ from dup_of==id")
    if unique["content_key"].duplicated().any():
        err("duplicate content_key in docs_meta_unique")
    stats["unique_docs"] = int(len(unique))

    # ---- official link coverage
    links = pd.read_parquet(args.links, columns=["id"])
    link_ids = set(links["id"].astype("int64"))
    outside = ids - link_ids
    if outside:
        err(f"{len(outside)} docs outside official links; examples={sorted(outside)[:10]}")
    stats["outside_links"] = int(len(outside))
    stats["missing_links"] = int(len(link_ids - ids))

    # ---- base comparison
    if args.base is not None:
        base = pd.read_parquet(args.base / "docs_meta.parquet", columns=["id", "content_key", "lang"])
        base_ids = set(base["id"].astype("int64"))
        removed = base_ids - ids
        if removed:
            err(f"{len(removed)} base ids missing from output; examples={sorted(removed)[:10]}")
        added = ids - base_ids
        stats["base_docs"] = int(len(base_ids))
        stats["added_docs"] = int(len(added))
        stats["removed_docs"] = int(len(removed))
        log(f"base comparison: added {len(added):,}, removed {len(removed):,}")

    # ---- corpus stream
    n_chars = dict(zip(meta["id"].astype("int64"), meta["n_chars"].astype("int64")))
    n_passages_by_id = dict(zip(meta["id"].astype("int64"), meta["n_passages"].astype("int64")))
    furniture_by_id = dict(zip(meta["id"].astype("int64"), meta["furniture_chars"].astype("int64")))
    corpus_ids: set[int] = set()
    corpus_docs = 0
    body_chars = 0
    if not args.no_corpus:
        for path in sorted((store / "corpus").glob("part-*.jsonl.gz")):
            part_docs = 0
            with gzip.open(path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = orjson.loads(line)
                    doc_id = int(rec["id"])
                    if doc_id in corpus_ids:
                        err(f"duplicate corpus id {doc_id} in {path.name}")
                    corpus_ids.add(doc_id)
                    body = str(rec.get("x") or "")
                    body_chars += len(body)
                    if doc_id not in ids:
                        err(f"corpus id outside metadata {doc_id}")
                    elif len(body) != int(n_chars.get(doc_id, -1)):
                        err(f"corpus body length != n_chars id={doc_id}")
                    part_docs += 1
                    corpus_docs += 1
            log(f"  corpus {path.name}: {part_docs:,} docs")
        missing = ids - corpus_ids
        extra = corpus_ids - ids
        if missing:
            err(f"{len(missing)} metadata ids missing from corpus; examples={sorted(missing)[:10]}")
        if extra:
            err(f"{len(extra)} corpus ids not in metadata; examples={sorted(extra)[:10]}")
        if corpus_docs != len(meta):
            err(f"corpus docs {corpus_docs} != metadata {len(meta)}")
        stats["corpus_docs"] = corpus_docs
        stats["corpus_chars"] = body_chars
        if body_chars != int(meta["n_chars"].sum()):
            err(f"corpus chars {body_chars} != metadata chars {int(meta['n_chars'].sum())}")
        log(f"corpus: {corpus_docs:,} docs, {body_chars:,} chars")

    # ---- passages tile bodies
    pas_by_doc: dict[int, list[tuple[int, int, int, int, str]]] = {}
    for row in pas.itertuples(index=False):
        pas_by_doc.setdefault(int(row.doc_id), []).append(
            (int(row.pid), int(row.start), int(row.end), int(row.content_chars), str(row.heading))
        )
    if set(pas_by_doc) != ids:
        err(f"passages doc set mismatch: missing={len(ids - set(pas_by_doc))}, extra={len(set(pas_by_doc) - ids)}")
    empty_docs: set[int] = set()
    for doc_id, rows in pas_by_doc.items():
        rows.sort()
        expected_n = int(n_chars.get(doc_id, -1))
        if not rows:
            err(f"no passages doc={doc_id}")
            continue
        if [r[0] for r in rows] != list(range(len(rows))):
            err(f"non-contiguous pids doc={doc_id}")
        if rows[0][1] != 0:
            err(f"first passage start != 0 doc={doc_id}")
        if rows[-1][2] != expected_n:
            err(f"last passage end != n_chars doc={doc_id}")
        if expected_n == 0:
            empty_docs.add(doc_id)
            if len(rows) != 1 or rows[0][1] != 0 or rows[0][2] != 0:
                err(f"empty doc has non-empty passage doc={doc_id}")
            continue
        for i, r in enumerate(rows):
            if r[1] >= r[2]:
                err(f"empty passage doc={doc_id} pid={r[0]}")
            if r[3] > r[2] - r[1]:
                err(f"content_chars > span doc={doc_id} pid={r[0]}")
            if i and rows[i - 1][2] != r[1]:
                err(f"passage gap/overlap doc={doc_id} pid={r[0]}")
        if int(n_passages_by_id.get(doc_id, -1)) != len(rows):
            err(f"n_passages mismatch doc={doc_id}")
    stats["passages"] = int(len(pas))
    stats["empty_docs"] = int(len(empty_docs))
    if empty_docs:
        warn(f"{len(empty_docs)} docs have empty bodies (kept for id coverage): {sorted(empty_docs)[:10]}")

    # ---- sections tile passages
    sec_by_doc: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for row in sec.itertuples(index=False):
        sec_by_doc.setdefault(int(row.doc_id), []).append(
            (int(row.sid), int(row.pid_first), int(row.pid_last), int(row.start), int(row.end))
        )
    if set(sec_by_doc) != ids:
        err(f"sections doc set mismatch: missing={len(ids - set(sec_by_doc))}, extra={len(set(sec_by_doc) - ids)}")
    for doc_id, rows in sec_by_doc.items():
        rows.sort()
        prs = pas_by_doc.get(doc_id, [])
        if not prs:
            err(f"sections without passages doc={doc_id}")
            continue
        if [r[0] for r in rows] != list(range(len(rows))):
            err(f"non-contiguous sids doc={doc_id}")
        if rows[0][1] != 0 or rows[0][3] != 0:
            err(f"first section does not start at 0 doc={doc_id}")
        if rows[-1][2] != len(prs) - 1 or rows[-1][4] != n_chars[doc_id]:
            err(f"last section does not end at body doc={doc_id}")
        prev_last = -1
        for sid, a, b, s, e in rows:
            if a != prev_last + 1:
                err(f"section pid gap doc={doc_id} sid={sid}")
            if not (0 <= a <= b < len(prs)):
                err(f"section pid out of range doc={doc_id} sid={sid}")
                continue
            if s != prs[a][1] or e != prs[b][2]:
                err(f"section bounds do not match passages doc={doc_id} sid={sid}")
            if e - s > 4800:
                err(f"section over max doc={doc_id} sid={sid} len={e - s}")
            prev_last = b
        if prev_last != len(prs) - 1:
            err(f"sections do not cover all passages doc={doc_id}")
    stats["sections"] = int(len(sec))

    # ---- furniture spans
    fur_by_doc: dict[int, list[tuple[int, int]]] = {}
    for row in fur.itertuples(index=False):
        fur_by_doc.setdefault(int(row.doc_id), []).append((int(row.start), int(row.end)))
    for doc_id, spans in fur_by_doc.items():
        spans.sort()
        expected_n = int(n_chars.get(doc_id, -1))
        prev_end = -1
        for s, e in spans:
            if not (0 <= s < e <= expected_n):
                err(f"furniture span out of range doc={doc_id} span={s}:{e}")
            if s < prev_end:
                err(f"overlapping furniture doc={doc_id} span={s}:{e}")
            prev_end = max(prev_end, e)
        if sum(e - s for s, e in spans) != int(furniture_by_id.get(doc_id, -1)):
            err(f"furniture_chars mismatch doc={doc_id}")
    stats["furniture"] = int(len(fur))

    # ---- manifest
    counts = manifest.get("counts", {})
    for key, actual in [("docs", len(meta)), ("unique_docs", len(unique)), ("passages", len(pas)),
                        ("sections", len(sec)), ("chars", int(meta["n_chars"].sum()))]:
        if int(counts.get(key, -1)) != actual:
            err(f"manifest counts.{key}={counts.get(key)} != {actual}")
    if int(manifest.get("sections", {}).get("over_max", -1)) != 0:
        err("manifest sections.over_max != 0")
    if int(manifest.get("docs_meta_unique", {}).get("docs", -1)) != len(unique):
        err("manifest docs_meta_unique mismatch")
    stats["manifest_version"] = manifest.get("version")
    stats["manifest_parent"] = manifest.get("parent")

    # ---- audits (optional)
    for name in ("delta_duplicates.parquet", "refresh_audit.parquet", "refresh_skipped.parquet"):
        p = store / name
        if not p.exists():
            warn(f"optional audit missing: {name}")
            continue
        d = pd.read_parquet(p)
        if "id" in d.columns:
            bad = set(d["id"].astype("int64")) - ids
            if bad:
                err(f"{name} ids outside docs_meta: {len(bad)}")

    stats["errors"] = len(errors)
    stats["warnings"] = len(warnings)
    out = {
        "store": str(store),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
        "errors": errors[:200],
        "warnings": warnings[:200],
    }
    target = args.json_out or (store / "validation.json")
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"stats": stats, "errors": errors[:20], "warnings": warnings[:20]},
                     ensure_ascii=False, indent=1))
    log(f"wrote {target}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
