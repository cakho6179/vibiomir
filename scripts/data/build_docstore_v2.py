"""Build docstore v2 = v1 + the final VI recovery delta.

The v1 store is copied unchanged.  Only the selected recovery sources are merged:

  cc_columnar   Common Crawl columnar output (text in ``x``, title in ``t``)
  direct_vi     direct recrawl
  wayback_vi    Wayback recrawl
  lc-final      final Long Chau recrawl

Rules
-----
* ID dedupe: an id already in v1 is never added again.  When several delta files
  contain the same new id, the best usable record wins (direct/lc > wayback > cc,
  then longest text).
* Content dedupe: ``content_key`` is computed on the same body v1 uses.  A new
  document whose body already exists gets ``dup_of`` pointing at the first id
  with that body.  It is kept in ``docs_meta.parquet`` (ids are the unit of gold
  labels), while ``docs_meta_unique.parquet`` is the clean, deduped view
  (``dup_of == id``) for indexing.  ``delta_duplicates.parquet`` lists every
  duplicate decision.  Use ``--drop-content-duplicates`` to omit duplicates
  from the main store instead of marking them.
* Language: only Vietnamese ids (``id <= 766441``) are merged by default.  The
  cc_columnar file also contains ZH rows; they are deliberately skipped because
  ZH retrieval is out of scope.  Use ``--include-langs vi,zh`` to include them.
* Junk filter: the same ``cleaning.junk_reason`` gate as v1.  Thin documents are
  kept and flagged, not dropped.
* Sections: 4800/2000 chars, the same chunk unit used by v1 and the current
  section BM25 experiment.

Outputs in ``data/docstore/v2`` (offsets are body character positions):
  corpus/part-XX.jsonl.gz       v1 parts + one delta part
  docs_meta.parquet             all unique ids; duplicate bodies marked by dup_of
  docs_meta_unique.parquet      docs_meta[dup_of == id] for a clean index
  passages.parquet              v1 passages + delta passages
  sections.parquet              v1 sections + delta sections
  furniture.parquet             v1 furniture + delta furniture
  delta_duplicates.parquet      content-duplicate audit trail
  delta_report.json             per-source / per-language recovery counts
  templates.json, slug_docs.parquet, MANIFEST.json

Usage:
  python scripts/data/build_docstore_v2.py
  python scripts/data/build_docstore_v2.py --version v2 --force
  python scripts/data/build_docstore_v2.py --drop-content-duplicates
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.cleaning import junk_reason  # noqa: E402
from r2ai2026.docstore import (THIN_CHARS, as_doc, content_chars, furniture_spans,  # noqa: E402
                               passage_spans, pack_sections)
from r2ai2026.paths import DOCSTORE  # noqa: E402
from r2ai2026.processing import TemplateModel, content_key, lang_of, line_key, prepare_body  # noqa: E402

VI_MAX_ID = 766_441
DEFAULT_SOURCES = [
    ("cc", ROOT / "data" / "raw" / "recovered" / "cc_columnar_2026-10-24.jsonl"),
    ("direct", ROOT / "data" / "raw" / "recovered" / "direct_vi_2026-10-22.jsonl"),
    ("wayback", ROOT / "data" / "raw" / "recovered" / "wayback_vi_2026-10-22.jsonl"),
    ("lc", ROOT / "kaggle" / "outputs" / "vibiomir-recrawl-lc-final" / "docs.jsonl"),
]
SOURCE_RANK = {"direct": 1, "lc": 1, "wayback": 2, "cc": 3}
META_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("host", pa.string()), ("lang", pa.string()),
    ("source", pa.string()), ("n_chars", pa.int32()), ("thin", pa.bool_()),
    ("content_key", pa.string()), ("dup_of", pa.int64()), ("furniture_chars", pa.int32()),
    ("n_passages", pa.int32()),
])
PAS_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("pid", pa.int32()), ("start", pa.int32()), ("end", pa.int32()),
    ("heading", pa.string()), ("content_chars", pa.int32()),
])
SEC_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("sid", pa.int64()), ("pid_first", pa.int64()), ("pid_last", pa.int64()),
    ("start", pa.int64()), ("end", pa.int64()), ("heading", pa.string()),
])
FUR_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("start", pa.int32()), ("end", pa.int32()), ("kind", pa.string()),
])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_sources(items: list[str] | None) -> list[tuple[str, Path]]:
    if not items:
        return DEFAULT_SOURCES
    out = []
    for item in items:
        label, sep, raw = item.partition("=")
        if not sep or not label or not raw:
            raise SystemExit(f"--source must be LABEL=PATH, got {item!r}")
        if label not in SOURCE_RANK:
            raise SystemExit(f"unknown source label {label!r}; use one of {sorted(SOURCE_RANK)}")
        out.append((label, Path(raw)))
    return out


def load_template_model(base: Path, min_share: float, seed: int) -> TemplateModel:
    """Rebuild the v1 template masks from the exported (share, line) examples."""
    model = TemplateModel(min_share=min_share, seed=seed)
    path = base / "templates.json"
    if not path.exists():
        return model
    raw = json.loads(path.read_text(encoding="utf-8"))
    for host, examples in raw.items():
        model.lines[host] = {line_key(line) for _, line in examples}
    return model


def write_parquet(df: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=DOCSTORE / "v1")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--source", action="append", default=None, metavar="LABEL=PATH")
    ap.add_argument("--include-langs", default="vi", help="comma-separated langs (default: vi)")
    ap.add_argument("--drop-content-duplicates", action="store_true",
                    help="omit duplicate bodies from docs_meta instead of marking dup_of")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--section-max-chars", type=int, default=4800)
    ap.add_argument("--section-min-chars", type=int, default=2000)
    ap.add_argument("--template-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = ap.parse_args()

    base = args.base
    out = args.out or (DOCSTORE / args.version)
    if not (base / "MANIFEST.json").exists() or not (base / "docs_meta.parquet").exists():
        raise SystemExit(f"base store is incomplete: {base}")
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} already exists; use --force to replace it")
        shutil.rmtree(out)
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    include_langs = {x.strip() for x in args.include_langs.split(",") if x.strip()}
    sources = parse_sources(args.source)
    for label, path in sources:
        if not path.exists():
            raise SystemExit(f"source {label} not found: {path}")

    base_manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    base_meta = pd.read_parquet(base / "docs_meta.parquet")
    base_ids = set(base_meta["id"].astype("int64"))
    base_key_to_id: dict[str, int] = {}
    for key, dup in zip(base_meta["content_key"], base_meta["dup_of"]):
        base_key_to_id.setdefault(str(key), int(dup))
    log(f"base {base.name}: {len(base_meta):,} docs, {len(base_key_to_id):,} content keys")

    # ---- S0: read every delta record, group by id, keep only usable records
    records: dict[int, list[dict]] = defaultdict(list)
    file_stats: dict[str, dict] = {}
    junk_by_source: dict[str, Counter] = defaultdict(Counter)
    for label, path in sources:
        lines = bad = 0
        with path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                lines += 1
                try:
                    rec = orjson.loads(line)
                except orjson.JSONDecodeError:
                    bad += 1
                    continue
                doc = as_doc(rec, label)
                if doc is None:
                    bad += 1
                    continue
                records[doc["id"]].append(doc)
        file_stats[label] = {"path": str(path), "sha16": sha16(path), "lines": lines, "bad": bad,
                             "unique_ids": len({d["id"] for d in records.values() for d in d if d["source"] == label})}
        log(f"  {label}: {lines:,} lines, {bad:,} bad")

    # ---- S0/S1: id/language/junk/dedup decisions, prepare bodies
    skipped_id = skipped_lang = 0
    skipped_junk = 0
    skipped_content = 0
    accepted: list[dict] = []
    duplicates: list[dict] = []
    new_key_to_id: dict[str, int] = {}
    delta_by_source: Counter = Counter()
    delta_by_lang: Counter = Counter()
    delta_chars = delta_thin = 0

    for doc_id in sorted(records):
        if doc_id in base_ids:
            skipped_id += 1
            continue
        lang = lang_of(doc_id)
        if lang not in include_langs:
            skipped_lang += 1
            continue
        good = []
        for doc in records[doc_id]:
            why = junk_reason(doc["text"], is_pubmed=False, dup_count=1)
            if why:
                junk_by_source[doc["source"]][why] += 1
            else:
                good.append(doc)
        if not good:
            skipped_junk += 1
            continue
        doc = min(good, key=lambda d: (SOURCE_RANK[d["source"]], -len(d["text"])))
        title, body = prepare_body(doc["title"], doc["text"])
        if not body.strip():
            skipped_junk += 1
            junk_by_source[doc["source"]]["empty_body"] += 1
            continue
        key = content_key(body)
        if key in base_key_to_id:
            dup_of, dup_kind = base_key_to_id[key], "base"
        elif key in new_key_to_id:
            dup_of, dup_kind = new_key_to_id[key], "delta"
        else:
            dup_of, dup_kind = doc_id, ""
            new_key_to_id[key] = doc_id
        row = {
            "id": doc_id, "url": doc["url"], "host": doc["host"], "lang": lang, "source": doc["source"],
            "n_chars": len(body), "thin": len(body) < THIN_CHARS, "content_key": key, "dup_of": dup_of,
            "furniture_chars": 0, "n_passages": 0,
            "_title": title, "_body": body,
        }
        if dup_of != doc_id:
            duplicates.append({"id": doc_id, "url": doc["url"], "host": doc["host"], "source": doc["source"],
                               "lang": lang, "n_chars": len(body), "content_key": key, "dup_of": dup_of,
                               "duplicate_of": dup_kind, "kept": not args.drop_content_duplicates})
            if args.drop_content_duplicates:
                skipped_content += 1
                continue
        accepted.append(row)
        delta_by_source[doc["source"]] += 1
        delta_by_lang[lang] += 1
        delta_chars += len(body)
        delta_thin += len(body) < THIN_CHARS

    log(f"delta: id-new {len(records) - skipped_id:,}, accepted {len(accepted):,}, "
        f"skip id={skipped_id:,} lang={skipped_lang:,} junk={skipped_junk:,} content={skipped_content:,}")

    # ---- S2/S3: furniture, passages, sections, corpus delta part
    model = load_template_model(base, args.template_share, args.seed)
    log(f"templates from v1: {len(model.lines):,} hosts")
    pas_rows: list[dict] = []
    sec_rows: list[dict] = []
    fur_rows: list[dict] = []
    for row in accepted:
        doc_id, body = row["id"], row["_body"]
        lines = body.split("\n")
        fur = furniture_spans(lines, model.template_mask(row["host"], lines))
        pas = passage_spans(lines, max_chars=args.max_chars, min_chars=args.min_chars)
        sec = pack_sections(pas, max_chars=args.section_max_chars, min_chars=args.section_min_chars)
        row["furniture_chars"] = sum(e - s for s, e, _ in fur)
        row["n_passages"] = len(pas)
        fur_rows.extend({"doc_id": doc_id, "start": s, "end": e, "kind": k} for s, e, k in fur)
        pas_rows.extend({"doc_id": doc_id, "pid": pid, "start": s, "end": e, "heading": h,
                         "content_chars": content_chars(s, e, fur)} for pid, (s, e, h) in enumerate(pas))
        sec_rows.extend({"doc_id": doc_id, "sid": sid, "pid_first": a, "pid_last": b, "start": s,
                         "end": e, "heading": pas[a][2]} for sid, (a, b, s, e) in enumerate(sec))

    # write the delta corpus part next to copied v1 parts
    base_parts = sorted((base / "corpus").glob("part-*.jsonl.gz"))
    if not base_parts:
        raise SystemExit(f"no corpus parts in {base / 'corpus'}")
    next_part = max(int(p.stem.split("-")[1].split(".")[0]) for p in base_parts) + 1
    for p in base_parts:
        shutil.copy2(p, out / "corpus" / p.name)
    delta_part = out / "corpus" / f"part-{next_part:02d}.jsonl.gz"
    with gzip.open(delta_part, "wb", compresslevel=5) as f:
        for row in sorted(accepted, key=lambda r: r["id"]):
            f.write(orjson.dumps({"id": row["id"], "t": row["_title"], "x": row["_body"]}) + b"\n")
    log(f"corpus: {len(base_parts)} v1 parts + {delta_part.name} ({len(accepted):,} docs)")

    # ---- combine metadata layers
    new_meta = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in accepted])
    if len(new_meta):
        meta = pd.concat([base_meta, new_meta], ignore_index=True).sort_values("id").reset_index(drop=True)
    else:
        meta = base_meta.copy()
    write_parquet(meta, out / "docs_meta.parquet", META_SCHEMA)
    unique_meta = meta[meta["dup_of"] == meta["id"]].copy()
    write_parquet(unique_meta, out / "docs_meta_unique.parquet", META_SCHEMA)

    base_pas = pd.read_parquet(base / "passages.parquet")
    base_sec = pd.read_parquet(base / "sections.parquet")
    base_fur = pd.read_parquet(base / "furniture.parquet")
    pas = pd.concat([base_pas, pd.DataFrame(pas_rows)], ignore_index=True).sort_values(
        ["doc_id", "pid"]).reset_index(drop=True)
    sec = pd.concat([base_sec, pd.DataFrame(sec_rows)], ignore_index=True).sort_values(
        ["doc_id", "sid"]).reset_index(drop=True)
    fur = pd.concat([base_fur, pd.DataFrame(fur_rows)], ignore_index=True).sort_values(
        ["doc_id", "start"]).reset_index(drop=True)
    write_parquet(pas, out / "passages.parquet", PAS_SCHEMA)
    write_parquet(sec, out / "sections.parquet", SEC_SCHEMA)
    write_parquet(fur, out / "furniture.parquet", FUR_SCHEMA)
    shutil.copy2(base / "templates.json", out / "templates.json")

    dup = pd.DataFrame(duplicates)
    if len(dup):
        dup = dup.sort_values("id").reset_index(drop=True)
    dup.to_parquet(out / "delta_duplicates.parquet", index=False)

    # ---- stats and manifest
    length = sec["end"] - sec["start"]
    vi = sec["doc_id"] <= VI_MAX_ID
    by_lang = meta["lang"].value_counts().to_dict()
    by_source = meta["source"].value_counts().to_dict()
    section_stats = {
        "max_chars": args.section_max_chars, "min_chars": args.section_min_chars, "sections": int(len(sec)),
        "per_doc": round(len(sec) / max(sec["doc_id"].nunique(), 1), 3),
        "vi_len_quantiles": {str(q): int(v) for q, v in length[vi].quantile([.05, .25, .5, .75, .95]).items()},
        "over_max": int((length > args.section_max_chars).sum()),
    }
    counts = {
        "docs": int(len(meta)), "passages": int(len(pas)), "sections": int(len(sec)),
        "chars": int(meta["n_chars"].sum()), "furniture_chars": int(meta["furniture_chars"].sum()),
        "thin": int(meta["thin"].sum()), "exact_duplicates": int((meta["dup_of"] != meta["id"]).sum()),
        "by_lang": by_lang, "by_source": by_source,
        "furniture_share": round(int(meta["furniture_chars"].sum()) / max(int(meta["n_chars"].sum()), 1), 4),
        "unique_docs": int(len(unique_meta)),
        "delta": {
            "accepted": int(len(accepted)),
            "unique_content": int(len(accepted) if args.drop_content_duplicates else len(accepted) - len(duplicates)),
            "content_duplicates_kept": int(len(duplicates)) if not args.drop_content_duplicates else 0,
            "content_duplicates_dropped": int(skipped_content),
            "skipped_id_in_base": int(skipped_id), "skipped_lang": int(skipped_lang),
            "skipped_junk": int(skipped_junk), "by_lang": dict(delta_by_lang),
            "by_source": dict(delta_by_source), "chars": int(delta_chars), "thin": int(delta_thin),
            "delta_part": delta_part.name,
        },
    }
    delta_report = {
        "base": str(base), "version": args.version, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": file_stats, "junk": {k: dict(v) for k, v in junk_by_source.items()},
        "include_langs": sorted(include_langs), "drop_content_duplicates": bool(args.drop_content_duplicates),
        "counts": counts["delta"],
    }
    (out / "delta_report.json").write_text(json.dumps(delta_report, ensure_ascii=False, indent=1), encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "r2ai2026", "scripts", "kaggle/common"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    manifest = {
        "version": args.version, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit,
        "git_dirty": dirty, "parent": base.name,
        "inputs": {
            "base/MANIFEST.json": sha16(base / "MANIFEST.json"),
            "base/docs_meta.parquet": sha16(base / "docs_meta.parquet"),
            "base/passages.parquet": sha16(base / "passages.parquet"),
            "base/sections.parquet": sha16(base / "sections.parquet"),
            "base/furniture.parquet": sha16(base / "furniture.parquet"),
            "base/templates.json": sha16(base / "templates.json"),
            **{f"{label}/{Path(s['path']).name}": s["sha16"] for label, s in file_stats.items()},
        },
        "params": {
            "max_chars": args.max_chars, "min_chars": args.min_chars, "template_share": args.template_share,
            "seed": args.seed, "thin_chars": THIN_CHARS, "parts": len(base_parts) + 1,
            "delta_part": delta_part.name, "section_max_chars": args.section_max_chars,
            "section_min_chars": args.section_min_chars, "include_langs": sorted(include_langs),
            "content_dedupe": "drop" if args.drop_content_duplicates else "mark_dup_of",
        },
        "counts": counts,
        "sections": section_stats,
        "corpus_parts": [p.name for p in sorted((out / "corpus").glob("part-*.jsonl.gz"))],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    # rebuild slug docs for the new coverage and let that script update slug_docs in the manifest
    subprocess.run([sys.executable, str(ROOT / "scripts" / "data" / "build_slug_docs.py"), "--store", str(out)],
                   check=True, cwd=ROOT)
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    man["delta"] = delta_report
    man["docs_meta_unique"] = {"docs": int(len(unique_meta))}
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: counts[k] for k in ("docs", "unique_docs", "passages", "sections", "chars")},
                   ensure_ascii=False))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
