"""Build docstore v2.1 from v2 + the selected tmh1008 merge set.

This is deliberately an *incremental* builder: the v2 corpus is streamed and
only the selected rows are rewritten.  Unchanged documents keep their exact
v2 bytes; refreshed documents get new bodies and new passages/sections; added
documents get a new corpus part.

Inputs are produced by ``scripts/data/select_hf_v2_1.py``:

  v2_1_add.jsonl.gz       missing VI/ZH documents (cleaned bodies)
  v2_1_refresh.jsonl.gz   existing VI documents with a better full-article body

Rules
-----
* v2 IDs are immutable; added IDs must not exist in v2.
* Content dedupe is still exact ``content_key`` dedupe.  Added duplicates are
  kept in ``docs_meta.parquet`` with ``dup_of`` and excluded from
  ``docs_meta_unique.parquet``; the audit trail is in ``delta_duplicates.parquet``.
* A refresh is skipped when its old body is an anchor with duplicates (that
  would break existing ``dup_of`` links), when the new key already belongs to
  another document, or when the new body is unchanged.
* Templates are inherited from v2; no new template model is fitted on external
  text.  Passages/sections use the same 1500/4800 settings as v2.

Usage::

    python scripts/data/build_docstore_v2_1.py --force
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
from collections import Counter
from pathlib import Path

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "r2ai2026").is_dir())
sys.path.insert(0, str(ROOT))
from r2ai2026.corpus.cleaning import junk_reason  # noqa: E402
from r2ai2026.corpus.docstore import (  # noqa: E402
    THIN_CHARS, content_chars, furniture_spans, passage_spans, pack_sections,
)
from r2ai2026.corpus.processing import (  # noqa: E402
    TemplateModel, content_key, lang_of, line_key, prepare_body,
)

DEFAULT_BASE = ROOT / "data" / "docstore" / "v2"
DEFAULT_ADD = ROOT / "data" / "external" / "hf_tmh" / "v2_1_add.jsonl.gz"
DEFAULT_REFRESH = ROOT / "data" / "external" / "hf_tmh" / "v2_1_refresh.jsonl.gz"
DEFAULT_SELECTION = ROOT / "data" / "external" / "hf_tmh" / "v2_1_selection.json"

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
DUP_SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("host", pa.string()), ("source", pa.string()),
    ("lang", pa.string()), ("n_chars", pa.int32()), ("content_key", pa.string()),
    ("dup_of", pa.int64()), ("duplicate_of", pa.string()), ("kept", pa.bool_()),
])
SKIP_SCHEMA = pa.schema([
    ("id", pa.int64()), ("host", pa.string()), ("reason", pa.string()), ("old_key", pa.string()),
    ("new_key", pa.string()), ("old_chars", pa.int32()), ("new_chars", pa.int32()),
])
REFRESH_AUDIT_SCHEMA = pa.schema([
    ("id", pa.int64()), ("host", pa.string()), ("old_chars", pa.int32()), ("new_chars", pa.int32()),
    ("gain_chars", pa.int32()), ("old_key", pa.string()), ("new_key", pa.string()),
    ("cov", pa.float64()),
])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_parquet(df: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def load_template_model(base: Path, min_share: float, seed: int) -> TemplateModel:
    model = TemplateModel(min_share=min_share, seed=seed)
    path = base / "templates.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for host, examples in raw.items():
            model.lines[host] = {line_key(line) for _, line in examples}
    return model


def read_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def compute_spans(doc_id: int, body: str, host: str, model: TemplateModel,
                  max_chars: int, min_chars: int, section_max_chars: int,
                  section_min_chars: int) -> tuple[dict, list[dict], list[dict], list[dict]]:
    lines = body.split("\n")
    fur = furniture_spans(lines, model.template_mask(host, lines))
    pas = passage_spans(lines, max_chars=max_chars, min_chars=min_chars)
    sec = pack_sections(pas, max_chars=section_max_chars, min_chars=section_min_chars)
    fur_rows = [{"doc_id": doc_id, "start": s, "end": e, "kind": k} for s, e, k in fur]
    pas_rows = [
        {"doc_id": doc_id, "pid": pid, "start": s, "end": e, "heading": h,
         "content_chars": content_chars(s, e, fur)}
        for pid, (s, e, h) in enumerate(pas)
    ]
    sec_rows = [
        {"doc_id": doc_id, "sid": sid, "pid_first": a, "pid_last": b, "start": s,
         "end": e, "heading": pas[a][2]}
        for sid, (a, b, s, e) in enumerate(sec)
    ]
    meta = {
        "furniture_chars": sum(e - s for s, e, _ in fur),
        "n_passages": len(pas),
    }
    return meta, pas_rows, sec_rows, fur_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--add", type=Path, default=DEFAULT_ADD)
    ap.add_argument("--refresh", type=Path, default=DEFAULT_REFRESH)
    ap.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "docstore" / "v2.1")
    ap.add_argument("--version", default="v2.1")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--section-max-chars", type=int, default=4800)
    ap.add_argument("--section-min-chars", type=int, default=2000)
    ap.add_argument("--template-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base, out = args.base, args.out
    if not (base / "MANIFEST.json").exists() or not (base / "docs_meta.parquet").exists():
        raise SystemExit(f"base store is incomplete: {base}")
    for p in (args.add, args.refresh, args.selection):
        if not p.exists():
            raise SystemExit(f"input not found: {p}")
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} exists; use --force to replace it")
        shutil.rmtree(out)
    (out / "corpus").mkdir(parents=True, exist_ok=True)

    log(f"base {base.name}")
    base_meta = pd.read_parquet(base / "docs_meta.parquet")
    base_ids = set(base_meta["id"].astype("int64"))
    base_key_to_anchor: dict[str, int] = {}
    for key, dup in zip(base_meta["content_key"], base_meta["dup_of"]):
        base_key_to_anchor.setdefault(str(key), int(dup))
    base_key_by_id = dict(zip(base_meta["id"].astype("int64"), base_meta["content_key"].astype(str)))
    base_row_by_id = base_meta.set_index(base_meta["id"].astype("int64"))
    dup_counts = base_meta.groupby("dup_of").size().to_dict()
    log(f"base docs {len(base_meta):,}; content keys {len(base_key_to_anchor):,}")

    add_in = read_jsonl_gz(args.add)
    refresh_in = read_jsonl_gz(args.refresh)
    log(f"inputs: add {len(add_in):,}, refresh {len(refresh_in):,}")

    model = load_template_model(base, args.template_share, args.seed)
    log(f"templates: {len(model.lines):,} hosts")

    # ---- added documents: id / language / junk / content dedupe
    add_rows: list[dict] = []
    add_duplicates: list[dict] = []
    add_skipped: list[dict] = []
    added_key_to_anchor: dict[str, int] = {}
    seen_add_ids: set[int] = set()
    for rec in sorted(add_in, key=lambda x: int(x["id"])):
        doc_id = int(rec["id"])
        if doc_id in seen_add_ids:
            add_skipped.append({"id": doc_id, "reason": "duplicate_input_id"})
            continue
        seen_add_ids.add(doc_id)
        if doc_id in base_ids:
            add_skipped.append({"id": doc_id, "reason": "id_in_base"})
            continue
        lang = lang_of(doc_id)
        if lang not in {"vi", "zh"}:
            add_skipped.append({"id": doc_id, "reason": f"lang_{lang}"})
            continue
        title, body = prepare_body(str(rec.get("title") or ""), str(rec.get("body") or ""))
        if not body.strip():
            add_skipped.append({"id": doc_id, "reason": "empty_body"})
            continue
        why = junk_reason(body, is_pubmed=False, dup_count=1)
        if why:
            add_skipped.append({"id": doc_id, "reason": f"junk_{why}"})
            continue
        key = content_key(body)
        if key in base_key_to_anchor:
            dup_of, dup_kind = base_key_to_anchor[key], "base"
        elif key in added_key_to_anchor:
            dup_of, dup_kind = added_key_to_anchor[key], "added"
        else:
            dup_of, dup_kind = doc_id, ""
            added_key_to_anchor[key] = doc_id
        row = {
            "id": doc_id, "url": str(rec.get("url") or ""), "host": str(rec.get("host") or ""),
            "lang": lang, "source": "tmh", "n_chars": len(body), "thin": len(body) < THIN_CHARS,
            "content_key": key, "dup_of": dup_of, "furniture_chars": 0, "n_passages": 0,
            "_title": title, "_body": body,
        }
        if dup_of != doc_id:
            add_duplicates.append({
                "id": doc_id, "url": row["url"], "host": row["host"], "source": "tmh",
                "lang": lang, "n_chars": len(body), "content_key": key, "dup_of": dup_of,
                "duplicate_of": dup_kind, "kept": True,
            })
        add_rows.append(row)
    log(f"added accepted {len(add_rows):,}; duplicates {len(add_duplicates):,}; skipped {len(add_skipped):,}")

    # ---- refresh documents: only unique anchors, no content collisions
    refresh_rows: list[dict] = []
    refresh_audit: list[dict] = []
    refresh_skipped: list[dict] = []
    refresh_map: dict[int, dict] = {}
    seen_refresh: set[int] = set()
    for rec in sorted(refresh_in, key=lambda x: int(x["id"])):
        doc_id = int(rec["id"])
        if doc_id in seen_refresh:
            refresh_skipped.append({"id": doc_id, "reason": "duplicate_input_id"})
            continue
        seen_refresh.add(doc_id)
        if doc_id not in base_ids:
            refresh_skipped.append({"id": doc_id, "reason": "id_not_in_base"})
            continue
        old = base_row_by_id.loc[doc_id]
        old_key = str(old["content_key"])
        if int(old["dup_of"]) != doc_id:
            refresh_skipped.append({"id": doc_id, "reason": "old_body_is_duplicate",
                                    "old_key": old_key})
            continue
        if dup_counts.get(doc_id, 0) > 1:
            refresh_skipped.append({"id": doc_id, "reason": "old_body_has_duplicates",
                                    "old_key": old_key})
            continue
        title, body = prepare_body(str(rec.get("title") or ""), str(rec.get("body") or ""))
        if not body.strip():
            refresh_skipped.append({"id": doc_id, "reason": "empty_body", "old_key": old_key})
            continue
        why = junk_reason(body, is_pubmed=False, dup_count=1)
        if why:
            refresh_skipped.append({"id": doc_id, "reason": f"junk_{why}", "old_key": old_key})
            continue
        key = content_key(body)
        if key == old_key:
            refresh_skipped.append({"id": doc_id, "reason": "unchanged", "old_key": old_key,
                                    "new_key": key})
            continue
        if key in base_key_to_anchor and base_key_to_anchor[key] != doc_id:
            refresh_skipped.append({"id": doc_id, "reason": "new_key_exists",
                                    "old_key": old_key, "new_key": key})
            continue
        if key in added_key_to_anchor:
            refresh_skipped.append({"id": doc_id, "reason": "new_key_added",
                                    "old_key": old_key, "new_key": key})
            continue
        # The old key is only safe to release because this is a unique anchor.
        if base_key_to_anchor.get(old_key) == doc_id:
            base_key_to_anchor.pop(old_key, None)
        base_key_to_anchor[key] = doc_id
        row = {
            "id": doc_id, "url": str(old["url"]), "host": str(old["host"]), "lang": str(old["lang"]),
            "source": str(old["source"]), "n_chars": len(body), "thin": len(body) < THIN_CHARS,
            "content_key": key, "dup_of": doc_id, "furniture_chars": 0, "n_passages": 0,
            "_title": title, "_body": body,
        }
        refresh_rows.append(row)
        refresh_map[doc_id] = row
        refresh_audit.append({
            "id": doc_id, "host": row["host"], "old_chars": int(old["n_chars"]),
            "new_chars": len(body), "gain_chars": len(body) - int(old["n_chars"]),
            "old_key": old_key, "new_key": key, "cov": float(rec.get("cov") or 0.0),
        })
    log(f"refreshed accepted {len(refresh_rows):,}; skipped {len(refresh_skipped):,}")

    # ---- passages / sections / furniture for changed docs
    pas_rows: list[dict] = []
    sec_rows: list[dict] = []
    fur_rows: list[dict] = []
    for row in add_rows + refresh_rows:
        meta, pas, sec, fur = compute_spans(
            row["id"], row["_body"], row["host"], model, args.max_chars, args.min_chars,
            args.section_max_chars, args.section_min_chars,
        )
        row.update(meta)
        pas_rows.extend(pas)
        sec_rows.extend(sec)
        fur_rows.extend(fur)
    log(f"recomputed spans: passages {len(pas_rows):,}, sections {len(sec_rows):,}, furniture {len(fur_rows):,}")

    # ---- corpus: stream v2, replace refresh bodies, append added part
    base_parts = sorted((base / "corpus").glob("part-*.jsonl.gz"))
    if not base_parts:
        raise SystemExit(f"no corpus parts in {base / 'corpus'}")
    seen_refresh: set[int] = set()
    base_corpus_docs = 0
    for p in base_parts:
        dst = out / "corpus" / p.name
        with gzip.open(p, "rb") as src, gzip.open(dst, "wb", compresslevel=5) as dstf:
            for line in src:
                if not line.strip():
                    continue
                base_corpus_docs += 1
                rec = orjson.loads(line)
                doc_id = int(rec["id"])
                if doc_id in refresh_map:
                    row = refresh_map[doc_id]
                    dstf.write(orjson.dumps({"id": doc_id, "t": row["_title"], "x": row["_body"]}) + b"\n")
                    seen_refresh.add(doc_id)
                else:
                    dstf.write(line if line.endswith(b"\n") else line + b"\n")
        log(f"  corpus {p.name}: base {base_corpus_docs:,} docs")
    missing_refresh = set(refresh_map) - seen_refresh
    if missing_refresh:
        raise SystemExit(f"refresh ids missing from base corpus: {len(missing_refresh)}")
    next_part = max(int(p.stem.split("-")[1].split(".")[0]) for p in base_parts) + 1
    add_part = out / "corpus" / f"part-{next_part:02d}.jsonl.gz"
    with gzip.open(add_part, "wb", compresslevel=5) as f:
        for row in sorted(add_rows, key=lambda r: r["id"]):
            f.write(orjson.dumps({"id": row["id"], "t": row["_title"], "x": row["_body"]}) + b"\n")
    log(f"corpus: {base_corpus_docs:,} base + {len(add_rows):,} added in {add_part.name}")

    # ---- metadata layers
    meta = base_meta.copy()
    update_ids = {row["id"] for row in refresh_rows}
    if update_ids:
        idx = meta.index[meta["id"].astype("int64").isin(update_ids)]
        upd = {row["id"]: row for row in refresh_rows}
        for i in idx:
            row = upd[int(meta.at[i, "id"])]
            for col in ("n_chars", "thin", "content_key", "dup_of", "furniture_chars", "n_passages"):
                meta.at[i, col] = row[col]
    new_meta = pd.DataFrame([
        {k: v for k, v in row.items() if not k.startswith("_")} for row in add_rows
    ])
    if len(new_meta):
        meta = pd.concat([meta, new_meta], ignore_index=True)
    meta = meta.sort_values("id").reset_index(drop=True)
    if meta["id"].duplicated().any():
        raise SystemExit("duplicate ids in output metadata")
    write_parquet(meta, out / "docs_meta.parquet", META_SCHEMA)
    unique_meta = meta[meta["dup_of"] == meta["id"]].copy()
    if unique_meta["content_key"].duplicated().any():
        raise SystemExit("duplicate content_key in docs_meta_unique")
    write_parquet(unique_meta, out / "docs_meta_unique.parquet", META_SCHEMA)
    log(f"meta {len(meta):,}; unique {len(unique_meta):,}")

    base_pas = pd.read_parquet(base / "passages.parquet")
    base_sec = pd.read_parquet(base / "sections.parquet")
    base_fur = pd.read_parquet(base / "furniture.parquet")
    changed = set(refresh_map) | {row["id"] for row in add_rows}
    pas = pd.concat([base_pas[~base_pas["doc_id"].isin(changed)], pd.DataFrame(pas_rows)],
                    ignore_index=True).sort_values(["doc_id", "pid"]).reset_index(drop=True)
    sec = pd.concat([base_sec[~base_sec["doc_id"].isin(changed)], pd.DataFrame(sec_rows)],
                    ignore_index=True).sort_values(["doc_id", "sid"]).reset_index(drop=True)
    fur = pd.concat([base_fur[~base_fur["doc_id"].isin(changed)], pd.DataFrame(fur_rows)],
                    ignore_index=True).sort_values(["doc_id", "start"]).reset_index(drop=True)
    write_parquet(pas, out / "passages.parquet", PAS_SCHEMA)
    write_parquet(sec, out / "sections.parquet", SEC_SCHEMA)
    write_parquet(fur, out / "furniture.parquet", FUR_SCHEMA)
    log(f"passages {len(pas):,}; sections {len(sec):,}; furniture {len(fur):,}")

    shutil.copy2(base / "templates.json", out / "templates.json")

    # ---- audits
    dup = pd.concat([pd.read_parquet(base / "delta_duplicates.parquet"),
                     pd.DataFrame(add_duplicates)], ignore_index=True)
    if len(dup):
        dup = dup.sort_values("id").reset_index(drop=True)
    write_parquet(dup, out / "delta_duplicates.parquet", DUP_SCHEMA)
    skip_rows = [
        {
            "id": int(x["id"]),
            "host": str(x.get("host") or ""),
            "reason": str(x.get("reason") or ""),
            "old_key": str(x.get("old_key") or ""),
            "new_key": str(x.get("new_key") or ""),
            "old_chars": int(x.get("old_chars") or 0),
            "new_chars": int(x.get("new_chars") or 0),
        }
        for x in refresh_skipped
    ]
    write_parquet(pd.DataFrame(skip_rows, columns=SKIP_SCHEMA.names).sort_values("id").reset_index(drop=True),
                  out / "refresh_skipped.parquet", SKIP_SCHEMA)
    audit_rows = [
        {
            "id": int(x["id"]), "host": str(x.get("host") or ""),
            "old_chars": int(x.get("old_chars") or 0), "new_chars": int(x.get("new_chars") or 0),
            "gain_chars": int(x.get("gain_chars") or 0), "old_key": str(x.get("old_key") or ""),
            "new_key": str(x.get("new_key") or ""), "cov": float(x.get("cov") or 0.0),
        }
        for x in refresh_audit
    ]
    write_parquet(pd.DataFrame(audit_rows, columns=REFRESH_AUDIT_SCHEMA.names).sort_values("id").reset_index(drop=True),
                  out / "refresh_audit.parquet", REFRESH_AUDIT_SCHEMA)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = {
        "base": str(base), "version": args.version, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection": selection,
        "added": {
            "input": len(add_in), "accepted": len(add_rows),
            "by_lang": dict(Counter(row["lang"] for row in add_rows)),
            "by_host": dict(Counter(row["host"] for row in add_rows).most_common()),
            "duplicates": len(add_duplicates),
            "skipped": dict(Counter(x["reason"] for x in add_skipped)),
            "chars": int(sum(row["n_chars"] for row in add_rows)),
        },
        "refresh": {
            "input": len(refresh_in), "accepted": len(refresh_rows),
            "by_host": dict(Counter(row["host"] for row in refresh_rows).most_common()),
            "skipped": dict(Counter(x["reason"] for x in refresh_skipped)),
            "old_chars": int(sum(x["old_chars"] for x in refresh_audit)),
            "new_chars": int(sum(x["new_chars"] for x in refresh_audit)),
            "gain_chars": int(sum(x["gain_chars"] for x in refresh_audit)),
        },
    }
    (out / "v2_1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- stats / manifest
    length = sec["end"] - sec["start"]
    vi = sec["doc_id"] <= 766_441
    section_stats = {
        "max_chars": args.section_max_chars, "min_chars": args.section_min_chars,
        "sections": int(len(sec)), "per_doc": round(len(sec) / max(sec["doc_id"].nunique(), 1), 3),
        "vi_len_quantiles": {str(q): int(v) for q, v in length[vi].quantile([.05, .25, .5, .75, .95]).items()},
        "over_max": int((length > args.section_max_chars).sum()),
    }
    counts = {
        "docs": int(len(meta)), "passages": int(len(pas)), "sections": int(len(sec)),
        "chars": int(meta["n_chars"].sum()), "furniture_chars": int(meta["furniture_chars"].sum()),
        "thin": int(meta["thin"].sum()), "exact_duplicates": int((meta["dup_of"] != meta["id"]).sum()),
        "by_lang": {str(k): int(v) for k, v in meta["lang"].value_counts().items()},
        "by_source": {str(k): int(v) for k, v in meta["source"].value_counts().items()},
        "furniture_share": round(int(meta["furniture_chars"].sum()) / max(int(meta["n_chars"].sum()), 1), 4),
        "unique_docs": int(len(unique_meta)),
    }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "r2ai2026", "scripts", "kaggle/common"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    manifest = {
        "version": args.version, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit, "git_dirty": dirty, "parent": base.name,
        "inputs": {
            "base/MANIFEST.json": sha16(base / "MANIFEST.json"),
            "base/docs_meta.parquet": sha16(base / "docs_meta.parquet"),
            "base/passages.parquet": sha16(base / "passages.parquet"),
            "base/sections.parquet": sha16(base / "sections.parquet"),
            "base/furniture.parquet": sha16(base / "furniture.parquet"),
            "base/templates.json": sha16(base / "templates.json"),
            "add/v2_1_add.jsonl.gz": sha16(args.add),
            "refresh/v2_1_refresh.jsonl.gz": sha16(args.refresh),
            "selection/v2_1_selection.json": sha16(args.selection),
        },
        "params": {
            "max_chars": args.max_chars, "min_chars": args.min_chars,
            "template_share": args.template_share, "seed": args.seed,
            "thin_chars": THIN_CHARS, "parts": len(base_parts) + 1,
            "add_part": add_part.name, "section_max_chars": args.section_max_chars,
            "section_min_chars": args.section_min_chars, "include_langs": ["vi", "zh"],
            "content_dedupe": "mark_dup_of",
        },
        "counts": counts,
        "sections": section_stats,
        "corpus_parts": [p.name for p in sorted((out / "corpus").glob("part-*.jsonl.gz"))],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    man["delta"] = report
    man["docs_meta_unique"] = {"docs": int(len(unique_meta))}
    (out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: counts[k] for k in ("docs", "unique_docs", "passages", "sections", "chars")},
                   ensure_ascii=False))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
