"""Entity index: canonical entity keys -> docs, and the query entities that hit them.

    python entity_index.py --ner DIR --out DIR [--frame DIR] [--max-df 0.02]

--ner holds doc_entities.parquet + query_entities.parquet (ner_tag.py); --frame optionally holds
query_frame.parquet (query_frame.py: LLM entities and aliases). Normalisation of a surface string:
NFC, lowercase, edge punctuation stripped, generic head words dropped ("benh", "chung", "tinh trang", ...),
then diacritics stripped -> key. Keys shorter than 3 chars or without a letter are dropped.
Outputs:
  entity_vocab.parquet  (eid, key, type, surface, df)      type/surface = most frequent for the key
  doc_entity.parquet    (doc_id, eid, field)               deduplicated
  query_entity.parquet  (qid, key, eid, source, df)        source ner|llm|alias; eid -1 when the key is unseen
  stats.json            coverage of query entities by the doc vocabulary
--max-df only marks keys (df / docs) as generic in the vocab (column generic), nothing is dropped.
Source is ASCII-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

T0 = time.time()
EDGE = re.compile(r"^[\W_]+|[\W_]+$")
SPACE = re.compile(r"\s+")
# generic head words (after diacritic stripping): "benh tieu duong" and "tieu duong" are one entity
HEADS = ("can benh", "benh ly", "benh", "chung benh", "chung", "tinh trang", "hien tuong", "trieu chung",
         "dau hieu", "phuong phap", "thuoc")
HEAD_RE = re.compile(r"^(?:" + "|".join(h.replace(" ", r"\s") for h in HEADS) + r")\s+")
D_FOLD = str.maketrans({chr(0x111): "d", chr(0x110): "d"})


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: Path, name: str) -> str | None:
    hits = sorted(glob.glob(str(root / "**" / name), recursive=True))
    return hits[0] if hits else None


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text).translate(D_FOLD)


def key_of(surface: str) -> str:
    s = SPACE.sub(" ", unicodedata.normalize("NFC", str(surface or "")).lower()).strip()
    s = EDGE.sub("", s)
    k = fold(s)
    k2 = HEAD_RE.sub("", k)
    k = k2 if len(k2) >= 3 else k  # "benh lao" -> "lao"; a bare "benh" stays and is generic
    return k if len(k) >= 3 and re.search(r"[a-z]", k) else ""


def keys_of(series: pd.Series) -> pd.Series:
    uniq = pd.unique(series)
    m = {u: key_of(u) for u in uniq}  # far fewer unique surfaces than rows
    return series.map(m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ner", type=Path, required=True)
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-df", type=float, default=0.02)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    de = pd.read_parquet(find(args.ner, "doc_entities.parquet"))
    log(f"doc entity rows {len(de):,}; unique surfaces {de['text'].nunique():,}")
    de["key"] = keys_of(de["text"])
    de = de[de["key"] != ""]
    n_docs = de["doc_id"].nunique()

    top = (de.groupby(["key", "type"]).size().rename("n").reset_index()
           .sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key"))
    surf = (de.groupby(["key", "text"]).size().rename("n").reset_index()
            .sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key"))
    df_ = de.groupby("key")["doc_id"].nunique().rename("df")
    vocab = top[["key", "type"]].merge(surf[["key", "text"]].rename(columns={"text": "surface"}), on="key")
    vocab = vocab.merge(df_, left_on="key", right_index=True).sort_values("df", ascending=False)
    vocab.insert(0, "eid", range(len(vocab)))
    vocab["generic"] = vocab["df"] / n_docs > args.max_df
    vocab.to_parquet(args.out / "entity_vocab.parquet", index=False)
    eid = dict(zip(vocab["key"], vocab["eid"]))
    dfe = dict(zip(vocab["key"], vocab["df"]))

    doc_ent = de.assign(eid=de["key"].map(eid))[["doc_id", "eid", "field"]].drop_duplicates()
    doc_ent.to_parquet(args.out / "doc_entity.parquet", index=False)
    log(f"vocab {len(vocab):,} keys over {n_docs:,} docs; doc_entity rows {len(doc_ent):,}")

    rows = []
    qe = pd.read_parquet(find(args.ner, "query_entities.parquet"))
    rows += [(int(q), key_of(t), "ner") for q, t in zip(qe["qid"], qe["text"])]
    fr = find(args.frame, "query_frame.parquet") if args.frame else None
    if fr:
        f = pd.read_parquet(fr)
        for q, ents in zip(f["qid"], f["entities"]):
            for e in json.loads(ents or "[]"):
                rows.append((int(q), key_of(e.get("text", "")), "llm"))
                rows += [(int(q), key_of(a), "alias") for a in e.get("aliases") or []]
    qdf = pd.DataFrame(rows, columns=["qid", "key", "source"])
    qdf = qdf[qdf["key"] != ""].drop_duplicates(["qid", "key"])  # first source wins: ner, llm, alias
    qdf["eid"] = qdf["key"].map(eid).fillna(-1).astype(int)
    qdf["df"] = qdf["key"].map(dfe).fillna(0).astype(int)
    qdf.to_parquet(args.out / "query_entity.parquet", index=False)

    hit = qdf[qdf["eid"] >= 0]
    specific = hit[~hit["eid"].map(dict(zip(vocab["eid"], vocab["generic"])))]
    n_q = qdf["qid"].nunique()
    stats = {
        "doc_rows": int(len(de)), "docs_with_entity": int(n_docs), "vocab": int(len(vocab)),
        "generic_keys": int(vocab["generic"].sum()), "query_entities": int(len(qdf)),
        "by_source": qdf["source"].value_counts().to_dict(),
        "hit_rate_by_source": hit["source"].value_counts().div(qdf["source"].value_counts()).round(3).to_dict(),
        "queries_with_entity": int(n_q),
        "queries_with_hit": int(hit["qid"].nunique()),
        "queries_with_specific_hit": int(specific["qid"].nunique()),
        "median_df_of_hits": float(hit["df"].median()) if len(hit) else 0.0,
        "seconds": round(time.time() - T0),
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
