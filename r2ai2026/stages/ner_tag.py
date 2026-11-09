"""Medical NER over the docstore (titles, headings, lead) and the queries, for the entity index.

    python ner_tag.py --data DIR --out DIR [--model cbc-528a/BamiBERT-ViMedNER] [--limit N] [--batch 256]

DIR holds the docstore files (part-*.jsonl[.gz], passages.parquet); query.parquet is searched under --queries.
Vietnamese docs only (id <= 766441). Units per doc: title, each distinct passage heading, the first ~400 chars of
the body ("lead"); each unit is split on whitespace and cut into 100-word pieces (model trained at 128 tokens).
Labels (BIO): ten_benh, trieu_chung_benh, bien_phap_chan_doan, bien_phap_dieu_tri, nguyen_nhan_benh.
Outputs: doc_entities.parquet (doc_id, field, type, text), query_entities.parquet (qid, type, text), stats.json.
Source is ASCII-only (Kaggle decodes kernel source badly otherwise).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

VI_MAX_ID = 766441
PIECE_WORDS = 100
LEAD_CHARS = 400
PUNCT = " .,;:?!()[]{}\"'-*/"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: Path, pattern: str) -> list[str]:
    return sorted(glob.glob(str(root / "**" / pattern), recursive=True))


def pieces(text: str) -> list[list[str]]:
    words = text.split()
    return [words[i:i + PIECE_WORDS] for i in range(0, len(words), PIECE_WORDS)]


def decode(words: list[str], word_ids: list, pred: list[int], id2label: dict) -> list[tuple[str, str]]:
    """BIO over words (label of each word's first sub-token) -> [(type, surface text)]."""
    labels = ["O"] * len(words)
    seen = set()
    for pos, w in enumerate(word_ids):
        if w is not None and w not in seen:
            seen.add(w)
            labels[w] = id2label[pred[pos]]
    out, cur, typ = [], [], None
    for word, lab in zip(words, labels):
        if lab.startswith("B-") or (lab.startswith("I-") and typ != lab[2:]):
            if cur:
                out.append((typ, " ".join(cur)))
            cur, typ = [word], lab[2:]
        elif lab.startswith("I-"):
            cur.append(word)
        else:
            if cur:
                out.append((typ, " ".join(cur)))
            cur, typ = [], None
    if cur:
        out.append((typ, " ".join(cur)))
    clean = [(t, x.strip(PUNCT)) for t, x in out]
    return [(t, x) for t, x in clean if len(x) >= 2]


class Tagger:
    def __init__(self, model: str, batch: int):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch = batch

    @torch.inference_mode()
    def tag(self, items: list[tuple[object, list[str]]]) -> list[tuple[object, str, str]]:
        """items: (key, words) -> [(key, type, text)]."""
        out = []
        order = sorted(range(len(items)), key=lambda i: len(items[i][1]))  # length buckets: less padding
        for b in range(0, len(order), self.batch):
            idx = order[b:b + self.batch]
            words = [items[i][1] for i in idx]
            enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=192, padding=True,
                           return_tensors="pt")
            logits = self.model(**{k: v.to(self.dev) for k, v in enc.items()}).logits
            pred = logits.argmax(-1).cpu().tolist()
            for j, i in enumerate(idx):
                for typ, text in decode(words[j], enc.word_ids(j), pred[j], self.id2label):
                    out.append((items[i][0], typ, text))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=None, help="dir holding query.parquet (default: --data)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="cbc-528a/BamiBERT-ViMedNER")
    ap.add_argument("--limit", type=int, default=0, help="first N Vietnamese docs only (dry run)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tagger = Tagger(args.model, args.batch)
    log(f"model on {tagger.dev}; labels {sorted(set(tagger.id2label.values()))}")

    q = pd.read_parquet(find(args.queries or args.data, "query.parquet")[0])
    q_items = [(int(i), p) for i, t in zip(q["id"], q["query"]) for p in pieces(str(t))]
    q_ents = tagger.tag(q_items)
    pd.DataFrame(q_ents, columns=["qid", "type", "text"]).to_parquet(args.out / "query_entities.parquet", index=False)
    log(f"queries: {len(q_ents)} entities")

    heads = pd.read_parquet(find(args.data, "passages.parquet")[0], columns=["doc_id", "heading"])
    heads = heads[(heads["doc_id"] <= VI_MAX_ID) & (heads["heading"].fillna("") != "")].drop_duplicates()
    head_of: dict[int, list[str]] = heads.groupby("doc_id")["heading"].apply(list).to_dict()

    items, n_docs = [], 0
    for part in find(args.data, "part-*.jsonl*"):
        with (gzip.open if part.endswith(".gz") else open)(part, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["id"] > VI_MAX_ID:
                    continue
                units = [("title", d["t"] or "")] + [("heading", h) for h in head_of.get(d["id"], [])]
                units.append(("lead", (d["x"] or "")[:LEAD_CHARS]))
                for field, text in units:
                    items += [((d["id"], field), p) for p in pieces(text)]
                n_docs += 1
                if args.limit and n_docs >= args.limit:
                    break
        if args.limit and n_docs >= args.limit:
            break
    log(f"docs {n_docs:,}; units {len(items):,}")
    ents = tagger.tag(items)
    df = pd.DataFrame([(k[0], k[1], t, x) for k, t, x in ents], columns=["doc_id", "field", "type", "text"])
    df.to_parquet(args.out / "doc_entities.parquet", index=False)
    stats = {"docs": n_docs, "units": len(items), "doc_entities": len(df), "query_entities": len(q_ents),
             "docs_with_entity": int(df["doc_id"].nunique()), "by_type": df["type"].value_counts().to_dict(),
             "seconds": round(time.time() - T0)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
