"""Query-only medical NER with BamiBERT-ViMedNER (Kaggle GPU, fast).

Reads query.parquet from the attached dataset, tags every query with the five
BamiBERT entity types, and writes query_entities.parquet + stats.json.

    python ner_query.py --queries /kaggle/input/vibiomir-query-only --out /kaggle/working
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PIECE_WORDS = 100
PUNCT = " .,;:?!()[]{}\"'-*/"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: Path, pattern: str) -> list[str]:
    hits = sorted(glob.glob(str(root / "**" / pattern), recursive=True))
    direct = root / pattern
    if direct.exists() and str(direct) not in hits:
        hits.insert(0, str(direct))
    return hits


def pieces(text: str) -> list[list[str]]:
    words = text.split()
    return [words[i:i + PIECE_WORDS] for i in range(0, len(words), PIECE_WORDS)]


def decode_offsets(text: str, offsets: list, pred: list[int], id2label: dict) -> list[tuple[str, str]]:
    """Merge BIO token labels into character spans using tokenizer offsets."""
    out: list[tuple[str, str]] = []
    cur: list | None = None
    for (start, end), label_id in zip(offsets, pred):
        if start == end:  # special / padding token
            continue
        lab = id2label[int(label_id)]
        if lab.startswith("B-") or (lab.startswith("I-") and cur is not None and cur[2] != lab[2:]):
            if cur is not None:
                out.append((cur[2], text[cur[0]:cur[1]]))
            cur = [start, end, lab[2:]]
        elif lab.startswith("I-") and cur is not None:
            cur[1] = end
        else:
            if cur is not None:
                out.append((cur[2], text[cur[0]:cur[1]]))
            cur = None
    if cur is not None:
        out.append((cur[2], text[cur[0]:cur[1]]))
    clean = [(typ, value.strip(PUNCT)) for typ, value in out]
    return [(typ, value) for typ, value in clean if len(value) >= 2]


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
    def tag(self, items: list[tuple[int, list[str]]]) -> list[tuple[int, str, str]]:
        out: list[tuple[int, str, str]] = []
        for start in range(0, len(items), self.batch):
            chunk = items[start:start + self.batch]
            texts = [" ".join(words) for _, words in chunk]
            enc = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            ).to(self.dev)
            offsets = enc.pop("offset_mapping").cpu().tolist()
            logits = self.model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
            for j, (qid, _words) in enumerate(chunk):
                n = min(len(preds[j]), len(offsets[j]))
                for typ, value in decode_offsets(texts[j], offsets[j][:n], preds[j][:n], self.id2label):
                    out.append((qid, typ, value))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--model", default="cbc-528a/BamiBERT-ViMedNER")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tagger = Tagger(args.model, args.batch)
    log(f"model on {tagger.dev}; labels {sorted(set(tagger.id2label.values()))}")

    hits = find(args.queries, "query.parquet")
    if not hits:
        print("input tree:", [str(p) for p in Path("/kaggle/input").rglob("*")][:50], flush=True)
        raise SystemExit("query.parquet not found")
    path = hits[0]
    log(f"queries: {path}")
    q = pd.read_parquet(path)
    items = []
    for qid, text in zip(q["id"], q["query"]):
        for p in pieces(str(text)):
            items.append((int(qid), p))
    ents = tagger.tag(items)
    df = pd.DataFrame(ents, columns=["qid", "type", "text"])
    df.to_parquet(args.out / "query_entities.parquet", index=False)
    stats = {
        "queries": int(q["id"].nunique()),
        "pieces": len(items),
        "entities": len(df),
        "queries_with_entity": int(df["qid"].nunique()),
        "by_type": df["type"].value_counts().to_dict(),
        "seconds": round(time.time() - T0),
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
