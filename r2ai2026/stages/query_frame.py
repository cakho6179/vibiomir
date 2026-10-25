"""Query frame with an open-weight LLM: entities + aliases, asked aspects, sub-questions, keywords, HyDE passage.

    python query_frame.py --data DIR --out DIR [--model Qwen/Qwen3.5-9B] [--batch 16] [--max-queries N]

One JSON object per query (greedy decoding). Fields:
  entities  [{"text", "type", "aliases"}]  type in disease|symptom|drug|test|procedure|body_part|other;
            "text" must occur in the query (flag in_query), aliases = other names (synonyms, lay/medical, EN term)
  aspects   subset of ASPECTS
  subqueries 1-5 short self-contained Vietnamese questions (one per asked point)
  keywords  Vietnamese keyword query for BM25
  query_zh  the question in Simplified Chinese, as a Chinese patient would ask it (Chinese health Q&A sites)
  keywords_zh / keywords_en  Chinese / English medical keywords for lexical search
  hyde      ~120-word Vietnamese passage that would answer the question (for dense retrieval); kept last so an
            answer cut by --max-new loses only part of it (salvage)
Writes query_frame.parquet (qid, raw, ok, + fields as JSON strings) and stats.json. Prompt is English so the
source stays ASCII (Kaggle decodes non-ASCII kernel source badly); outputs are Vietnamese.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
ASPECTS = ["symptoms", "causes", "diagnosis_tests", "treatment", "medication_dosage", "prevention",
           "diet_lifestyle", "complications_prognosis", "pregnancy_children", "other"]
TYPES = ["disease", "symptom", "drug", "test", "procedure", "body_part", "other"]
SYSTEM = (
    "You analyse Vietnamese patient questions for a medical search engine over Vietnamese health articles. "
    "Return ONLY one JSON object, no markdown, with keys:\n"
    '"entities": list of {"text": exact span copied from the question, "type": one of ' + "|".join(TYPES) + ', '
    '"aliases": up to 4 other names for it (Vietnamese synonyms, lay or medical names, English term)};\n'
    '"aspects": list from [' + ", ".join(ASPECTS) + '] that the question asks about;\n'
    '"subqueries": 1 to 5 short self-contained Vietnamese questions, one per point asked, each naming the entity;\n'
    '"keywords": Vietnamese keywords for lexical search (entities, aliases, key symptoms), space separated;\n'
    '"query_zh": the question translated into natural Simplified Chinese, as a patient would ask it;\n'
    '"keywords_zh": Simplified Chinese medical keywords (standard disease, symptom, drug and test names), '
    'space separated;\n'
    '"keywords_en": English medical keywords (standard terms), space separated;\n'
    '"hyde": a factual Vietnamese passage of about 120 words, written like a health article, that answers the '
    "question. Put hyde last.\nAll other text values in Vietnamese, except English aliases and the _zh / _en "
    "fields.")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def parse(raw: str) -> dict | None:
    """The JSON object, skipping any reasoning text printed before it ("Thinking Process: ...")."""
    first = re.search(r'\{\s*"entities"', raw)
    raw = raw[first.start():] if first else raw
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    if obj is None:
        obj = salvage(raw)
    return obj if isinstance(obj, dict) else None


def salvage(raw: str) -> dict | None:
    """Output cut by max_new inside the last field ("hyde"): close the object before it, keep the partial text."""
    start, cut = raw.find("{"), raw.rfind('"hyde"')
    if start < 0 or cut < start:
        return None
    try:
        obj = json.loads(raw[start:cut].rstrip().rstrip(",") + "}")
    except json.JSONDecodeError:
        return None
    tail = re.match(r'"hyde"\s*:\s*"(.*)', raw[cut:], re.S)
    obj["hyde"] = tail.group(1).rsplit(".", 1)[0] + "." if tail else ""
    return obj


FIELDS = ("entities", "aspects", "subqueries", "keywords", "query_zh", "keywords_zh", "keywords_en", "hyde")


def aliases_of(value) -> list[str]:
    """A list of names; a bare string ("a, b; c") is split, never iterated character by character."""
    items = re.split(r"[,;\n]", value) if isinstance(value, str) else (value or [])
    return [str(a).strip() for a in items if str(a).strip()][:4]


def text_of(value) -> str:
    if isinstance(value, list):  # keywords sometimes come back as a list of terms
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def empty_frame() -> dict:
    return {k: [] if k in ("entities", "aspects", "subqueries") else "" for k in FIELDS}


def normalise(obj: dict, query: str) -> dict:
    low = query.lower()
    ents = []
    for e in obj.get("entities") or []:
        if isinstance(e, dict) and str(e.get("text", "")).strip():
            text = str(e["text"]).strip()
            ents.append({"text": text, "type": e.get("type") if e.get("type") in TYPES else "other",
                         "aliases": aliases_of(e.get("aliases")), "in_query": text.lower() in low})
    subs = []
    for s in obj.get("subqueries") or []:
        if isinstance(s, dict):  # small models sometimes wrap each question: {"text": "..."}
            s = next((v for v in s.values() if isinstance(v, str)), "")
        if str(s).strip():
            subs.append(str(s).strip())
    subs = subs[:5]
    return {"entities": ents, "aspects": [a for a in (obj.get("aspects") or []) if a in ASPECTS],
            "subqueries": subs, **{k: text_of(obj.get(k)) for k in FIELDS[3:]}}


def load(name: str):
    """Tokenizer + model in fp16 (T4 has no bf16); split over GPUs only when there are several."""
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    gpu = torch.cuda.is_available()
    multi = gpu and torch.cuda.device_count() > 1
    kw = dict(dtype=torch.float16 if gpu else torch.float32, device_map="auto" if multi else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except (ValueError, KeyError):  # Qwen3.5 checkpoints are image-text-to-text; the text path still works
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    model.eval()
    if gpu and not multi:
        model.to("cuda")
    return tok, model


def generate(tok, model, queries: list[str], max_new: int) -> list[str]:
    """Greedy answers for one batch; thinking is switched off on Qwen3-family templates."""
    msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}] for q in queries]
    try:
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for m in msgs]
    except TypeError:
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    out = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip() for t in out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1300)
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(glob.glob(f"{args.data}/**/query.parquet", recursive=True)[0])
    if args.max_queries:
        queries = queries.head(args.max_queries)
    tok, model = load(args.model)
    log(f"loaded {args.model}")
    texts = queries["query"].astype(str).tolist()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths per batch
    raw: dict[int, str] = {}
    for lo in range(0, len(order), args.batch):
        idx = order[lo:lo + args.batch]
        for i, t in zip(idx, generate(tok, model, [texts[i] for i in idx], args.max_new)):
            raw[i] = t
        log(f"{len(raw)}/{len(texts)}")

    rows = []
    for i, (qid, q) in enumerate(zip(queries["id"], texts)):
        obj = parse(raw[i])
        frame = normalise(obj, q) if obj else empty_frame()
        rows.append({"qid": int(qid), "ok": obj is not None, "raw": raw[i],
                     **{k: json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v for k, v in frame.items()}})
    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "query_frame.parquet", index=False)
    ents = [e for r in rows for e in json.loads(r["entities"])]
    stats = {"queries": len(df), "parsed": int(df["ok"].sum()),
             "entities_per_q": round(len(ents) / len(df), 2),
             "entities_in_query": round(sum(e["in_query"] for e in ents) / max(len(ents), 1), 3),
             "subqueries_per_q": round(sum(len(json.loads(r["subqueries"])) for r in rows) / len(df), 2),
             **{f"with_{k}": int(sum(bool(r[k]) for r in rows)) for k in FIELDS[3:]},
             "seconds": round(time.time() - T0)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
