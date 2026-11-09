"""Training-free yes/no relevance decisions with AnyJev (Decider) over (query, passage) pairs.

    python decide.py --data DIR --out DIR --model PATH_OR_ID [--level L0] [--prior none] [--dtype float16]
                     [--shard 0 --nshards 1] [--max-chars 1500] [--batch 16] [--device-map auto] [--offline]

DIR holds pairs*.parquet (build_pairs.py: qid, doc_id, rank, text) and query.parquet, the same input as judge.py.
State = the patient question + the passage (cut at --max-chars). --questions picks from QUESTIONS (comma list,
one model load): noul questions ("helps", "direct") give p_yes / logodds; "graded" is an ordered 5-level score
(never permuted, one forward) whose expected level is the score. For noul, AnyJev reads the
restricted Yes/No softmax at the answer position; L0 reads it under both phrasings ("Answer Yes or No." /
"Answer No or Yes.") and averages in log space, so the answer cannot depend on the label order. The batch prior
is off by default: it accumulates across calls, so scores would depend on batch order and sharding, and it pulls
toward 50/50 on pools where few pairs are relevant. With L0 and no prior the two phrasings are scored through one
prefix forward (decide_shared, same numbers as the Decider); other settings go through the Decider.

--model takes a local directory (glob allowed, the first match holding config.json) or a hub id; --offline sets
HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE before transformers is imported, so nothing is fetched. anyjev comes from
vendor/anyjev-*.whl when it is not installed. With 2 GPUs and no --device-map the shard is split in two halves,
one subprocess per GPU (as judge.py); --device-map auto instead spreads one model over every GPU (Qwen3-8B fp16
does not fit one T4). Output per question (in OUT/<question>/ when several are asked): scores_<tag>.parquet
(qid, doc_id, rank, score, + noul: p_yes, logodds, p_yes_yes_first, p_yes_no_first | graded: p_l0..p_l4; mass)
and stats_<tag>.json. score = logodds (noul) or expected level (graded). Source is ASCII-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = {
    # measured 25/10 on Qwen3-4B, G0 shard 0 (6000 pairs): Yes on 97.6%, too lenient to rank a top-150 list
    "helps": ("noul", "Does the document contain information that helps answer at least one of the points asked in "
                      "the patient's question?", None),
    "direct": ("noul", "Does the document directly answer the patient's main question, about the same specific "
                       "condition and the specific point asked, rather than only discussing a related topic?", None),
    "graded": ("score", "How well does the document answer the patient's question?",
               ["unrelated to the question",
                "same general health area, but a different specific problem",
                "same specific problem, but does not address what is asked",
                "same specific problem and partly answers what is asked",
                "same specific problem and directly answers what is asked"]),
    # Measured 28/10 on the tail sample (ranks 150-600): "direct" saturates there too -- median logodds 6.0,
    # 5 docs/query at logodds >= 8 against a 2 % gold density. The model is answering "is this on the topic?",
    # so the three below try to force the stricter "is this the answer?" reading. `strict` adds the
    # not-merely-a-general-discussion clause; `only_source` makes sufficiency the test; `expert_pick` asks for
    # the selection decision itself, which is what the gold set actually encodes.
    "strict": ("noul", "Does this document contain the specific fact, diagnosis, treatment, or recommendation the "
                       "patient asked about, rather than only a general discussion of the same disease?", None),
    "only_source": ("noul", "Could the patient's question be answered using only the information in this document, "
                            "without needing any other source?", None),
    "expert_pick": ("noul", "Would a medical expert choose this document as one of the few best sources to answer "
                            "this exact question, rather than as background reading?", None),
}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find(root: Path, name: str) -> str:
    return sorted(glob.glob(str(root / "**" / name), recursive=True))[0]


def import_anyjev():
    try:
        import anyjev  # noqa: F401
    except ImportError:
        wheels = (sorted(ROOT.glob("vendor/anyjev-*.whl")) or sorted(Path(__file__).parent.glob("anyjev-*.whl"))
                  or sorted(glob.glob("/kaggle/input/**/anyjev-*.whl", recursive=True)))
        if not wheels:
            raise SystemExit("anyjev is not installed and no vendor/anyjev-*.whl was found")
        sys.path.insert(0, str(wheels[-1]))
    from anyjev import Decider, Question
    return Decider, Question


def resolve_model(model: str) -> str:
    if "*" not in model:
        return model
    hits = sorted(Path(p).parent for p in glob.glob(str(Path(model) / "config.json"), recursive=True))
    if not hits:
        raise SystemExit(f"no model directory matches {model}")
    return str(hits[0])


def render(query: str, text: str, max_chars: int) -> str:
    return f"Patient question (Vietnamese):\n{query}\n\nDocument:\n{text[:max_chars]}"


def fix_sharded(backend) -> None:
    """anyjev 0.3.0 score_shared indexes the backbone output with tensors on model.device (the first GPU), but
    under device_map the last hidden state and the lm_head output live on the last GPU. Bring both back."""
    import types

    dev = backend.model.device
    inner, project = backend.backbone, backend._project

    def backbone(*a, **kw):
        out = inner(*a, **kw)
        out.last_hidden_state = out.last_hidden_state.to(dev)
        return out

    backend.backbone = backbone
    backend._project = types.MethodType(lambda self, h: project(h).to(dev), backend)


def make_decider(backend, level: str, prior: str, name: str = "helps", shared: bool = True):
    Decider, Question = import_anyjev()
    decider = Decider(backend, level=level, prior=prior, shared_prefix=shared)
    kind, text, levels = QUESTIONS[name]
    question = Question.noul(text, name=name) if kind == "noul" else Question.score(text, levels=levels, name=name)
    return decider, question


def row(p_yes_by_phrase: list[float], p_no_by_phrase: list[float], mass: float) -> dict:
    """L0 with no prior: average log p over the phrasings, renormalise over Yes/No."""
    z_yes = sum(math.log(max(p, 1e-12)) for p in p_yes_by_phrase) / len(p_yes_by_phrase)
    z_no = sum(math.log(max(p, 1e-12)) for p in p_no_by_phrase) / len(p_no_by_phrase)
    p = 1 / (1 + math.exp(z_no - z_yes))
    return {"score": z_yes - z_no, "p_yes": p, "logodds": z_yes - z_no, "p_yes_yes_first": p_yes_by_phrase[0],
            "p_yes_no_first": p_yes_by_phrase[-1], "mass": mass}


def decide_shared(backend, question, states: list[str], chunk: int = 512) -> pd.DataFrame:
    """Same numbers as Decider(level="L0", prior="none") on a noul question, one prefix forward per state.
    The Decider groups prompts by their label-id order, and the two phrasings of a noul question list Yes/No
    in opposite orders, so it never shares their prefix and runs two full forwards per pair. Here both
    phrasings are scored through backend.score_shared, with the ids in option order (Yes, No) for both."""
    from anyjev.readout import DEFAULT_SYSTEM, build_prompt, render_chat_parts, resolve_labels

    labels, ids = resolve_labels(backend.tokenizer, question)
    rows = []
    for start in range(0, len(states), chunk):
        groups = []
        for st in states[start:start + chunk]:
            parts = [render_chat_parts(backend.tokenizer, build_prompt(st, question, perm, DEFAULT_SYSTEM, labels))
                     for perm in ([0, 1], [1, 0])]
            if st[-200:] not in parts[0][0]:  # e.g. Qwen3-Reranker's template drops the user turn
                raise SystemExit("the chat template does not pass the state through; use a chat (instruct) model")
            if parts[0][0] != parts[1][0] or not parts[0][1]:
                raise SystemExit("phrasings do not share a prefix under this chat template; use --prior content_free")
            groups.append((parts[0][0], [parts[0][1], parts[1][1]]))
        for lps in backend.score_shared(groups, [ids] * len(groups)):
            ps = [[math.exp(v) for v in lp] for lp in lps]           # per phrasing: [p(Yes), p(No)], full vocab
            norm = [(a / (a + b), b / (a + b)) for a, b in ps]
            rows.append(row([n[0] for n in norm], [n[1] for n in norm], sum(a + b for a, b in ps) / len(ps)))
        if start // chunk % 20 == 0:
            log(f"{min(start + chunk, len(states)):,}/{len(states):,} pairs")
    return pd.DataFrame(rows)


def decide(decider, question, states: list[str], chunk: int = 512) -> pd.DataFrame:
    """Through the Decider (any level / prior). One row per state. noul: p_yes, its log-odds, p_yes under each
    phrasing; score: expected level and the level distribution. Both: label mass in the full vocabulary."""
    rows = []
    for start in range(0, len(states), chunk):
        for d in decider.decide_batch(states[start:start + chunk], question):
            diag = d.diagnostics
            if question.kind == "score":
                rows.append({"score": d.value, **{f"p_l{i}": float(x) for i, x in enumerate(d.probs)},
                             "mass": diag["answer_mass"]})
                continue
            # p_pos_raw[s, j]: probability of the label shown at position j under phrasing s; perms[s][j] is
            # its option index (0 = Yes)
            by_phrase = [float(p[list(perm).index(0)]) for p, perm in zip(diag["p_pos_raw"], diag["perms"])]
            p = min(max(d.p_true, 1e-12), 1 - 1e-12)
            rows.append({"score": math.log(p / (1 - p)), "p_yes": d.p_true, "logodds": math.log(p / (1 - p)),
                         "p_yes_yes_first": by_phrase[0], "p_yes_no_first": by_phrase[-1],
                         "mass": diag["answer_mass"]})
        if start // chunk % 20 == 0:
            log(f"{min(start + chunk, len(states)):,}/{len(states):,} pairs")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", default="L0", choices=["L0", "raw"])
    ap.add_argument("--prior", default="none", choices=["none", "content_free", "batch"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device-map", default="", help="e.g. auto: one model over all GPUs, no halves")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--half", type=int, default=-1, help="internal: which GPU half of the shard")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-shared", action="store_true",
                    help="score every prompt in full: the prefix KV cache cannot be repeated for models with linear "
                         "attention layers (Qwen3.5)")
    ap.add_argument("--questions", default="helps", help="comma list of " + ",".join(QUESTIONS))
    args = ap.parse_args()
    names = [x for x in args.questions.split(",") if x]
    if unknown := set(names) - set(QUESTIONS):
        raise SystemExit(f"unknown questions {sorted(unknown)}")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.out.mkdir(parents=True, exist_ok=True)
    outs = {n: (args.out / n if len(names) > 1 else args.out) for n in names}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    model = resolve_model(args.model)

    pairs = pd.read_parquet(find(args.data, "pairs*.parquet"))
    qids = sorted(pairs["qid"].unique())
    mine = set(qids[args.shard::args.nshards])  # shard by query so a query's list is scored in one place
    pairs = pairs[pairs["qid"].isin(mine)].reset_index(drop=True)
    if args.limit:
        pairs = pairs.head(args.limit)

    import torch
    n_gpu = torch.cuda.device_count()
    if args.half < 0 and n_gpu > 1 and not args.device_map:  # one subprocess per GPU, then merge
        procs = []
        for h in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(h))
            procs.append(subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--half", str(h)], env=env))
        if any(p.wait() for p in procs):
            raise SystemExit("a GPU worker failed")
        for out in outs.values():
            parts = [pd.read_parquet(out / f"scores_{args.shard}_h{h}.parquet") for h in range(2)]
            st = [json.loads((out / f"stats_{args.shard}_h{h}.json").read_text()) for h in range(2)]
            pd.concat(parts).to_parquet(out / f"scores_{args.shard}.parquet", index=False)
            stats = {**st[0], "pairs": sum(s["pairs"] for s in st), "seconds": round(time.time() - T0),
                     "shared_fallbacks": sum(s["shared_fallbacks"] for s in st)}
            (out / f"stats_{args.shard}.json").write_text(json.dumps(stats, indent=1))
            log(json.dumps(stats))
        return

    if args.half >= 0:
        pairs = pairs.iloc[args.half::2].reset_index(drop=True)
    q = pd.read_parquet(find(args.data, "query.parquet"))
    queries = {int(i): str(t) for i, t in zip(q["id"], q["query"])}
    states = [render(queries[int(qid)], str(t), args.max_chars) for qid, t in zip(pairs["qid"], pairs["text"])]
    log(f"shard {args.shard}/{args.nshards} half {args.half}: {len(pairs):,} pairs, model {model}")

    import_anyjev()
    from anyjev.backends.hf import HFBackend
    cuda = n_gpu > 0
    dtype = args.dtype if cuda else "float32"  # fp16 matmuls are not supported / slow on CPU
    kw = {"device_map": args.device_map} if args.device_map else {}
    backend = HFBackend(model, device="cuda" if cuda else "cpu", dtype=dtype, batch_size=args.batch, **kw)
    if args.device_map:
        fix_sharded(backend)
    tag = f"{args.shard}_h{args.half}" if args.half >= 0 else f"{args.shard}"
    for name, out in outs.items():
        t0 = time.time()
        decider, question = make_decider(backend, args.level, args.prior, name, shared=not args.no_shared)
        shared = question.kind == "noul" and args.level == "L0" and args.prior == "none" and not args.no_shared
        scored = decide_shared(backend, question, states) if shared else decide(decider, question, states)
        res = pd.concat([pairs[["qid", "doc_id", "rank"]], scored], axis=1)
        res.to_parquet(out / f"scores_{tag}.parquet", index=False)
        stats = {"pairs": len(res), "seconds": round(time.time() - t0), "model": model, "level": args.level,
                 "prior": args.prior, "path": "shared" if shared else "decider", "dtype": dtype,
                 "max_chars": args.max_chars, "question": name, "text": question.text,
                 "shared_fallbacks": int(getattr(backend, "shared_fallbacks", 0)), "decider": decider.stats,
                 "mass_median": float(res["mass"].median()) if len(res) else None}
        (out / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1))
        log(json.dumps(stats))


if __name__ == "__main__":
    main()
