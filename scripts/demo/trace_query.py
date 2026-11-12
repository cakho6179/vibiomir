"""Step-by-step walkthrough of the production run for one question, following one real document from the crawl to
the submitted chunk. Reads only the shipped artifacts; CPU. It never makes a new answer and shows no relevance
labels: p(gold) is the model's estimate.

    python run.py --demo --prepare --qids 1,401,556     # once (~2 min): document text + chunk text for these questions
    python run.py --demo --qid 401                      # out/demo/q401.html: one step per screen, arrow keys to move
    python run.py --demo --qid 401 --doc 1341776        # follow another document of the list
    python run.py --demo --grid                         # once (~25 s): chunk of the top-100 documents of every question,
                                                        # read from submissions/N1_n125.zip (sha256 checked vs #7916)

From step 6 on, the 100 highest-p documents of the question are drawn as a fixed 10x10 grid, coloured by what each
step knows (language and best channel rank; rr3; p and the cut-lines; submitted chunk length), with a table below.
On the web site a cell opens the walkthrough for that document.

Steps: crawl -> docstore record -> index -> query frame -> retrieval channels -> pool -> rerank -> p(gold) and depth
-> chunk -> submission entry. Every number comes from the shipped files; the depth and the lists are checked
against the submitted lists (argmax E[F2] == N_q, top N_q == #7892 docs, top 1.25 N_q == #7916 docs, chunks), and the
chunk shown is checked to be an exact slice of the document. --prepare runs the materialize code embedded in the
production run (kernel bundle vibiomir-mat-st3-acc2) for the chunk text, it does not re-implement it.
"""
from __future__ import annotations

import argparse
import codecs
import glob
import gzip
import hashlib
import html
import json
import re
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from pgold_fit import f2_exp  # noqa: E402

OUT = ROOT / "out" / "demo"
DS = ROOT / "data/docstore/v2.1"
PG = ROOT / "runs/reports/pgold_feats/base2"
LISTS = {"pN3_base": ROOT / "kaggle/datasets/mat_lists22/lists.parquet",
         "N1_n125": ROOT / "kaggle/datasets/mat_lists30/lists.parquet"}
FRAME = ROOT / "kaggle/outputs/query-frame-v3/query_frame.parquet"
VARIANTS = ROOT / "kaggle/datasets/qvariants_v3_acc1/query_variants.parquet"
CHUNKS, DOCS = OUT / "chunks.parquet", OUT / "docs.parquet"
# the 100-document grid of steps 6-9: chunk length and a preview for every top-100 document with a submitted chunk
# (built once by --grid from the submitted file), and the docstore/index steps saved from pages rendered with the
# docstore present (used only when data/docstore/v2.1 is not on this machine)
TOP100, SNAP, GRID = OUT / "top100.parquet", OUT / "snapshot.json", 100
SUBMISSION = ROOT / "submissions/N1_n125.zip"  # #7916: same chunk lists as #7892 (checked on every page)
CONV, GC, DEPTH = 0.77, 75 / 71, 1.25  # as in scripts/analysis/pn3_lists.py
LANG = {"vi": "VI", "zh": "ZH", "en": "EN"}
R, Q = "data/rankings_v21", "kaggle/outputs"
# retrieval channels that feed the pool (scripts/kaggle_ops/lang_pool_sets.py): (label, file, variant)
CHANNELS = {
    "vi": [("BGE-M3 dense · câu hỏi VI", f"{R}/m3_vi.parquet", None),
           ("BM25 VI-doc · câu hỏi VI", f"{R}/bm25_vi_doc.parquet", None),
           ("BM25 VI-section · câu hỏi VI", f"{R}/bm25_vi_sec.parquet", None)],
    "zh": [("BGE-M3 dense · câu hỏi VI", f"{R}/m3_zh.parquet", None),
           ("BGE-M3 dense · zh_q", f"{Q}/qv3-m3-zh/ranking.parquet", "zh_q"),
           ("BGE-M3 dense · zh_kw", f"{Q}/qv3-m3-zh/ranking.parquet", "zh_kw"),
           ("BM25 ZH · zh_q", f"{Q}/qv3-bm25-zh/ranking.parquet", "zh_q"),
           ("BM25 ZH · zh_kw", f"{Q}/qv3-bm25-zh/ranking.parquet", "zh_kw")],
    "en": [("BGE-M3 dense · câu hỏi VI", f"{R}/m3_en.parquet", None),
           ("BGE-M3 dense · en_kw", f"{Q}/qv3-m3-en/ranking.parquet", "en_kw"),
           ("BM25 EN · câu hỏi VI", f"{R}/bm25_en_viquery.parquet", None),
           ("BM25 EN · en_kw", f"{Q}/qv3-bm25-en/ranking.parquet", "en_kw")],
}
SOURCE = {"crawl": "crawler của đội", "tmh": "dữ liệu từ Hugging Face", "cc": "Common Crawl",
          "wayback": "Internet Archive Wayback", "direct": "recrawl trực tiếp", "pubmed": "NCBI E-utilities (PubMed)",
          "lc": "recrawl Long Châu"}


CACHE: dict | None = None  # the web server sets {}: each table is loaded once and indexed by its key column


def read(path: Path, qid: int, col: str = "qid", **kw) -> pd.DataFrame:
    if CACHE is None or kw:
        return pd.read_parquet(path, filters=[(col, "==", qid)], **kw)
    key = (str(path), col)
    if key not in CACHE:
        df = pd.read_parquet(path)
        CACHE[key] = (df, df.groupby(col).indices)
    df, idx = CACHE[key]
    return df.iloc[idx.get(qid, [])]


def read_many(pattern: str, qid: int) -> pd.DataFrame:
    parts = [read(Path(p), qid) for p in sorted(glob.glob(str(ROOT / pattern)))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def need(paths: list[Path], what: str) -> None:
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}: {what} (README, section 'Dữ liệu')")


# ------------------------------------------------------------------------------------------------ data of one question

def trace(qid: int, doc: int | None) -> dict:
    need([ROOT / "data/raw/btc/query.parquet", PG / "cands_p.parquet", *LISTS.values()], "unzip artifacts_A_pgold.zip")
    query = read(ROOT / "data/raw/btc/query.parquet", qid, "id")
    if query.empty:
        raise SystemExit(f"no question {qid} (ids 1..1200)")
    c = read(PG / "cands_p.parquet", qid).sort_values("p", ascending=False).reset_index(drop=True)
    nq = read(PG / "n_per_query.parquet", qid).iloc[0]
    sigma = json.loads((PG / "fit.json").read_text())["theta"]["sigma"]
    pool = read(ROOT / "kaggle/datasets/rerank_pool/cands.parquet", qid)
    lists = {j: read(p, qid) for j, p in LISTS.items()}
    lists = {j: d[d["job"] == j] for j, d in lists.items()}

    p = c["p"].to_numpy()
    f_doc = f2_exp(np.cumsum(p), nq["g_exp"], np.arange(1, len(p) + 1), sigma)
    pc = p[:1200] * CONV
    f_chunk = f2_exp(np.cumsum(pc), GC * nq["g_exp"], np.arange(1, len(pc) + 1), sigma)
    n_doc, n_chunk, n_deep = int(f_doc.argmax()) + 1, int(f_chunk.argmax()) + 1, int(round(nq["n_docs"] * DEPTH))

    def ids(job: str, kind: str) -> list[int]:
        d = lists[job]
        return d[d["kind"] == kind].sort_values("pos")["doc_id"].astype(int).tolist()

    top = c["doc_id"].astype(int).tolist()
    checks = {
        "N_q = argmax E[F2] của docs": (n_doc, int(nq["n_docs"])),
        "danh sách docs #7892 = N_q tài liệu p cao nhất": (top[:n_doc] == ids("pN3_base", "doc"), True),
        "danh sách docs #7916 = round(1,25·N_q) tài liệu p cao nhất": (top[:n_deep] == ids("N1_n125", "doc"), True),
        "n_c = argmax E[F2] của chunks": (n_chunk, len(ids("pN3_base", "chunk"))),
        "danh sách chunks = n_c tài liệu p cao nhất (#7892 và #7916)":
            (top[:n_chunk] == ids("pN3_base", "chunk") == ids("N1_n125", "chunk"), True),
        f"lưới {GRID} ô (theo p) khớp đầu danh sách docs #7892":
            (top[:min(n_doc, GRID)] == ids("pN3_base", "doc")[:min(n_doc, GRID)], True),
    }
    rr3 = read_many("kaggle/outputs/rr3-p*/rerank.parquet", qid)
    m3 = read_many("kaggle/outputs/m3m-p*/m3multi.parquet", qid)
    sc = c.merge(rr3[["doc_id", "score"]].rename(columns={"score": "rr3"}), on="doc_id", how="left")
    if not m3.empty:
        m3 = m3.assign(m3all=.4 * m3["dense"] + .2 * m3["sparse"] + .4 * m3["colbert"])
        sc = sc.merge(m3[["doc_id", "m3all", "dense", "sparse", "colbert"]], on="doc_id", how="left")
    sc = sc.merge(pool[["doc_id", "set", "rank"]].rename(columns={"rank": "pool_rank"}), on="doc_id", how="left")
    if (DS / "docs_meta.parquet").exists():
        meta = pd.read_parquet(DS / "docs_meta.parquet", filters=[("id", "in", top[:max(n_deep, 40)])],
                               columns=["id", "url", "host", "source"])
        sc = sc.merge(meta.rename(columns={"id": "doc_id"}), on="doc_id", how="left")
    docs = read(DOCS, qid) if DOCS.exists() else pd.DataFrame()
    chunks = read(CHUNKS, qid) if CHUNKS.exists() else pd.DataFrame()
    if doc is None:  # the highest-p document whose text was prepared (top 1 by default)
        have = set(docs["doc_id"].astype(int)) if not docs.empty else set()
        doc = next((d for d in top[:n_chunk] if d in have), top[0])
    t = dict(qid=qid, query=str(query.iloc[0]["query"]), c=sc, nq=nq, f_doc=f_doc, f_chunk=f_chunk, n_doc=n_doc,
             n_chunk=n_chunk, n_deep=n_deep, checks=checks, pool=pool, doc=int(doc), top=top,
             frame=read(FRAME, qid) if FRAME.exists() else pd.DataFrame(),
             variants=read(VARIANTS, qid) if VARIANTS.exists() else pd.DataFrame(), chunks=chunks)
    t["grid"] = grid_frame(qid, sc)
    t.update(follow(t, docs))
    t["story"] = chunk_story(t)
    return t


def channel_ranks(qid: int, lang: str) -> list[tuple[str, dict[int, int]]]:
    """Rank of every document in each retrieval channel of one language (1 = top)."""
    out = []
    for label, path, variant in CHANNELS.get(lang, []):
        f = ROOT / path
        if not f.exists():
            continue
        x = read(f, qid)
        if variant is not None and "variant" in x:
            x = x[x["variant"] == variant]
        x = x.sort_values("score", ascending=False)  # dense shards restart at rank 0
        out.append((label, dict(zip(x["doc_id"].astype(int), range(1, len(x) + 1)))))
    return out


def grid_frame(qid: int, sc: pd.DataFrame) -> pd.DataFrame:
    """The first GRID candidates in final p order, with their best channel rank and their submitted chunk."""
    g = sc.head(GRID).reset_index(drop=True)
    ch = read(TOP100, qid) if TOP100.exists() else pd.DataFrame(columns=["doc_id", "chunk_len", "preview"])
    g = g.merge(ch[["doc_id", "chunk_len", "preview"]], on="doc_id", how="left")  # keeps the p order
    ranks = {lang: channel_ranks(qid, lang) for lang in g["lang"].unique()}
    best, label, hits = [], [], []
    for d, lang in zip(g["doc_id"].astype(int), g["lang"]):
        rs = sorted((m[d], lab) for lab, m in ranks.get(lang, []) if d in m)
        best.append(rs[0][0] if rs else None)
        label.append(rs[0][1] if rs else "")
        hits.append(f"{len(rs)}/{len(ranks.get(lang, []))}")
    return g.assign(best_rank=best, best_ch=label, n_ch=hits)


def follow(t: dict, docs: pd.DataFrame) -> dict:
    """Everything about the followed document: docstore record, channel ranks, rerank profile, chunk span."""
    qid, d = t["qid"], t["doc"]
    row = t["c"][t["c"]["doc_id"] == d]
    if row.empty:
        raise SystemExit(f"doc {d} is not among the 2,500 candidates of question {qid}")
    r = row.iloc[0]
    lang = r["lang"]
    out = {"row": r, "pos": int(row.index[0]) + 1, "lang": lang}
    if (DS / "docs_meta.parquet").exists():
        meta = pd.read_parquet(DS / "docs_meta.parquet", filters=[("id", "==", d)])
        out["meta"] = meta.iloc[0] if not meta.empty else None
        for name in ("passages", "sections", "furniture"):
            p = DS / f"{name}.parquet"
            out[name] = read(p, d, "doc_id") if p.exists() else pd.DataFrame()
    ranks = []
    for label, path, variant in CHANNELS.get(lang, []):
        f = ROOT / path
        if not f.exists():
            continue
        x = read(f, qid)
        if variant is not None and "variant" in x:
            x = x[x["variant"] == variant]
        x = x.sort_values("score", ascending=False).reset_index(drop=True)  # dense shards restart at rank 0
        hit = x.index[x["doc_id"] == d]
        ranks.append((label, int(hit[0]) + 1 if len(hit) else None, len(x)))
    out["ranks"] = ranks
    if lang == "vi" and (ROOT / f"{R}/pool5000_vi.parquet").exists():
        x = read(ROOT / f"{R}/pool5000_vi.parquet", qid)
        hit = x[x["doc_id"] == d]
        out["pool5000"] = int(hit["rank"].iloc[0]) + 1 if len(hit) else None
    pas = ROOT / "kaggle/datasets/pas_scores/pas_scores.parquet"
    prof = read(pas, qid) if pas.exists() else pd.DataFrame(columns=["doc_id", "passage", "score"])
    out["profile"] = prof[prof["doc_id"] == d].sort_values("passage")
    body = docs[docs["doc_id"] == d] if not docs.empty else docs
    out["body"] = None
    if len(body):
        b = body.iloc[0]
        out["body"] = {"title": b["title"], "text": b["text"], "spans": json.loads(b["spans"])}
        ch = t["chunks"][t["chunks"]["doc_id"] == d] if not t["chunks"].empty else pd.DataFrame()
        if len(ch):
            out["chunk"] = locate(ch.iloc[0]["chunk_text"], b["title"], b["text"])
    return out


def locate(chunk: str, title: str, text: str) -> dict:
    """Where the submitted chunk sits in the body; it must be an exact slice (title kept when it starts the text)."""
    full = f"{title}\n{text}".strip()
    if chunk == full:
        return {"start": 0, "end": len(text), "whole": True, "text": chunk}
    if title and chunk.startswith(title.strip()):
        rest = chunk[len(title.strip()):].lstrip("\n")
        if text.startswith(rest):
            return {"start": 0, "end": len(rest), "whole": False, "text": chunk}
    k = text.find(chunk)
    if k < 0:
        raise SystemExit("the submitted chunk is not a slice of the document: replay does not match")
    return {"start": k, "end": k + len(chunk), "whole": False, "text": chunk}


# ------------------------------------------------------------------------------------------------ terminal

def lang_mix(langs: pd.Series) -> str:
    v = langs.map(LANG).value_counts()
    return " · ".join(f"{k} {v.get(k, 0)}" for k in ("VI", "ZH", "EN"))


def print_trace(t: dict) -> bool:
    ok = all(g == w for g, w in t["checks"].values())
    r = t["row"]
    print(f"\nCâu hỏi #{t['qid']}: {t['query'][:160]}")
    print(f"Theo dõi tài liệu {t['doc']} ({LANG.get(t['lang'], t['lang'])}, hạng p #{t['pos']}): "
          f"{getattr(t.get('meta'), 'url', '') if t.get('meta') is not None else ''}")
    for label, rank, n in t["ranks"]:
        print(f"  {label:32s} hạng {rank if rank else '—'} / {n}")
    print(f"  pool: {r.get('set')} #{int(r['pool_rank']) + 1 if pd.notna(r.get('pool_rank')) else '—'}"
          f" · rr3 {r.get('rr3', float('nan')):.2f} · m3all {r.get('m3all', float('nan')):.3f} · p {r['p']:.3f}")
    print(f"N_q = {t['n_doc']} (#7892), 1,25·N_q = {t['n_deep']} (#7916), n_c = {t['n_chunk']}; "
          f"ngôn ngữ N_q docs: {lang_mix(t['c']['lang'].head(t['n_doc']))}")
    for name, (got, want) in t["checks"].items():
        print(f"  {'OK ' if got == want else 'SAI'} {name}")
    if t.get("chunk"):
        print(f"  OK  chunk là lát cắt nguyên văn [{t['chunk']['start']}, {t['chunk']['end']}) của tài liệu")
    same = (t.get("story") or {}).get("same")
    if same is not None:
        print(f"  {'OK ' if same else 'SAI'} chunk_stair sản xuất cho đúng văn bản chunk đã nộp")
        ok &= same
    elif t["body"] is None:
        print(f"  (chưa có văn bản: chạy `python run.py --demo --prepare --qids {t['qid']}` một lần)")
    return ok


# ------------------------------------------------------------------------------------------------ page

E = html.escape


def fmt(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def bar(n: int, layers: list[str], h: int = 120, ticks: str = "") -> str:
    return (f'<svg viewBox="0 0 1000 {h}" width="100%" role="img" aria-label="thân bài, {n} ký tự">'
            f'<rect x="0" y="30" width="1000" height="44" fill="#ebe9e2"/>{"".join(layers)}{ticks}'
            f'<text x="0" y="{h - 8}" font-size="15" fill="#5b6570">0</text>'
            f'<text x="1000" y="{h - 8}" font-size="15" fill="#5b6570" text-anchor="end">{fmt(n)} ký tự</text></svg>')


def xs(a: int, b: int, n: int) -> tuple[float, float]:
    return 1000 * a / max(n, 1), max(1000 * (b - a) / max(n, 1), 1.5)


_SNAP: dict | None = None


def snapshot(t: dict, k: int) -> str | None:
    """Step k as rendered when the docstore was on this machine (same question and document), else None."""
    global _SNAP
    if (DS / "docs_meta.parquet").exists() or not SNAP.exists():
        return None
    if _SNAP is None:
        _SNAP = json.loads(SNAP.read_text(encoding="utf-8"))
    return _SNAP.get(f"{t['qid']}:{t['doc']}", {}).get(str(k))


NO_DS = ("<p class=note>Docstore v2.1 (URL, mục, khung trang) và các tệp chỉ mục không có trên máy này; "
         "phần dưới dùng văn bản đã chuẩn bị của tài liệu.</p>")

# ------------------------------------------------------------------------------------------------ 100-document grid

LANGC = {"vi": ("#1f5fa8", "#fff"), "zh": ("#e8a33d", "#14213d"), "en": ("#7fb0e8", "#14213d")}
MODES = {"lang": 6, "rr3": 7, "p": 8, "chunk": 9}  # grid colour of each step (step number for the cell links)


def heat(k: float) -> tuple[str, str]:
    a, b = (238, 242, 247), (20, 33, 61)
    c = tuple(int(x + (y - x) * k) for x, y in zip(a, b))
    return f"rgb{c}", "#fff" if k > .5 else "#1d232a"


def kchars(n: float) -> str:
    return fmt(n / 1000, 1) + "k"


def membership(t: dict, i: int) -> str:
    parts = ["docs #7892" if i < t["n_doc"] else ("docs #7916" if i < t["n_deep"] else "")]
    parts.append("chunk" if i < t["n_chunk"] else "")
    return " · ".join(p for p in parts if p) or "không nộp"


def grid(t: dict, mode: str) -> str:
    g, step = t["grid"], MODES[mode]
    lo = hi = 0.0
    if mode in ("rr3", "p"):
        v = g[mode].astype(float)
        lo, hi = float(v.min()), float(v.max())
    cells = []
    for r in g.itertuples():
        i, d = r.Index, int(r.doc_id)
        cls = ["c"]
        if mode == "lang":
            (bg, fg), sub = LANGC.get(r.lang, ("#ccc", "#000")), (f"#{int(r.best_rank)}" if pd.notna(r.best_rank) else "—")
        elif mode in ("rr3", "p"):
            val = getattr(r, mode)
            k = 0.0 if pd.isna(val) or hi <= lo else (float(val) - lo) / (hi - lo)
            (bg, fg), sub = heat(k), ("—" if pd.isna(val) else fmt(float(val), 2 if mode == "rr3" else 3))
        else:
            inside = i < t["n_chunk"]
            bg, fg = ("#d9822b", "#fff") if inside else ("#ebe9e2", "#5b6570")
            sub = kchars(r.chunk_len) if inside and pd.notna(r.chunk_len) else "—"
        cut = t["n_chunk"] if mode == "chunk" else t["n_doc"]
        if mode in ("p", "chunk") and i >= cut:
            cls.append("deep" if mode == "p" and i < t["n_deep"] else "dim")
        if d == t["doc"]:
            cls.append("me")
        rr3 = "—" if pd.isna(r.rr3) else f"{r.rr3:.2f}"
        pool = f"{r.set} #{int(r.pool_rank) + 1}" if pd.notna(r.pool_rank) else "—"
        best = f"{r.best_ch} #{int(r.best_rank)}" if pd.notna(r.best_rank) else "không kênh nào"
        tip = (f"#{i + 1} · tài liệu {d} · {LANG.get(r.lang, r.lang)} · p {r.p:.3f} · rr3 {rr3} · pool {pool}\n"
               f"kênh tốt nhất: {best} (có mặt ở {r.n_ch} kênh)\nvào bài: {membership(t, i)}"
               + (f" · đoạn trích {fmt(r.chunk_len)} ký tự" if pd.notna(r.chunk_len) and i < t["n_chunk"] else ""))
        inner = f"<b>{i + 1}</b><i>{LANG.get(r.lang, r.lang)}</i><span>{E(sub)}</span>"
        attrs = f'class="{" ".join(cls)}" style="background:{bg};color:{fg}" title="{E(tip)}"'
        cells.append(f'<a {attrs} href="?doc={d}#{step}">{inner}</a>' if t.get("web") else f"<div {attrs}>{inner}</div>")
    return f'<div class=grid>{"".join(cells)}</div>'


def grid_block(t: dict, mode: str) -> str:
    g, n = t["grid"], len(t["grid"])
    if mode == "lang":
        far = int((g["best_rank"].isna() | (g["best_rank"].astype(float) > 100)).sum())
        top = int((g["best_rank"].astype(float) <= 10).sum())
        pools = " · ".join(f"{k} {v}" for k, v in g["set"].value_counts().items())
        head = (f"<b>{n} tài liệu đầu của câu hỏi</b>, xếp theo p cuối cùng (giữ nguyên thứ tự ở các bước 6–9). "
                f"Màu = ngôn ngữ, số = hạng tốt nhất ở các kênh truy hồi cùng ngôn ngữ. {lang_mix(g['lang'])}; pool: {E(pools)}. "
                f"{top} tài liệu đứng top 10 ở ít nhất một kênh; {far} tài liệu không kênh nào xếp trong top 100, "
                "chúng lên được đây nhờ gộp và xếp hạng lại.")
        legend = "".join(f'<span style="background:{LANGC[k][0]}"></span>{LANG[k]}' for k in LANGC)
    elif mode == "rr3":
        v = g["rr3"].astype(float)
        rho = v.rank().corr(pd.Series(range(n), dtype=float), method="spearman")
        head = (f"Cùng {n} ô, màu = điểm <b>rr3</b> của bge-reranker-v2-m3 (đậm = cao, từ {fmt(v.min(), 2)} đến {fmt(v.max(), 2)}). "
                f"Thứ tự ô vẫn theo p: tương quan hạng giữa rr3 và thứ tự p là {fmt(-rho, 2)}, nên reranker là một tín hiệu "
                "mạnh nhưng không phải tín hiệu duy nhất (BGE-M3 ba đầu, ngôn ngữ, thứ hạng truy hồi).")
        legend = '<span style="background:rgb(238,242,247)"></span>thấp<span style="background:#14213d"></span>cao'
    elif mode == "p":
        cutline = (f"Ô 1–{t['n_doc']} vào danh sách docs (#7892); ô {t['n_doc'] + 1}–{min(t['n_deep'], n)} viền tím: chỉ #7916 "
                   f"(1,25·N_q); ô mờ: không nộp." if t["n_doc"] < n else
                   f"N_q = {t['n_doc']} ≥ {n}: cả lưới đều vào bài, ranh giới nằm ngoài lưới.")
        head = (f"Màu = <b>p(gold)</b>. Tổng p của {n} ô = {fmt(g['p'].sum(), 1)}; kỳ vọng số tài liệu đúng của câu hỏi "
                f"G = {fmt(t['nq']['g_exp'], 1)}. {cutline}")
        legend = ('<span style="background:rgb(238,242,247)"></span>p thấp<span style="background:#14213d"></span>p cao'
                  '<span style="border:2px dashed #6b4fa0;background:none"></span>chỉ #7916')
    else:
        c = g.head(min(t["n_chunk"], n))["chunk_len"].dropna()
        stats = (f"; dài trung bình {fmt(c.mean())} ký tự, ngắn nhất {fmt(c.min())}, dài nhất {fmt(c.max())}" if len(c) else
                 " (chạy `python run.py --demo --grid` để có độ dài đoạn trích)")
        head = (f"Ô cam = tài liệu có <b>đoạn trích</b> trong bài nộp (n_c = {t['n_chunk']}), số = độ dài đoạn trích{stats}. "
                "Mỗi đoạn là lát cắt nguyên văn ≤ 4.500 ký tự.")
        legend = '<span style="background:#d9822b"></span>có đoạn trích<span style="background:#ebe9e2"></span>không'
    tip = "Bấm một ô để theo dõi tài liệu đó; di chuột để xem chi tiết." if t.get("web") else "Di chuột lên ô để xem chi tiết."
    return (f'<div class=gwrap><p class=gh>{head}</p>{grid(t, mode)}<p class=legend>{legend}'
            f'<span style="outline:3px solid #e8c547;background:none"></span>tài liệu đang theo dõi · {tip}</p></div>')


def grid_table(t: dict) -> str:
    rows = []
    for r in t["grid"].itertuples():
        i, d = r.Index, int(r.doc_id)
        link = f'<a href="?doc={d}#{MODES["chunk"]}">{d}</a>' if t.get("web") else str(d)
        rows.append(
            f"<tr{' class=me' if d == t['doc'] else ''}><td>{i + 1}</td><td>{link}</td><td>{LANG.get(r.lang, r.lang)}</td>"
            f"<td>{E(str(r.set))} #{int(r.pool_rank) + 1 if pd.notna(r.pool_rank) else '—'}</td>"
            f"<td>{E(r.best_ch)} {('#' + str(int(r.best_rank))) if pd.notna(r.best_rank) else '—'}</td>"
            f"<td class=n>{'—' if pd.isna(r.rr3) else fmt(r.rr3, 2)}</td>"
            f"<td class=n>{'—' if pd.isna(getattr(r, 'm3all', np.nan)) else fmt(r.m3all, 3)}</td>"
            f"<td class=n><b>{fmt(r.p, 3)}</b></td><td>{membership(t, i)}</td>"
            f"<td class=n>{fmt(r.chunk_len) if pd.notna(r.chunk_len) and i < t['n_chunk'] else '—'}</td>"
            f"<td class=u>{E(str(r.preview)[:110]) + ' …' if isinstance(r.preview, str) and i < t['n_chunk'] else ''}</td></tr>")
    return (f"<details><summary>Bảng {len(rows)} tài liệu (cùng thứ tự với lưới)</summary><div class=tw><table>"
            "<tr><th>#</th><th>doc</th><th>NN</th><th>pool</th><th>kênh tốt nhất</th><th class=n>rr3</th><th class=n>m3all</th>"
            f"<th class=n>p</th><th>vào bài</th><th class=n>đoạn trích</th><th>đầu đoạn trích</th></tr>{''.join(rows)}</table></div></details>")


def step_crawl(t: dict) -> str:
    if (s := snapshot(t, 1)) is not None:
        return s
    m = json.loads((DS / "MANIFEST.json").read_text(encoding="utf-8"))["counts"] if (DS / "MANIFEST.json").exists() \
        else {"by_source": {}, "by_lang": {}, "docs": 0}
    total, urls = m["docs"], 1_220_598
    rows = "".join(
        f'<tr><td>{E(SOURCE.get(k, k))}</td><td class=n>{fmt(v)}</td>'
        f'<td><div class=hb style="width:{max(100 * v / max(m["by_source"].values()), .6):.1f}%"></div></td></tr>'
        for k, v in sorted(m["by_source"].items(), key=lambda kv: -kv[1]))
    src = getattr(t.get("meta"), "source", None)
    src = E(SOURCE.get(src, str(src))) if src is not None else "— (cần docstore)"
    return f"""<div class=two><div>
<p>Danh sách của BTC có <b>{fmt(urls)} URL</b> chính thức (VI, ZH, EN). Đội chỉ lấy văn bản của chính các URL này.</p>
<ul><li><b>Crawler bất đồng bộ</b> (<code>r2ai2026/crawl/fetch.py</code>): ít kết nối mỗi host, tôn trọng
<code>robots.txt</code>, trích nội dung bằng trafilatura và có đường lui lxml. Tiến độ lưu SQLite, dữ liệu lưu JSONL nối tiếp,
nên crawl 1,2 triệu URL dừng được và chạy tiếp được.</li>
<li><b>Cứu URL chết hoặc bị chặn</b> (bảng bên phải): Common Crawl (tra CDX, lấy bản ghi WARC theo byte),
Internet Archive Wayback, NCBI E-utilities cho trang PubMed, recrawl trực tiếp, và dữ liệu crawl công khai từ Hugging Face
của cùng danh sách URL (qua cổng chất lượng).</li>
<li>Còn <b>27.925 URL</b> không lấy được (403, rỗng).</li></ul>
<p class=note>Tài liệu đang theo dõi: <b>{t['doc']}</b>, nguồn <b>{src}</b>.</p>
</div><div><table class=src><tr><th>Nguồn</th><th class=n>Tài liệu</th><th></th></tr>{rows}
<tr><td><b>Docstore v2.1</b></td><td class=n><b>{fmt(total)}</b></td><td>{' · '.join(f'{LANG[k]} {fmt(v)}' for k, v in m['by_lang'].items())}</td></tr>
</table></div></div>"""


def step_docstore(t: dict) -> str:
    if (s := snapshot(t, 2)) is not None:
        return s
    meta, body = t.get("meta"), t["body"]
    if meta is None and not body:
        return (NO_DS.replace("phần dưới dùng văn bản đã chuẩn bị của tài liệu.", "tài liệu này chưa có văn bản đã chuẩn bị.")
                + preview_block(t))
    n = int(meta["n_chars"]) if meta is not None else (len(body["text"]) if body else 0)
    layers, ticks = [], ""
    for i, p in enumerate(t.get("passages", pd.DataFrame()).itertuples()):
        x, w = xs(p.start, p.end, n)
        layers.append(f'<rect x="{x:.1f}" y="30" width="{w:.1f}" height="44" fill="{"#9fbfe0" if i % 2 else "#c9dbef"}"/>')
    for f in t.get("furniture", pd.DataFrame()).itertuples():
        x, w = xs(f.start, f.end, n)
        layers.append(f'<rect x="{x:.1f}" y="30" width="{w:.1f}" height="44" fill="#c0392b" opacity=".75"/>')
    for s in t.get("sections", pd.DataFrame()).itertuples():
        x, _ = xs(s.start, s.end, n)
        ticks += f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="86" stroke="#1d232a" stroke-width="2"/>'
    sec = "".join(f"<li>{E(s.heading or '(mở đầu)')} <span class=note>[{fmt(s.start)}–{fmt(s.end)})</span></li>"
                  for s in t.get("sections", pd.DataFrame()).itertuples())
    text = ""
    if body:
        x = body["text"]
        cuts = sorted((int(f.start), int(f.end)) for f in t.get("furniture", pd.DataFrame()).itertuples())
        pieces, pos = [], 0
        for a, b in cuts:
            pieces += [E(x[pos:a]), f"<mark>{E(x[a:b])}</mark>"]
            pos = b
        pieces.append(E(x[pos:]))
        text = f'<pre class=doc><b>{E(body["title"])}</b>\n{"".join(pieces)}</pre>'
    fields = "" if meta is None else (
        f"<dl><dt>doc_id</dt><dd>{t['doc']}</dd><dt>URL</dt><dd>{E(meta['url'])}</dd><dt>Ngôn ngữ · nguồn</dt>"
        f"<dd>{LANG.get(meta['lang'], meta['lang'])} · {E(SOURCE.get(meta['source'], meta['source']))}</dd>"
        f"<dt>Thân bài</dt><dd>{fmt(n)} ký tự · {int(meta['n_passages'])} đoạn · {len(t.get('sections', []))} mục · khung trang {fmt(meta['furniture_chars'])} ký tự</dd></dl>")
    return f"""{NO_DS if meta is None else ''}<p>Mỗi URL thành <b>một bản ghi</b> <code>{{id, url, t, x}}</code>.
Văn bản được chuẩn hoá (entity, Unicode, khoảng trắng) nhưng <b>không viết lại</b>. Phần khung trang (menu, footer, CTA…)
được <b>đánh dấu</b> bằng span chứ không xoá. Vì thế mọi chunk nộp đều cắt nguyên văn được từ thân bài.</p>
{fields}{bar(n, layers, ticks=ticks)}
<p class=legend><span style="background:#c9dbef"></span><span style="background:#9fbfe0"></span>đoạn (<code>passages.parquet</code>: theo đoạn văn, ~1.500 ký tự)
<span style="background:#1d232a;width:3px"></span>ranh giới mục (<code>sections.parquet</code>: theo heading, ≤ 4.800 ký tự)
<span style="background:#c0392b"></span>khung trang (đánh dấu, không xoá)</p>
<p class=note>Mỗi chỉ mục cắt tài liệu theo đơn vị riêng của nó (bước Chỉ mục đếm trực tiếp trong các tệp chỉ mục).</p>
<details><summary>Các mục của tài liệu</summary><ul>{sec}</ul></details>
<details><summary>Thân bài (khung trang được tô đỏ)</summary>{text or '<p class=note>Chạy --prepare để có văn bản.</p>'}</details>"""


_IDX: dict = {}


def index_units(doc: int, lang: str) -> dict[str, int | None]:
    """Units of the document in each index, counted in the index files themselves (None: index files absent)."""
    def arr(p: Path) -> np.ndarray:
        if p not in _IDX:
            _IDX[p] = np.load(p, mmap_mode="r")
        return _IDX[p]

    out: dict[str, int | None] = {}
    base = DS / "index"
    for k in {"vi": ["vi_doc", "vi_sec"], "zh": ["zh"], "en": ["en"]}.get(lang, []):
        d, u = base / f"bm25/{k}/index/doc_ids.npy", base / f"bm25/{k}/index/unit_doc.npy"
        if d.exists() and u.exists():
            pos = np.flatnonzero(arr(d) == doc)
            out[k] = int((arr(u) == pos[0]).sum()) if len(pos) else 0
        else:
            out[k] = None
    ids = base / f"dense/{lang}/ids.npy"
    out["dense"] = int((arr(ids) == doc).sum()) if ids.exists() else None
    return out


def step_index(t: dict) -> str:
    if (s := snapshot(t, 3)) is not None:
        return s
    lang = t["lang"]
    units = index_units(t["doc"], lang)
    cards = [("vi_doc", "BM25 VI-doc", "732.246 tài liệu tiếng Việt · token âm tiết giữ dấu · k1 1,5, b 0,75", "tài liệu"),
             ("vi_sec", "BM25 VI-section", "1.208.509 mục theo heading, 2.000–4.800 ký tự", "mục"),
             ("zh", "BM25 ZH", "220.532 tài liệu tiếng Trung · tách từ jieba", "tài liệu"),
             ("en", "BM25 EN", "239.895 tài liệu tiếng Anh", "tài liệu"),
             ("dense", f"BGE-M3 dense {LANG.get(lang, lang)}",
              "đoạn 500 ký tự, chồng 100 · " if lang == "zh" else "đoạn ≤ 1.500 ký tự theo đoạn văn, có tiêu đề ở đầu · ", "vector")]
    total_dense = {"vi": "2.224.624", "zh": "557.732", "en": "337.744"}.get(lang, "")
    html_cards = ""
    for key, name, desc, unit in cards:
        if key == "dense":
            desc += f"{total_dense} vector 1.024 chiều, fp16 · điểm tài liệu = đoạn tốt nhất (MaxP)"
        on = key in units
        n = units.get(key)
        note = ("không thuộc chỉ mục này (khác ngôn ngữ)" if not on else
                "thiếu tệp chỉ mục trên máy" if n is None else f"tài liệu này: {n} {unit}")
        html_cards += (f'<div class="card{" on" if on else ""}"><h3>{E(name)}</h3><p>{E(desc)}</p>'
                       f'<p class=note>{E(note)}</p></div>')
    return f"""<p>Chỉ mục dựng <b>một lần</b> trên toàn bộ docstore. Lúc truy vấn chỉ cần mã hoá câu hỏi rồi tra.
Các ô được tô là những chỉ mục chứa tài liệu {t['doc']} ({LANG.get(lang, lang)}); số đơn vị đếm trực tiếp trong tệp chỉ mục.</p>
<div class=cards>{html_cards}</div>"""


def step_query(t: dict) -> str:
    if t["frame"].empty:
        return "<p class=note>Thiếu khung truy vấn (gói D).</p>"
    f = t["frame"].iloc[0]
    v = t["variants"]
    vrows = "" if v.empty else "".join(f"<tr><td>{E(r.variant)}</td><td>{LANG.get(r.lang, r.lang)}</td>"
                                       f"<td>{E(str(r.text)[:160])}</td></tr>" for r in v.itertuples())
    return f"""<div class=q>{E(t['query'])}</div>
<p>Qwen3.5-9B đọc câu hỏi một lần (giải mã tham lam) và trả về JSON:</p>
<dl><dt>Từ khoá VI</dt><dd>{E(f['keywords'])}</dd><dt>Câu hỏi ZH</dt><dd>{E(f['query_zh'])}</dd>
<dt>Từ khoá ZH</dt><dd>{E(f['keywords_zh'])}</dd><dt>Từ khoá EN</dt><dd>{E(f['keywords_en'])}</dd></dl>
<details><summary>{len(v)} biến thể truy vấn ({lang_mix(v['lang']) if not v.empty else ''})</summary>
<table>{vrows}</table></details>"""


def step_retrieval(t: dict) -> str:
    r = t["row"]
    rows = "".join(f"<tr><td>{E(a)}</td><td class=n>{('#' + fmt(k)) if k else '—'}</td><td class=n>{fmt(n)}</td></tr>"
                   for a, k, n in t["ranks"])
    pool = ", ".join(f"{k}: {v}" for k, v in t["pool"]["set"].value_counts().items())
    grid_html = grid_block(t, "lang") + grid_table(t)
    vi = ("<li>VI: RRF k=60, trọng số dense : BM25-doc : BM25-section = 2 : 1 : 1 → <b>pool5000_vi</b>"
          + (f" (tài liệu này: #{t['pool5000']})" if t.get("pool5000") else "") + ", lấy 1.500 đầu.</li>")
    return f"""{grid_html}<h3 class=fol>Tài liệu đang theo dõi: {t['doc']}</h3>
<div class=two><div><p>Tài liệu {t['doc']} trong từng kênh của ngôn ngữ {LANG.get(t['lang'], t['lang'])}
(hạng / độ dài danh sách):</p><table><tr><th>Kênh · biến thể</th><th class=n>Hạng</th><th class=n>Danh sách</th></tr>{rows}</table>
<p class=note>— = không có trong danh sách của kênh đó.</p></div><div>
<p><b>Gộp thành pool</b> (Reciprocal Rank Fusion, k = 60):</p><ul>{vi}
<li>ZH, EN: RRF trọng số đều trên mọi kênh × biến thể → 500 mỗi ngôn ngữ.</li></ul>
<div class=kpi><div>Pool của câu hỏi<b>{len(t['pool']):,}</b><span>{E(pool)}</span></div>
<div>Tài liệu {t['doc']}<b>{E(str(r.get('set')))} #{int(r['pool_rank']) + 1 if pd.notna(r.get('pool_rank')) else '—'}</b></div></div>
<p class=note>Hai bài thăm dò cùng 5.000 tài liệu/câu hỏi: chỉ VI → docs recall 0,7317; VI 3.000 + ZH 1.000 + EN 1.000 → 0,936.</p>
</div></div>"""


def step_rerank(t: dict) -> str:
    r, prof, body = t["row"], t["profile"], t["body"]
    layer, note = [], ""
    if body and len(prof):
        n, s = len(body["text"]), prof["score"].to_numpy()
        lo, hi = float(s.min()), float(s.max())
        for (a, b), v in zip(body["spans"], s):
            x, w = xs(a, b, n)
            k = (v - lo) / (hi - lo) if hi > lo else 1
            layer.append(f'<rect x="{x:.1f}" y="30" width="{w:.1f}" height="44" fill="rgb({int(225 - 190 * k)},'
                         f'{int(232 - 140 * k)},{int(242 - 70 * k)})" stroke="#fff"/>')
        seed = int(s.argmax())
        x, w = xs(*body["spans"][seed], n)
        layer.append(f'<text x="{x + w / 2:.1f}" y="22" font-size="15" text-anchor="middle" fill="#1f5fa8">▼ cao nhất {hi:.2f}</text>')
        note = (f"{len(s)} đoạn của bộ chấm; điểm thấp nhất {lo:.2f}, cao nhất {hi:.2f} (đậm = cao)."
                if len(s) == len(body["spans"]) else "số đoạn của hồ sơ khác số span: bỏ qua hình.")
        if len(s) != len(body["spans"]):
            layer = []
    m = "" if pd.isna(r.get("m3all")) else (f"<div>m3all<b>{r['m3all']:.3f}</b><span>0,4·dense {r['dense']:.3f} + "
                                             f"0,2·sparse {r['sparse']:.3f} + 0,4·ColBERT {r['colbert']:.3f}</span></div>")
    return f"""{grid_block(t, "rr3")}{grid_table(t)}<h3 class=fol>Tài liệu đang theo dõi: {t['doc']}</h3>
<p>Ba đặc trưng chấm lại pool, đều từ mô hình mở (BAAI):</p>
<div class=kpi><div>rr3 · bge-reranker-v2-m3<b>{r.get('rr3', float('nan')):.2f}</b><span>logit cross-encoder trên đoạn trùng từ nhất với câu hỏi</span></div>{m}</div>
<p><b>rr4</b>: cùng reranker chấm <b>mọi đoạn</b> của tài liệu → hồ sơ điểm theo đoạn, dùng để đặt cửa sổ chunk:</p>
{bar(len(body['text']), layer) if layer else f'<p class=note>Hồ sơ rr4 theo đoạn cần văn bản đầy đủ của tài liệu {t["doc"]}: có sẵn cho 5 tài liệu đầu mỗi câu hỏi.</p>'}
<p class=note>{E(note)}</p>"""


def svg_curve(t: dict) -> str:
    w, h, pad = 1000, 400, 60
    xmax = int(min(len(t["f_doc"]), max(300, 2 * t["n_deep"])))
    ymax = float(max(t["f_doc"][:xmax].max(), t["f_chunk"][:min(xmax, len(t["f_chunk"]))].max())) * 1.2

    def xy(x: float, y: float) -> tuple[float, float]:
        return pad + (x - 1) / (xmax - 1) * (w - 2 * pad), h - pad - y / ymax * (h - 2 * pad)

    def line(f: np.ndarray, color: str) -> str:
        pts = " ".join("%.1f,%.1f" % xy(i + 1, v) for i, v in enumerate(f[:xmax]))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>'

    def mark(x: int, color: str, label: str, row: int) -> str:
        x0, _ = xy(x, 0)
        return (f'<line x1="{x0:.1f}" y1="{pad}" x2="{x0:.1f}" y2="{h - pad}" stroke="{color}" stroke-dasharray="6 5" stroke-width="2"/>'
                f'<text x="{w - pad - 8}" y="{pad + 22 + 22 * row}" fill="{color}" font-size="17" text-anchor="end">'
                f'{E(label)}</text>')

    ticks = "".join(f'<text x="{xy(x, 0)[0]:.1f}" y="{h - pad + 24}" font-size="15" text-anchor="middle" fill="#5b6570">'
                    f'{x}</text>' for x in np.linspace(1, xmax, 6).astype(int))
    yt = "".join(f'<text x="{pad - 8}" y="{xy(1, v)[1] + 5:.1f}" font-size="15" text-anchor="end" fill="#5b6570">'
                 f'{v:.2f}</text>' for v in np.linspace(0, ymax / 1.2, 3))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-height:440px" role="img" aria-label="E[F2] theo số tài liệu">'
            f'<rect x="{pad}" y="{pad}" width="{w - 2 * pad}" height="{h - 2 * pad}" fill="#fbfaf6" stroke="#ddd"/>'
            + line(t["f_doc"], "#1f5fa8") + line(t["f_chunk"], "#d9822b")
            + mark(t["n_doc"], "#1f5fa8", f"N_q = {t['n_doc']} docs (#7892)", 0)
            + mark(t["n_deep"], "#6b4fa0", f"1,25·N_q = {t['n_deep']} docs (#7916)", 1)
            + mark(t["n_chunk"], "#d9822b", f"n_c = {t['n_chunk']} chunks", 2) + ticks + yt
            + f'<text x="{w / 2}" y="{h - 12}" font-size="16" text-anchor="middle" fill="#333">số tài liệu n, xếp theo p giảm dần</text>'
            + f'<text x="{pad}" y="{pad - 18}" font-size="16" fill="#333">E[F2]: <tspan fill="#1f5fa8">docs</tspan>'
              f' · <tspan fill="#d9822b">chunks</tspan></text></svg>')


def step_depth(t: dict) -> str:
    c, r = t["c"], t["row"]
    checks = "".join(f"<li class={'ok' if g == w else 'bad'}>{'✔' if g == w else '✘'} {E(n)}"
                     f"{'' if isinstance(g, bool) else f': {g}'}</li>" for n, (g, w) in t["checks"].items())
    return f"""<div class=two><div>
<p><b>p(gold)</b>: logit = a<sub>ngôn ngữ</sub> + e·log(1+hạng) + h + b·rr3 + c·m3z + d·m3rel.
14 tham số được fit sao cho P/R/F2 tổng hợp mà mô hình dự đoán khớp P/R/F2 public của 27 bài của đội. Không dùng nhãn từng cặp.</p>
<p><b>Độ sâu riêng cho câu hỏi này</b>: lấy n làm cực đại F2 kỳ vọng. Chunks dùng p<sub>c</sub> = 0,77·p và số gold chunk = 75/71 × số gold doc.</p>
<div class=kpi><div>p của tài liệu {t['doc']}<b>{r['p']:.3f}</b><span>hạng p #{t['pos']}</span></div>
<div>Kỳ vọng số gold<b>{t['nq']['g_exp']:.1f}</b></div><div>N_q<b>{t['n_doc']}</b><span>{lang_mix(c['lang'].head(t['n_doc']))}</span></div></div>
</div><div><p><b>Đối chiếu với bài đã nộp</b> (tính lại mỗi lần mở demo):</p><ul class=checks>{checks}</ul></div></div>
{svg_curve(t)}
{grid_block(t, "p")}{grid_table(t)}"""


def production_chunker():
    """stair_chunk and lexical_best exactly as the production materialize step imported them (from --prepare)."""
    import importlib.util
    f, code = OUT / "tools/chunk_stair.py", OUT / "mat_in/vibiomir-code"
    if not f.exists() or not code.exists():
        return None, None
    spec = importlib.util.spec_from_file_location("chunk_stair_prod", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from r2ai2026.stages.build_submission import lexical_best  # noqa: E402
    return mod.stair_chunk, lexical_best


def chunk_story(t: dict) -> dict | None:
    """How the submitted chunk of the followed document was cut: seed, every growth round, and a check that the
    production function gives exactly the submitted text."""
    body, ch = t["body"], t.get("chunk")
    if not body or not ch:
        return None
    title, text, spans = body["title"], body["text"], [tuple(x) for x in body["spans"]]
    prof = t["profile"]["score"].tolist() if len(t["profile"]) else None
    stair, lexical_best = production_chunker()
    lex = lexical_best(t["query"], title, text, spans) if lexical_best else None
    out = {"whole": len((title + "\n" + text).strip()) <= 4500, "rounds": [], "same": None}
    if stair:
        out["same"] = stair(title, text, spans, None, prof, 4500, 4500, 0.30, lex) == ch["text"]
    if out["whole"] or not spans:
        return out
    best = lex if lex is not None else 0
    if prof:
        m = min(len(prof), len(spans))
        best = max(range(m), key=lambda k: (prof[k], -k))
    out["seed"] = best
    i = j = best  # the loop of chunk_stair._window, recorded round by round
    grew = True
    while grew:
        grew, tried = False, []
        for ni, nj in ((i, j + 1), (i - 1, j)):  # both candidates come from the window at the start of the round
            ok = 0 <= ni and nj < len(spans) and spans[nj][1] - spans[ni][0] <= 4500
            tried.append((ni, nj, ok, spans[nj][1] - spans[ni][0] if 0 <= ni and nj < len(spans) else None))
            if ok:
                i, j, grew = ni, nj, True
        out["rounds"].append((tried, (i, j)))
    out["window"] = (i, j)
    return out


def preview_block(t: dict) -> str:
    """The submitted chunk of the followed document as stored by --grid (first characters), when no body is prepared."""
    g = t["grid"]
    r = g[g["doc_id"] == t["doc"]]
    if r.empty or pd.isna(r.iloc[0]["chunk_len"]) or int(r.index[0]) >= t["n_chunk"]:
        return f"<p class=note>Tài liệu {t['doc']} không có đoạn trích trong bài nộp (hạng p #{t['pos']} > n_c = {t['n_chunk']}).</p>"
    x = r.iloc[0]
    return (f"<p>Đoạn trích đã nộp cho tài liệu {t['doc']}: <b>{fmt(x['chunk_len'])} ký tự</b> nguyên văn "
            f"(hạng p #{t['pos']}). Phần đầu đoạn:</p><pre class=doc>{E(str(x['preview']))} …</pre>"
            f"<p class=note>Phát lại từng vòng cắt cần văn bản đầy đủ của tài liệu: chỉ có sẵn cho 5 tài liệu đầu mỗi câu hỏi "
            f"(<code>python run.py --demo --prepare</code>, cần docstore).</p>")


def step_chunk(t: dict) -> str:
    return grid_block(t, "chunk") + grid_table(t) + f"<h3 class=fol>Tài liệu đang theo dõi: {t['doc']}</h3>" + chunk_detail(t)


def chunk_detail(t: dict) -> str:
    body, ch = t["body"], t.get("chunk")
    if not body or not ch:
        return preview_block(t)
    st = t.get("story") or {}
    n = len(body["text"])
    layers = []
    for i, (a, b) in enumerate(body["spans"]):
        sx, sw = xs(a, b, n)
        layers.append(f'<rect x="{sx:.1f}" y="30" width="{sw:.1f}" height="44" fill="{"#c9dbef" if i % 2 else "#dde8f4"}"/>'
                      f'<text x="{sx + sw / 2:.1f}" y="58" font-size="15" text-anchor="middle" fill="#5b6570">{i}</text>')
    if "seed" in st:
        sx, sw = xs(*body["spans"][st["seed"]], n)
        layers.append(f'<rect x="{sx:.1f}" y="30" width="{sw:.1f}" height="44" fill="#1f5fa8" opacity=".85"/>'
                      f'<text x="{sx + sw / 2:.1f}" y="20" font-size="15" text-anchor="middle" fill="#1f5fa8">hạt giống rr4</text>')
    x, w = xs(ch["start"], ch["end"], n)
    layers.append(f'<rect x="{x:.1f}" y="26" width="{w:.1f}" height="52" fill="none" stroke="#d9822b" stroke-width="5"/>')
    if st.get("whole"):
        how = "<p>Tiêu đề + thân bài ≤ 4.500 ký tự, nên chunk là <b>cả tài liệu</b>.</p>"
    else:
        rows = ""
        for k, (tried, (i, j)) in enumerate(st.get("rounds", []), 1):
            tt = " · ".join(f"[{a}–{b}] {'—' if L is None else fmt(L) + ' ký tự'} {'✔' if ok else '✘'}" for a, b, ok, L in tried)
            rows += f"<tr><td>{k}</td><td>{E(tt)}</td><td>[{i}–{j}]</td></tr>"
        how = (f"<p>Thân bài chia thành {len(body['spans'])} span (cùng các span reranker đã chấm ở rr4). "
               f"Hạt giống là span có điểm rr4 cao nhất: <b>span {st.get('seed')}</b>. Mỗi vòng, từ cửa sổ đầu vòng, thử nối "
               "thêm một span phía sau rồi một span phía trước; phép thử nào vẫn ≤ 4.500 ký tự thì thay cửa sổ. Vì thế khi cả hai "
               "đều vừa thì phép thử thứ hai (phía trước) được giữ. Dừng khi không phép thử nào vừa.</p>"
               f"<table><tr><th>Vòng</th><th>Thử [span đầu–span cuối]</th><th>Giữ</th></tr>{rows}</table>")
    same = st.get("same")
    check = ("" if same is None else
             f"<li class={'ok' if same else 'bad'}>{'✔' if same else '✘'} hàm chunk_stair của lần chạy sản xuất cho đúng văn bản chunk đã nộp</li>")
    return f"""{bar(n, layers)}
<p class=legend><span style="background:#1f5fa8"></span>hạt giống rr4 <span style="border:3px solid #d9822b;background:none"></span>
chunk nộp: lát cắt nguyên văn [{fmt(ch['start'])}, {fmt(ch['end'])}), {fmt(len(ch['text']))} ký tự</p>
<div class=two><div>{how}</div><div>
<ul class=checks><li class=ok>✔ chunk là lát cắt nguyên văn của thân bài</li>{check}</ul></div></div>
<pre class=doc>{E(ch['text'][:1600])}{' …' if len(ch['text']) > 1600 else ''}</pre>"""


def step_output(t: dict) -> str:
    docs = t["top"][:t["n_doc"]]
    chunk = t.get("chunk")
    entry = {"id": t["qid"], "relevant_docs": docs[:6] + ["…"],
             "relevant_chunks": [{"doc_id": t["doc"], "chunk_text": (chunk["text"][:120] + " …") if chunk else "…"}, "…"]}
    js = json.dumps(entry, ensure_ascii=False, indent=2).replace('"…"', "…")
    return f"""<p>Bài nộp là <code>submission.json</code> cho 1.200 câu hỏi, nén LZMA (≈ 93 MB). Mục của câu hỏi này:</p>
<pre class=doc>{E(js)}</pre>
<div class=kpi><div>relevant_docs<b>{t['n_doc']}</b><span>#7916: {t['n_deep']}</span></div><div>relevant_chunks<b>{t['n_chunk']}</b><span>mỗi chunk ≤ 4.500 ký tự, nguyên văn</span></div></div>"""


def step_overview(t: dict) -> str:
    boxes = [("Thu thập", "1,22 triệu URL BTC → crawler + cứu URL chết"),
             ("Docstore v2.1", "1.192.673 tài liệu nguyên văn, đoạn, mục, khung trang"),
             ("Chỉ mục", "BM25 ×4 + BGE-M3 dense 3,1 triệu đoạn"),
             ("Khung truy vấn", "Qwen3.5-9B: từ khoá VI, câu hỏi/từ khoá ZH, EN"),
             ("Truy hồi + pool", "12 kênh → RRF → 2.500 ứng viên VI/ZH/EN"),
             ("Xếp hạng lại", "bge-reranker-v2-m3 + BGE-M3 ba đầu"),
             ("p(gold) + độ sâu", "N_q = argmax E[F2] cho từng câu hỏi"),
             ("Chunk", "cửa sổ nguyên văn ≤ 4.500 ký tự quanh đoạn tốt nhất"),
             ("Bài nộp", "submission.json → zip LZMA ≈ 93 MB")]
    flow = "".join(f'<a class=box href="#{k + 2}" onclick="go({k + 1});return false"><i>{k + 1}</i><b>{E(a)}</b>'
                   f'<span>{E(b)}</span></a>' for k, (a, b) in enumerate(boxes))
    return f"""<div class=q>Câu hỏi #{t['qid']}: {E(t['query'])}</div>
<p>Hệ thống trả lời bằng <b>danh sách tài liệu</b> và <b>đoạn trích</b>. Điểm = trung bình F2 của docs và F2 của chunks, nên
câu hỏi khó nhất là <b>lấy bao nhiêu tài liệu cho từng câu hỏi</b>. Demo đi qua từng tầng với câu hỏi này và theo dõi
một tài liệu thật (<b>{t['doc']}</b>) từ lúc crawl tới lúc thành chunk trong bài nộp. Từ bước 6, cả <b>{len(t['grid'])} tài liệu
đầu</b> của câu hỏi hiện thành một lưới cố định thứ tự; mỗi bước tô lưới theo đúng thông tin của tầng đó.</p>
<div class=flow>{flow}</div>
<p class=note>Bấm một khối hoặc dùng phím ← → để chuyển bước. Mọi số trên các trang lấy từ tệp của lần chạy sản xuất.</p>"""


STEPS = [("Kiến trúc", step_overview), ("Thu thập dữ liệu", step_crawl), ("Docstore: một tài liệu thật", step_docstore), ("Chỉ mục", step_index),
         ("Câu hỏi và khung truy vấn", step_query), ("Truy hồi đa ngữ và pool", step_retrieval),
         ("Xếp hạng lại", step_rerank), ("p(gold) và độ sâu theo câu hỏi", step_depth),
         ("Chunk nguyên văn", step_chunk), ("Bài nộp", step_output)]
CSS = """
:root{--ink:#1d232a;--mute:#5b6570;--bg:#f4f3ee;--card:#fffefb;--line:#e3e2dc;--acc:#1f5fa8}
*{box-sizing:border-box} body{margin:0;font:19px/1.5 'Segoe UI',Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:2;background:#14213d;color:#f4f3ee;padding:10px 24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header b{font-size:18px} nav{display:flex;gap:4px;flex-wrap:wrap;flex:1}
nav button{font:14px 'Segoe UI',Arial;border:0;border-radius:14px;padding:4px 11px;background:#2b3a5c;color:#dfe6f2;cursor:pointer}
nav button.on{background:#e8c547;color:#14213d;font-weight:600} .ctl button{font-size:15px;padding:5px 12px;cursor:pointer}
section.step{max-width:1500px;margin:0 auto;padding:22px 28px 60px;border-bottom:2px solid var(--line)}
body.js section.step{display:none;border:0} body.js section.step.on{display:block}
body.js.all section.step{display:block;border-bottom:2px solid var(--line)}
a.home{color:#e8c547;text-decoration:none;font-size:15px}
h2{font-size:34px;margin:6px 0 14px} h2 small{color:var(--mute);font-size:18px;font-weight:400;display:block}
h3{font-size:19px;margin:0 0 6px} code{font-size:.85em;background:#ebe9e2;padding:0 4px;border-radius:3px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:32px} @media(max-width:1000px){.two{grid-template-columns:1fr}}
.q{background:var(--card);border-left:5px solid var(--acc);padding:14px 18px;font-size:22px;margin-bottom:12px}
.note{color:var(--mute);font-size:16px} dl{display:grid;grid-template-columns:200px 1fr;gap:6px 14px} dt{color:var(--mute)}
table{border-collapse:collapse;width:100%;font-size:16px;background:var(--card)} td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
th{background:#ecebe5} .n{text-align:right;font-variant-numeric:tabular-nums} td.u{color:var(--mute);font-size:14px}
tr.me{background:#fff3c4} .hb{height:14px;background:var(--acc);border-radius:3px}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0} .kpi div{background:var(--card);border:1px solid var(--line);padding:10px 16px;min-width:180px}
.kpi b{display:block;font-size:30px} .kpi span{display:block;color:var(--mute);font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);padding:14px;opacity:.5} .card.on{opacity:1;border:2px solid var(--acc)}
pre.doc{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);padding:14px;font:15px/1.5 Consolas,monospace;max-height:420px;overflow:auto}
mark{background:#f3c6c0} .legend{font-size:15px;color:var(--mute)} .legend span{display:inline-block;width:22px;height:14px;margin:0 6px 0 14px;vertical-align:middle}
.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:18px 34px;margin:18px 0}
.box{position:relative;display:block;background:var(--card);border:2px solid var(--line);border-left:6px solid var(--acc);padding:14px 16px;color:var(--ink);text-decoration:none}
.box:hover{border-color:var(--acc)} .box i{font-style:normal;color:var(--acc);font-weight:700;margin-right:8px} .box b{font-size:21px}
.box span{display:block;color:var(--mute);font-size:16px} .box:not(:nth-child(3n))::after{content:'→';position:absolute;right:-28px;top:34%;font-size:24px;color:var(--acc)}
ul.checks{list-style:none;padding:0} li.ok{color:#1d6b3a} li.bad{color:#b3261e;font-weight:700} details{margin:10px 0} summary{cursor:pointer;color:var(--acc)}
.gwrap{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:4px 0 14px}
.gh{margin:0 0 8px;font-size:17px} .grid{display:grid;grid-template-columns:repeat(10,1fr);gap:5px}
.grid .c{position:relative;display:flex;flex-direction:column;justify-content:space-between;height:58px;padding:3px 7px;
border-radius:6px;text-decoration:none;font-variant-numeric:tabular-nums;border:2px solid transparent;transition:transform .08s}
.grid .c b{font-size:12px;font-weight:600;opacity:.85} .grid .c i{position:absolute;top:3px;right:7px;font:11px 'Segoe UI',Arial;opacity:.8}
.grid .c span{font-size:17px;font-weight:700;text-align:right} .grid a.c:hover{transform:scale(1.08);z-index:2;box-shadow:0 4px 14px rgba(0,0,0,.25)}
.grid .dim{opacity:.22} .grid .deep{opacity:.6;border:2px dashed #6b4fa0}
.grid .me{opacity:1;outline:4px solid #e8c547;outline-offset:1px;z-index:1;box-shadow:0 0 0 8px rgba(232,197,71,.35)}
h3.fol{margin:18px 0 8px;padding-top:12px;border-top:2px solid var(--line);font-size:22px}
.tw{max-height:520px;overflow:auto} .tw table{font-size:15px} .tw th{position:sticky;top:0}
"""
JS = """
const S=[...document.querySelectorAll('section.step')],B=[...document.querySelectorAll('nav button')];let i=0;
function go(k){i=Math.max(0,Math.min(S.length-1,k));S.forEach((s,j)=>s.classList.toggle('on',j==i));B.forEach((b,j)=>b.classList.toggle('on',j==i));history.replaceState(null,'','#'+(i+1));}
B.forEach((b,j)=>b.onclick=()=>go(j));document.getElementById('prev').onclick=()=>go(i-1);document.getElementById('next').onclick=()=>go(i+1);
document.getElementById('all').onclick=()=>document.body.classList.toggle('all');
addEventListener('keydown',e=>{if(e.key=='ArrowRight'||e.key=='PageDown')go(i+1);if(e.key=='ArrowLeft'||e.key=='PageUp')go(i-1);});
document.body.classList.add('js');go((parseInt(location.hash.slice(1))||1)-1);
"""


def render_html(t: dict, ok: bool, home: str = "") -> str:
    nav = "".join(f"<button>{k + 1}. {E(name)}</button>" for k, (name, _) in enumerate(STEPS))
    body = "".join(f'<section class=step><h2><small>Bước {k + 1}/{len(STEPS)} · câu hỏi #{t["qid"]} · tài liệu {t["doc"]}</small>'
                   f"{E(name)}</h2>{fn(t)}</section>" for k, (name, fn) in enumerate(STEPS))
    back = f'<a class=home href="{home}">☰ 1.200 câu hỏi</a>' if home else ""
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ViBioMIR · câu hỏi #{t['qid']}</title><style>{CSS}</style></head><body>
<header><b>ViBioMIR · Băng Cá Khô</b>{back}<nav>{nav}</nav><span class=ctl><button id=prev>◀</button> <button id=next>▶</button>
<button id=all>tất cả</button></span></header>{body}
<p class=note style="max-width:1500px;margin:0 auto;padding:0 28px 24px">Phát lại lần chạy sản xuất từ các tệp đã chia sẻ; không phải câu trả lời mới,
không có nhãn gold; p(gold) là ước lượng của mô hình. Đối chiếu với bài đã nộp: <b>{'khớp' if ok else 'KHÔNG KHỚP'}</b>.</p>
<script>{JS}</script></body></html>"""


def write_html(t: dict, path: Path, ok: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(t, ok), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ prepare

def prepare(qids: list[int], per_query: int) -> None:
    """Document text (top documents of each question) and chunk text from the production materialize code."""
    need([ROOT / "kaggle/datasets/pas_scores/pas_scores.parquet", ROOT / "kaggle/chunk_v21_assets/vibiomir-code.zip",
          DS / "corpus"], "unzip artifacts_B_materialize.zip and put docstore_v2.1/ at data/docstore/v2.1/")
    sys.path.insert(0, str(ROOT))
    from reproduce import link_dir  # noqa: E402
    stage, tools = OUT / "mat_in", OUT / "tools"
    corpus = stage / "corpus"
    if corpus.is_symlink() or getattr(corpus, "is_junction", lambda: False)():
        os.rmdir(corpus) if os.name == "nt" else corpus.unlink()
    for d in (stage, tools, OUT / "zips"):
        if d.exists():
            shutil.rmtree(d)
    stage.mkdir(parents=True)
    lists = pd.read_parquet(LISTS["pN3_base"], filters=[("qid", "in", qids)])
    lists = lists[lists["job"] == "pN3_base"]
    # only the chunks the walkthrough can show (top documents per question): each chunk depends on its own
    # (question, document) pair only, so a shorter list gives the same text for these pairs
    ch = lists[lists["kind"] == "chunk"].sort_values(["qid", "pos"]).groupby("qid").head(per_query)
    lists = pd.concat([lists[lists["kind"] == "doc"], ch], ignore_index=True)
    lists.to_parquet(stage / "lists.parquet", index=False)
    pd.read_parquet(ROOT / "data/raw/btc/query.parquet", filters=[("id", "in", qids)]).to_parquet(stage / "query.parquet", index=False)
    pd.read_parquet(ROOT / "kaggle/datasets/pas_scores/pas_scores.parquet", filters=[("qid", "in", qids)]) \
        .to_parquet(stage / "pas_scores.parquet", index=False)
    with zipfile.ZipFile(ROOT / "kaggle/chunk_v21_assets/vibiomir-code.zip") as z:
        z.extractall(stage / "vibiomir-code")
    link_dir((DS / "corpus").resolve(), corpus)
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts/release/extract_bundle.py"),
                        str(ROOT / "kaggle/kernels_generic/vibiomir-mat-st3-acc2"), "--out", str(tools)], check=True)
        subprocess.run([sys.executable, str(tools / "materialize_lists.py"), "--input", str(stage),
                        "--out", str(OUT / "zips")], check=True)
    finally:
        os.rmdir(corpus) if os.name == "nt" else corpus.unlink()
    with zipfile.ZipFile(OUT / "zips/pN3_base.zip") as z:
        items = json.loads(z.read("submission.json").decode("utf-8"))
    chunks = pd.DataFrame([(int(x["id"]), i, int(ch["doc_id"]), ch["chunk_text"]) for x in items
                           for i, ch in enumerate(x["relevant_chunks"])], columns=["qid", "pos", "doc_id", "chunk_text"])
    # document text of the top documents of each question, with the passage spans the reranker scored (rr_spans of the
    # same pinned code the materialize step imports)
    sys.path.insert(0, str(stage / "vibiomir-code"))
    from r2ai2026.stages.build_submission import rr_spans  # noqa: E402
    ch = lists[lists["kind"] == "chunk"].sort_values(["qid", "pos"])
    want = {(int(q), int(d)) for q, d in ch.groupby("qid").head(per_query)[["qid", "doc_id"]].itertuples(index=False)}
    ids = {d for _, d in want}
    bodies: dict[int, tuple[str, str]] = {}
    for part in sorted((DS / "corpus").glob("part-*.jsonl*")):
        opener = gzip.open if part.suffix == ".gz" else open
        with opener(part, "rt", encoding="utf-8") as f:
            for line in f:
                if len(bodies) == len(ids):
                    break
                d = json.loads(line)
                if d["id"] in ids and d["id"] not in bodies:
                    bodies[d["id"]] = (d.get("t") or "", d.get("x") or "")
    docs = pd.DataFrame([(q, d, *bodies[d], json.dumps(rr_spans(bodies[d][1]))) for q, d in sorted(want) if d in bodies],
                        columns=["qid", "doc_id", "title", "text", "spans"])
    for path, new in ((CHUNKS, chunks), (DOCS, docs)):
        if path.exists():
            old = pd.read_parquet(path)
            new = pd.concat([old[~old["qid"].isin(qids)], new], ignore_index=True)
        new.to_parquet(path, index=False)
    print(f"{len(qids)} questions: {len(chunks)} chunks, {len(docs)} document texts -> {OUT.relative_to(ROOT)}")


def iter_submission(path: Path, digest: "hashlib._Hash"):
    """Entries of submission.json inside the zip, parsed one by one (the file is ~0.9 GB: never loaded whole)."""
    dec, text = json.JSONDecoder(), codecs.getincrementaldecoder("utf-8")()
    with zipfile.ZipFile(path) as z, z.open("submission.json") as raw:
        def more() -> tuple[str, bool]:
            b = raw.read(1 << 22)
            digest.update(b)
            return text.decode(b, final=not b), not b
        buf, eof = more()
        buf = buf.lstrip()
        if not buf.startswith("["):
            raise SystemExit(f"{path}: submission.json is not a JSON list")
        buf = buf[1:]
        while True:
            buf = buf.lstrip(" \t\r\n,")
            if buf.startswith("]"):
                return
            try:
                obj, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk, eof = more()
                buf += chunk
                continue
            yield obj
            buf = buf[end:]


def build_grid(snapshot_dir: Path | None) -> None:
    """out/demo/top100.parquet: for the GRID highest-p documents of every question, the length and the first characters of
    the chunk submitted for it (from SUBMISSION). With --snapshot-from: the crawl/docstore/index steps of pages rendered
    with the docstore present, reused when the docstore is not on this machine."""
    need([PG / "cands_p.parquet", SUBMISSION], "unzip artifacts_A_pgold.zip; submitted zips in submissions/")
    c = pd.read_parquet(PG / "cands_p.parquet", columns=["qid", "doc_id", "p"])
    top = c.sort_values(["qid", "p"], ascending=[True, False]).groupby("qid").head(GRID)
    want = {(int(q), int(d)) for q, d in top[["qid", "doc_id"]].itertuples(index=False)}
    sys.path.insert(0, str(ROOT))
    from reproduce import SUBMITTED_JSON_SHA256  # noqa: E402
    h, rows = hashlib.sha256(), []
    for k, e in enumerate(iter_submission(SUBMISSION, h), 1):
        q = int(e["id"])
        rows += [(q, int(x["doc_id"]), len(x["chunk_text"]), x["chunk_text"][:600])
                 for x in e["relevant_chunks"] if (q, int(x["doc_id"])) in want]
        if k % 200 == 0:
            print(f"  {k} questions", flush=True)
    ok = h.hexdigest() == SUBMITTED_JSON_SHA256["N1_n125"]
    print(f"{SUBMISSION.name}: submission.json sha256 {'matches #7916' if ok else 'DIFFERS from #7916'}")
    if not ok:
        raise SystemExit("not the submitted file")
    out = pd.DataFrame(rows, columns=["qid", "doc_id", "chunk_len", "preview"])
    out.to_parquet(TOP100, index=False)
    print(f"{len(out)} chunks of top-{GRID} documents -> {TOP100.relative_to(ROOT)}")
    if snapshot_dir:
        snap = {}
        for page in sorted(snapshot_dir.glob("q*.html")):
            secs = page.read_text(encoding="utf-8").split("<section class=step>")[1:]
            m = re.search(r"câu hỏi #(\d+) · tài liệu (\d+)", secs[0]) if secs else None
            if not m or len(secs) != len(STEPS):
                continue
            snap[f"{m.group(1)}:{m.group(2)}"] = {str(k): secs[k].split("</h2>", 1)[1].rsplit("</section>", 1)[0]
                                                   for k in (1, 2, 3)}
        SNAP.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        print(f"{len(snap)} saved pages ({', '.join(snap)}) -> {SNAP.relative_to(ROOT)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", type=int, default=401, help="question id, 1..1200")
    ap.add_argument("--doc", type=int, default=None, help="document to follow (default: highest p with text)")
    ap.add_argument("--prepare", action="store_true", help="fetch document text and chunk text for --qids")
    ap.add_argument("--qids", default="", help="comma list for --prepare (default: --qid)")
    ap.add_argument("--docs-per-query", type=int, default=5, help="--prepare: text of the top-p documents per question")
    ap.add_argument("--html", type=Path, default=None, help="page path (default: out/demo/q<qid>.html)")
    ap.add_argument("--grid", action="store_true", help=f"build {TOP100.name} (chunk of the top-{GRID} documents) once")
    ap.add_argument("--snapshot-from", type=Path, default=None,
                    help="--grid: folder of pages rendered with the docstore (their steps 2-4 are reused without it)")
    args = ap.parse_args()
    if args.grid:
        build_grid(args.snapshot_from)
        return
    if args.prepare:
        qids = list(range(1, 1201)) if args.qids == "all" else [int(x) for x in args.qids.split(",") if x] or [args.qid]
        prepare(qids, args.docs_per_query)
        return
    t = trace(args.qid, args.doc)
    ok = print_trace(t)
    page = args.html or OUT / f"q{args.qid}.html"
    write_html(t, page, ok)
    print(f"-> {page}")
    if not ok:
        raise SystemExit("replay does not match the submitted lists")


if __name__ == "__main__":
    main()
