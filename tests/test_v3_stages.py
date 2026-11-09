"""The v3 wiring (configs/system_v3.yaml): the list decision, the pool builders and the materialize stage.

Everything runs on small fixtures under tmp_path; no production file is read or written.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STAGES = ROOT / "r2ai2026" / "stages"
ANALYSIS = ROOT / "scripts" / "analysis"
KAGGLE_OPS = ROOT / "scripts" / "kaggle_ops"


def load(name: str, path: Path):
    """Import a script from its file under a private name (the stages do the same)."""
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decide_lists = load("decide_lists", ANALYSIS / "decide_lists.py")
lang_pool_sets = load("lang_pool_sets", KAGGLE_OPS / "lang_pool_sets.py")

VI_MAX, ZH_MAX = 766441, 4392214  # official id blocks


# ------------------------------------------------------------- decide_lists.order

def small_pool() -> pd.DataFrame:
    """Two queries x eight docs: four VI (one at language rank 500, for `vitail`), two ZH, two EN.

    ``score`` and ``rr_rank`` both descend with the position of the doc in this frame, so `rr` returns
    the frame's own order; ``m3all`` ranks by position *within* its language, so `m3rr` differs.
    """
    vi_ids, zh_ids, en_ids = [1, 2, 3, 4], [900_000, 900_001], [5_000_000, 5_000_001]
    rows = []
    for qid in (10, 20):
        for lang, ids in (("vi", vi_ids), ("zh", zh_ids), ("en", en_ids)):
            for i, doc_id in enumerate(ids):
                rows.append({
                    "qid": qid, "doc_id": doc_id, "lang": lang,
                    "rank": 500 if lang == "vi" and i == 3 else i,  # only this one passes vitail's rank >= 300
                    "score": 100.0 - len([r for r in rows if r["qid"] == qid]),
                    "rr_rank": len([r for r in rows if r["qid"] == qid]),
                    "m3all": float(i),   # opposite direction to the frame order, ties broken by frame order
                })
    return pd.DataFrame(rows)


def order_of(pool: pd.DataFrame, method: str, qid: int = 10) -> list[int]:
    """Doc ids of one query after ordering, in list order."""
    out = decide_lists.order(pool.copy(), method)
    return [int(d) for d in out[out["qid"] == qid]["doc_id"]]


def test_order_rr_follows_the_reranker_score():
    assert order_of(small_pool(), "rr") == [1, 2, 3, 4, 900_000, 900_001, 5_000_000, 5_000_001]


def test_order_rr_plus_drops_the_head():
    # rr_rank >= 3 keeps VI rank 3 and every ZH / EN doc, still ordered by score
    assert order_of(small_pool(), "rr+3") == [4, 900_000, 900_001, 5_000_000, 5_000_001]


def test_order_by_language():
    assert order_of(small_pool(), "vi") == [1, 2, 3, 4]
    assert order_of(small_pool(), "zh") == [900_000, 900_001]


def test_order_vitail_takes_only_the_tail():
    assert order_of(small_pool(), "vitail") == [4]  # VI at language rank 500


def test_order_zhen_alternates_languages():
    assert order_of(small_pool(), "zhen") == [900_000, 5_000_000, 900_001, 5_000_001]


def test_order_m3rr_is_rrf_of_the_two_ranks():
    pool = small_pool()
    got = order_of(pool, "m3rr")
    frame = pool[pool["qid"] == 10].copy()
    frame["m3r"] = frame.groupby("qid")["m3all"].rank(ascending=False, method="first") - 1
    expect = (frame.assign(key=-(1 / (60 + frame["m3r"]) + 1 / (60 + frame["rr_rank"])))
              .sort_values("key")["doc_id"].tolist())
    assert got == [int(d) for d in expect]
    assert got[0] == 4          # m3all prefers the last VI doc, the rr order starts with 1
    assert got != order_of(pool, "rr")


def test_order_language_quota_interleaves_in_proportion():
    # rr@vi2+zh1: two VI and one ZH, spread out by their share of the quota -> VI, ZH, VI
    assert order_of(small_pool(), "rr@vi2+zh1") == [1, 900_000, 2]


# ------------------------------------------------------------- decide_lists.load_pool overrides

def tiny_feature_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    pool = pd.DataFrame([{"qid": 10, "doc_id": 1, "set": "vi800", "rank": 0},
                         {"qid": 10, "doc_id": 900_000, "set": "zh200", "rank": 0},
                         {"qid": 10, "doc_id": 5_000_000, "set": "en200", "rank": 0}])
    rr = pd.DataFrame([{"qid": 10, "doc_id": 1, "score": 1.5},
                       {"qid": 10, "doc_id": 900_000, "score": -2.0}])
    m3 = pd.DataFrame([{"qid": 10, "doc_id": 1, "dense": 0.3, "sparse": 0.2, "colbert": 0.4}])
    for name, df in [("cands.parquet", pool), ("rerank.parquet", rr), ("m3multi.parquet", m3)]:
        df.to_parquet(tmp_path / name, index=False)
    return tmp_path / "cands.parquet", tmp_path / "rerank.parquet", tmp_path / "m3multi.parquet"


def test_load_pool_reads_the_files_it_is_given(tmp_path: Path):
    cands, rr, m3 = tiny_feature_files(tmp_path)
    absent = tmp_path / "no_such_file.parquet"
    pool = decide_lists.load_pool({"vi": 800, "zh": 200, "en": 200}, pool_file=cands, rr=str(rr),
                                  m3m=str(m3), judge=str(absent), dq=str(absent), e1=absent, pas=absent)
    assert len(pool) == 3
    assert set(pool["lang"]) == {"vi", "zh", "en"}
    assert pool["score"].notna().all() and pool["rr_rank"].notna().all()
    scored = pool[pool["doc_id"] == 1]
    assert scored["m3all"].iloc[0] == pytest.approx(0.4 * 0.3 + 0.2 * 0.2 + 0.4 * 0.4)
    assert pool[pool["doc_id"] == 900_000]["m3all"].isna().all()
    # a channel whose file is missing is dropped, never an error
    for col in ("judge", "dq_score", "e1_rank", "pmax"):
        assert col not in pool


def test_load_pool_overrides_all_default_to_none():
    """No override at all means the production locations are used (paths only, nothing is read here)."""
    import inspect
    params = inspect.signature(decide_lists.load_pool).parameters
    for name in ("pool_file", "rr", "m3m", "judge", "dq", "e1", "pas"):
        assert params[name].default is None, name


# ------------------------------------------------------------- lang_pool_sets

def write_ranking(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_lang_pool_sets_overrides_build_the_requested_sets(tmp_path: Path):
    vi = write_ranking(tmp_path / "vi_pool.parquet", [
        {"qid": 10, "doc_id": 1, "rank": 0}, {"qid": 10, "doc_id": 2, "rank": 1}])
    zh = write_ranking(tmp_path / "zh.parquet", [   # one file, two variants
        {"qid": 10, "doc_id": 900_000, "variant": "zh_q", "score": 9.0},
        {"qid": 10, "doc_id": 900_001, "variant": "zh_q", "score": 4.0},
        {"qid": 10, "doc_id": 900_000, "variant": "zh_kw", "score": 8.0},
        {"qid": 10, "doc_id": 900_001, "variant": "zh_kw", "score": 5.0}])
    en = write_ranking(tmp_path / "en.parquet", [{"qid": 10, "doc_id": 5_000_000, "score": 3.0}])
    additions = tmp_path / "additions.parquet"
    assert not additions.exists()  # the stage creates it on the first run

    res = subprocess.run([sys.executable, str(KAGGLE_OPS / "lang_pool_sets.py"),
                          "--vi", "2", "--zh", "1", "--en", "1",
                          "--vi-pool", str(vi),
                          "--zh-list", f"zh_q={zh}", "--zh-list", f"zh_kw={zh}",
                          "--en-list", str(en),
                          "--additions", str(additions)],
                         capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8")
    assert res.returncode == 0, res.stderr
    out = pd.read_parquet(additions)
    assert set(out["set"]) == {"vi2", "zh1", "en1"}
    assert len(out) == 4  # 2 VI + 1 ZH + 1 EN
    # 900000 tops both ZH variants, so it beats 900001 on the fused list (1/60 + 1/60 > 1/61 + 1/61)
    assert set(out[out["set"] == "zh1"]["doc_id"]) == {900_000}
    assert set(out[out["set"] == "en1"]["doc_id"]) == {5_000_000}


def test_parse_list_spec_keeps_a_windows_drive_letter_out_of_the_variant():
    assert lang_pool_sets.parse_list_spec("zh_q=D:/a/b/ranking.parquet") == (Path("D:/a/b/ranking.parquet"), "zh_q")
    assert lang_pool_sets.parse_list_spec("D:/a/b/ranking.parquet") == (Path("D:/a/b/ranking.parquet"), None)
    assert lang_pool_sets.parse_list_spec("kaggle/outputs/x/ranking.parquet") == (
        Path("kaggle/outputs/x/ranking.parquet"), None)


def test_lang_pool_sets_keeping_production_defaults():
    """Without --zh-list / --en-list the production channel sets are still what gets fused."""
    import inspect
    defaults = lang_pool_sets.DEFAULT_LISTS
    assert len(defaults["zh"]) == 5 and len(defaults["en"]) == 4
    assert inspect.getsource(lang_pool_sets.fused).count("for path, variant in lists") == 1


# ------------------------------------------------------------- rerank_pool_cands

def test_rerank_pool_cands_deduplicates_and_copies_the_query(tmp_path: Path):
    pd.DataFrame([{"qid": 10, "set": "vi1500", "rank": 0, "doc_id": 1},
                  {"qid": 10, "set": "zh500", "rank": 0, "doc_id": 1},   # same pair, later set loses
                  {"qid": 10, "set": "en500", "rank": 0, "doc_id": 5_000_000}]).to_parquet(
        tmp_path / "additions.parquet", index=False)
    pd.DataFrame({"id": [10], "query": ["Sốt xuất huyết là gì?"]}).to_parquet(
        tmp_path / "query.parquet", index=False)
    out = tmp_path / "out"

    res = subprocess.run([sys.executable, str(KAGGLE_OPS / "rerank_pool_cands.py"),
                          "--sets", "vi1500,zh500,en500",
                          "--additions", str(tmp_path / "additions.parquet"),
                          "--query", str(tmp_path / "query.parquet"),
                          "--out", str(out)],
                         capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8")
    assert res.returncode == 0, res.stderr
    cands = pd.read_parquet(out / "cands.parquet")
    assert len(cands) == 2
    assert cands[cands["doc_id"] == 1]["set"].iloc[0] == "vi1500"
    assert pd.read_parquet(out / "query.parquet")["id"].tolist() == [10]


# ------------------------------------------------------------- materialize stage

def run_materialize(root: Path, lists: pd.DataFrame, docs: dict[int, tuple[str, str]],
                    hf: pd.DataFrame | None = None, pas: pd.DataFrame | None = None,
                    sections: pd.DataFrame | None = None, window: int = 4500) -> subprocess.CompletedProcess:
    corpus = root / "corpus"
    (corpus / "parts").mkdir(parents=True)
    with (corpus / "parts" / "part-00.jsonl").open("w", encoding="utf-8") as f:
        for doc_id, (title, body) in docs.items():
            f.write(json.dumps({"id": doc_id, "t": title, "x": body}, ensure_ascii=False) + "\n")
    pd.DataFrame({"id": [1, 2], "query": ["Sốt xuất huyết?", "Đau dạ dày nên ăn gì?"]}).to_parquet(
        root / "query.parquet", index=False)
    (root / "lists_in").mkdir(exist_ok=True)
    lists.to_parquet(root / "lists_in" / "lists.parquet", index=False)
    cmd = [sys.executable, str(STAGES / "materialize_lists.py"),
           "--lists", str(root / "lists_in"), "--queries", str(root / "query.parquet"),
           "--corpus", str(corpus), "--out", str(root / "out"), "--window", str(window)]
    for frame, name in ((hf, "hf_text.parquet"), (pas, "pas_scores.parquet"),
                        (sections, "sections.parquet")):
        if frame is not None:
            frame.to_parquet(root / name, index=False)
            cmd += ["--extra", str(root / name)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8")


def test_materialize_stage_builds_a_valid_zip(tmp_path: Path):
    from r2ai2026.core.submission import read_submission_zip, validate_submission

    docs = {101: ("Sốt xuất huyết", "Sốt xuất huyết là bệnh truyền nhiễm do virus Dengue gây ra. " * 4),
            303: ("Tiểu đường", "Tiểu đường type 2 cần theo dõi đường huyết. " * 4)}
    lists = pd.DataFrame([("t", 1, "doc", 0, 101), ("t", 1, "chunk", 0, 101),
                          ("t", 2, "doc", 0, 303), ("t", 2, "chunk", 0, 303)],
                         columns=["job", "qid", "kind", "pos", "doc_id"]).assign(size=0, src="")
    res = run_materialize(tmp_path, lists, docs)
    assert res.returncode == 0, res.stderr

    out = tmp_path / "out"
    assert (out / "stats.json").exists()
    assert {"lists.parquet", "query.parquet", "store"} <= {p.name for p in (out / "input").iterdir()}
    assert (out / "input" / "store" / "parts" / "part-00.jsonl").exists()  # linked, not copied

    recs = read_submission_zip(out / "t.zip")
    assert validate_submission(recs, expected_ids=[1, 2]) == []
    assert recs[0]["relevant_docs"] == [101]
    # a document shorter than the window goes in whole: title, then the body verbatim
    # window_chunk strips the assembled text, so the expected value is the same one stripped
    assert recs[0]["relevant_chunks"][0]["chunk_text"] == f"{docs[101][0]}\n{docs[101][1]}".strip()


def test_materialize_stage_honours_size_and_hf_text(tmp_path: Path):
    from r2ai2026.core.submission import read_submission_zip

    docs = {101: ("Sốt xuất huyết", "Sốt xuất huyết là bệnh truyền nhiễm do virus Dengue gây ra. " * 40)}
    lists = pd.DataFrame([("t", 1, "doc", 0, 101), ("t", 1, "chunk", 0, 101)],
                         columns=["job", "qid", "kind", "pos", "doc_id"]).assign(size=600, src="")
    lists.loc[lists["kind"] == "chunk", "src"] = "hf"
    hf = pd.DataFrame([{"doc_id": 101, "title": "Sốt xuất huyết (HF)",
                        "content": "Nội dung lấy từ kho Hugging Face. "}])
    res = run_materialize(tmp_path, lists, docs, hf=hf)
    assert res.returncode == 0, res.stderr

    recs = read_submission_zip(tmp_path / "out" / "t.zip")
    chunk = next(c for c in recs[0]["relevant_chunks"] if c["doc_id"] == 101)
    assert chunk["chunk_text"] == "Sốt xuất huyết (HF)\nNội dung lấy từ kho Hugging Face."
    assert len(chunk["chunk_text"]) <= 600   # the lists' size column wins over --window
    assert recs[0]["relevant_docs"] == [101]  # the doc list is unaffected by the chunk text


def test_materialize_stage_reports_zip_size(tmp_path: Path):
    docs = {101: ("Tài liệu", "Nội dung đủ dài để tạo zip. " * 20)}
    lists = pd.DataFrame([("t", 1, "doc", 0, 101), ("t", 1, "chunk", 0, 101)],
                         columns=["job", "qid", "kind", "pos", "doc_id"]).assign(size=4500, src="")
    res = run_materialize(tmp_path, lists, docs)
    assert res.returncode == 0, res.stderr
    stats = json.loads((tmp_path / "out" / "stats.json").read_text(encoding="utf-8"))
    assert stats["t"]["over_limit"] is False   # --max-mb only flags the job, it never fails the stage
    assert stats["t"]["zip_mb"] >= 0


# ------------------------------------------------------------- chunk modes: src = seed / stair

def long_doc() -> tuple[str, str]:
    """A document longer than the 4500-char window, so the seed actually decides where the chunk starts."""
    body = " ".join(f"Cau so {i} ve benh so tuiet xuat huyet va cach dieu tri an toan." for i in range(110))
    return "Sot tuiet xuat huyet", body


def test_materialize_stage_seeds_the_window_from_the_reranker(tmp_path: Path):
    from r2ai2026.core.submission import read_submission_zip
    from r2ai2026.stages.build_submission import lexical_best, rr_spans, window_chunk
    from r2ai2026.stages.chunk_stair import stair_chunk

    title, body = long_doc()
    docs = {101: (title, body)}
    lists = pd.DataFrame([("t", 1, "doc", 0, 101), ("t", 1, "chunk", 0, 101)],
                         columns=["job", "qid", "kind", "pos", "doc_id"]).assign(size=4500, src="")
    lists.loc[lists["kind"] == "chunk", "src"] = "seed"
    spans = rr_spans(body)
    pscore = [0.1] * len(spans)
    pscore[-1] = 9.9                       # the reranker's best passage is the last one, not the lexical best
    pas = pd.DataFrame([{"qid": 1, "doc_id": 101, "passage": k, "score": s}
                        for k, s in enumerate(pscore)])

    res = run_materialize(tmp_path, lists, docs, pas=pas)
    assert res.returncode == 0, res.stderr
    got = read_submission_zip(tmp_path / "out" / "t.zip")[0]["relevant_chunks"][0]["chunk_text"]
    query = "Sot tuiet xuat huyet"
    expected = stair_chunk(title, body, spans, None, pscore, 4500, 4500, 0.30,
                           lexical_best(query, title, body, spans))
    assert got == expected
    # and the seed really did move it, otherwise the test proves nothing
    assert expected != window_chunk(title, body, lexical_best(query, title, body, spans), spans, 4500, 4500)
    assert len(got) <= 4500


def test_materialize_stage_cuts_on_a_section_boundary(tmp_path: Path):
    from r2ai2026.core.submission import read_submission_zip
    from r2ai2026.stages.build_submission import rr_spans

    title, body = long_doc()
    docs = {101: (title, body)}
    cut = len(body) // 2
    sections = pd.DataFrame([{"doc_id": 101, "sid": 0, "start": 0, "end": cut},
                             {"doc_id": 101, "sid": 1, "start": cut, "end": len(body)}])
    lists = pd.DataFrame([("t", 1, "doc", 0, 101), ("t", 1, "chunk", 0, 101)],
                         columns=["job", "qid", "kind", "pos", "doc_id"]).assign(size=4500, src="")
    lists.loc[lists["kind"] == "chunk", "src"] = "stair"
    spans = rr_spans(body)
    pas = pd.DataFrame([{"qid": 1, "doc_id": 101, "passage": k, "score": 0.0} for k in range(len(spans))])

    res = run_materialize(tmp_path, lists, docs, sections=sections, pas=pas)
    assert res.returncode == 0, res.stderr
    got = read_submission_zip(tmp_path / "out" / "t.zip")[0]["relevant_chunks"][0]["chunk_text"]
    # every score ties, so the seed is passage 0 -> the section holding it, verbatim, with the title
    assert got == f"{title}\n{body[:cut]}".strip()
    assert len(got) <= 4500


# ------------------------------------------------------------- pas_agg (the p* / m3p* channel)

def test_pas_agg_writes_the_passage_aggregate(tmp_path: Path):
    cands, rr, m3 = tiny_feature_files(tmp_path)
    passages = pd.DataFrame([{"qid": 10, "doc_id": 1, "passage": 0, "score": 0.5},
                             {"qid": 10, "doc_id": 1, "passage": 1, "score": 1.5}])
    passages.to_parquet(tmp_path / "passages.parquet", index=False)
    out = tmp_path / "pool_pas.parquet"

    res = subprocess.run([sys.executable, str(ANALYSIS / "pas_agg.py"),
                          "--pas", str(tmp_path / "passages.parquet"), "--out", str(out),
                          "--pool", "vi800,zh200,en200", "--pool-file", str(cands),
                          "--rr", str(rr), "--m3m", str(m3)],
                         capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8")
    assert res.returncode == 0, res.stderr
    agg = pd.read_parquet(out)
    assert set(agg.columns) == {"qid", "doc_id", "pmax", "ptop2", "pmix"}
    hit = agg[agg["doc_id"] == 1].iloc[0]
    assert hit["pmax"] == pytest.approx(1.5)          # best passage of the pair
    assert hit["ptop2"] == pytest.approx(1.0)         # mean of both
    assert hit["pmix"] == pytest.approx(1.5 + 0.25 * 1.0)


# ------------------------------------------------------------- the config itself

V3_STAGES = ["frame", "variants", "bm25_vi_doc", "bm25_vi_sec", "bm25_zh", "bm25_en",
             "dense_vi", "dense_zh", "dense_en", "pool_vi", "pool_sets", "pool_cands",
             "rr3", "rr4", "m3m", "pas", "decide", "materialize"]


def test_system_v3_config_dry_run_resolves_every_stage(tmp_path: Path):
    """The runner resolves every template and knows every stage: the wiring itself stays valid."""
    res = subprocess.run([sys.executable, "-m", "r2ai2026", "run",
                          "--config", "configs/system_v3.yaml", "--dry-run",
                          "--var", f"work={tmp_path / 'v3'}"],
                         capture_output=True, text=True, cwd=str(ROOT), encoding="utf-8")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "would run" in res.stdout
    for name in V3_STAGES:
        assert f"{name}: would run" in res.stdout, f"missing stage {name}"
    # no unresolved {template} may survive into a command
    for line in res.stdout.splitlines():
        if "would run" in line and "builtin" not in line:
            assert "{" not in line.split(" would run ", 1)[1], line