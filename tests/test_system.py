"""Runner behaviour: templating, builtin candidate fusion, caching by config hash, missing outputs."""
from __future__ import annotations

import sys

import pandas as pd
import pytest

from r2ai2026.system import runner as system


def make_cfg(tmp_path, script_args):
    store = tmp_path / "store"
    store.mkdir()
    pd.DataFrame({"id": [1, 2], "query": ["a", "b"]}).to_parquet(tmp_path / "q.parquet")
    script = tmp_path / "touch.py"
    script.write_text("import sys, pathlib\np = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)\n"
                      "p.write_text(sys.argv[2] if len(sys.argv) > 2 else 'x')\n")
    return {"name": "t", "data": str(store), "queries": str(tmp_path / "q.parquet"), "work": str(tmp_path / "w"),
            "prepare": False, "vars": {},
            "stages": {"a": {"script": str(script), "args": script_args, "outputs": ["out.txt"]}}}


def test_fill_nested():
    ctx = {"out": "/o", "bm25": "/w/bm25"}
    assert system.fill({"x": ["{out}/r", {"y": "{bm25}/ranking.parquet"}], "n": 3}, ctx) == \
        {"x": ["/o/r", {"y": "/w/bm25/ranking.parquet"}], "n": 3}


def test_stage_runs_then_caches(tmp_path, capsys):
    cfg = make_cfg(tmp_path, ["{out}/out.txt", "v1"])
    system.run(cfg)
    assert (tmp_path / "w" / "a" / "out.txt").read_text() == "v1"
    system.run(cfg)
    assert "a: cached" in capsys.readouterr().out
    cfg["stages"]["a"]["args"] = ["{out}/out.txt", "v2"]  # config change invalidates the cache
    system.run(cfg)
    assert (tmp_path / "w" / "a" / "out.txt").read_text() == "v2"


def test_missing_output_fails(tmp_path):
    cfg = make_cfg(tmp_path, ["{out}/other.txt"])
    with pytest.raises(SystemExit, match="without its outputs"):
        system.run(cfg)


def test_builtin_cands_rrf(tmp_path):
    a = pd.DataFrame({"qid": [1, 1, 1], "doc_id": [10, 11, 900000], "score": [3.0, 2.0, 1.0]})
    b = pd.DataFrame({"qid": [1, 1], "doc_id": [11, 12], "score": [0.9, 0.8]})
    a.to_parquet(tmp_path / "a.parquet")
    b.to_parquet(tmp_path / "b.parquet")
    system.builtin_cands({"systems": [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")],
                          "weights": [1, 2], "vi_only": True, "k": 2}, tmp_path)
    c = pd.read_parquet(tmp_path / "cands.parquet")
    assert c["doc_id"].tolist() == [11, 12]  # 11: 1/62 + 2/61 beats 12: 2/62 and 10: 1/61; 900000 is not VI


def test_unknown_stage(tmp_path):
    cfg = make_cfg(tmp_path, ["{out}/out.txt"])
    with pytest.raises(SystemExit, match="unknown stages"):
        system.run(cfg, stages=["nope"])


def test_main_entry_dry_run(tmp_path, capsys, monkeypatch):
    import yaml

    cfg = make_cfg(tmp_path, ["{out}/out.txt"])
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["x"])
    assert system.main(["--config", str(path), "--dry-run"]) == 0
    assert "a: would run" in capsys.readouterr().out


def test_builtin_cands_quota_per_language(tmp_path):
    # ids: 10, 11 VI; 800000, 800001 ZH; 5000000 EN
    a = pd.DataFrame({"qid": [1] * 5, "doc_id": [800000, 10, 5000000, 800001, 11], "score": [5.0, 4, 3, 2, 1]})
    a.to_parquet(tmp_path / "a.parquet")
    system.builtin_cands({"systems": [str(tmp_path / "a.parquet")], "quota": {"vi": 2, "zh": 1}}, tmp_path)
    c = pd.read_parquet(tmp_path / "cands.parquet")
    assert c["doc_id"].tolist() == [800000, 10, 11]  # EN not asked for, second ZH over its quota
    assert c["lang"].tolist() == ["zh", "vi", "vi"] and c["rank"].tolist() == [0, 1, 2]
