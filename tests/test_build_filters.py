import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_submission", ROOT / "kaggle" / "build_exp" / "build_submission.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def ranking(path: Path, rows: list[tuple[int, int, float]], **extra) -> str:
    df = pd.DataFrame(rows, columns=["qid", "doc_id", "score"])
    for k, v in extra.items():
        df[k] = v
    df.to_parquet(path, index=False)
    return str(path)


def test_keep_and_exclude_make_a_stratum(tmp_path):
    union = ranking(tmp_path / "u.parquet", [(1, d, 10.0 - d) for d in range(1, 9)],
                    n_families=[3, 1, 2, 2, 1, 2, 1, 2])
    main = ranking(tmp_path / "m.parquet", [(1, 1, 5.0), (1, 3, 4.0), (1, 2, 3.0)])
    job = {"systems": [union], "n": [100],
           "keep": {"spec": union, "query": "n_families >= 2"},
           "exclude": {"systems": [main], "n": 2}}
    # union order 1..8; keep n_families >= 2 -> 1,3,4,6,8; drop main top-2 = {1, 3}
    assert bs.doc_lists(job) == {1: [4, 6, 8]}


def test_doc_lists_without_filters_matches_fused_top(tmp_path):
    a = ranking(tmp_path / "a.parquet", [(1, 1, 3.0), (1, 2, 2.0), (2, 5, 1.0)])
    assert bs.doc_lists({"systems": [a], "n": [[1, 1]]}) == {1: [1], 2: [5]}


def test_append_adds_new_stratum_docs_only_above_gate(tmp_path):
    main = ranking(tmp_path / "m.parquet", [(1, 1, 3.0), (1, 2, 2.0), (2, 5, 1.0)])
    extra = ranking(tmp_path / "e.parquet", [(1, 2, 9.0), (1, 7, 8.0), (1, 8, 7.0), (2, 6, 1.0)])
    job = {"systems": [main], "n": [10], "append": {"systems": [extra], "n": [10], "k": 1, "gate_min_n": 2}}
    # query 1 has 2 docs -> gets 7 (2 is already listed, k=1); query 2 has 1 doc -> unchanged
    assert bs.doc_lists(job) == {1: [1, 2, 7], 2: [5]}


def test_chunk_extra_leaves_docs_alone(tmp_path):
    main = ranking(tmp_path / "m.parquet", [(1, 1, 3.0), (1, 2, 2.0)])
    extra = ranking(tmp_path / "e.parquet", [(1, 2, 9.0), (1, 7, 8.0), (1, 8, 7.0)])
    job = {"systems": [main], "n": [10], "chunk_append": {"systems": [extra], "n": [10], "k": 5}}
    by_q = bs.doc_lists(job)
    assert by_q == {1: [1, 2]}
    assert bs.chunk_extra(job, by_q) == {1: [7, 8]}


def test_list_depth_keeps_base_while_cap_is_larger(tmp_path):
    main = ranking(tmp_path / "m.parquet", [(1, d, 10.0 - d) for d in range(1, 6)])
    extra = ranking(tmp_path / "e.parquet", [(1, 9, 1.0)])
    job = {"systems": [main], "n": [[100, 2]], "list_depth": 2, "append": {"systems": [extra], "n": [10]}}
    assert bs.doc_lists(job) == {1: [1, 2, 9]}


def test_base_lengths_only_with_append(tmp_path):
    main = ranking(tmp_path / "m.parquet", [(1, 1, 3.0), (1, 2, 2.0)])
    extra = ranking(tmp_path / "e.parquet", [(1, 7, 8.0)])
    job = {"systems": [main], "n": [[10, 10]], "append": {"systems": [extra], "n": [10], "chunks": 1}}
    fused = bs.fuse(job)
    assert bs.base_lengths(job, fused) == {1: 2}
    assert bs.base_lengths({k: v for k, v in job.items() if k != "append"}, fused) is None
