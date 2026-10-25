"""Tests for the local replica of the R2AI2026 scoring rule."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_normalize_text_applies_official_pipeline():
    from r2ai2026.core.scoring import normalize_text

    assert normalize_text("  BỆNH\tTim   Mạch \n") == "bệnh tim mạch"
    assert normalize_text("Nhồi&nbsp;máu cơ tim") == "nhồi máu cơ tim"
    # NFKC folds compatibility characters (ligature, full-width digits)
    assert normalize_text("ﬁbrin １２３") == "fibrin 123"


def test_lcs_length_matches_dynamic_programming():
    from r2ai2026.core.scoring import _lcs_dp, lcs_length

    rng = random.Random(20261018)
    for _ in range(50):
        a = [rng.choice("abc") for _ in range(rng.randint(0, 10))]
        b = [rng.choice("abc") for _ in range(rng.randint(0, 10))]
        assert lcs_length(a, b) == _lcs_dp(a, b)

    # order matters: gaps allowed, crossings not
    assert lcs_length(list("abc"), list("acb")) == 2
    assert lcs_length([], list("abc")) == 0


def test_chunk_overlap_uses_gold_length_as_denominator():
    from r2ai2026.core.scoring import chunk_overlap, tokenize

    gold = tokenize("đau ngực khi gắng sức kèm khó thở")
    assert chunk_overlap(gold, gold) == 1.0

    long_chunk = tokenize("bệnh nhân " + " ".join(["mở rộng"] * 200) + " đau ngực khi gắng sức kèm khó thở")
    assert chunk_overlap(long_chunk, gold) == 1.0  # gold span contained verbatim

    partial = tokenize("đau ngực " + " ".join(["khác"] * 40))
    assert chunk_overlap(partial, gold) < 0.4


def test_score_query_document_level_precision_recall_f2():
    from r2ai2026.core.scoring import score_query

    score = score_query(
        predicted_docs=[1, 2, 3, 4],
        gold_docs=[1, 2, 9],
        predicted_chunks=[],
        gold_chunks=[],
    )
    assert score.docs_precision == 0.5  # 2 of 4 submitted are gold
    assert round(score.docs_recall, 4) == round(2 / 3, 4)
    # F2 = 5PR/(4P+R)
    expected = 5 * 0.5 * (2 / 3) / (4 * 0.5 + 2 / 3)
    assert abs(score.docs_f2 - expected) < 1e-9


def test_score_query_empty_prediction_scores_zero():
    from r2ai2026.core.scoring import score_query

    score = score_query([], [1, 2], [], [(1, "đau đầu kéo dài")])
    assert score.docs_precision == 0.0 and score.docs_recall == 0.0 and score.docs_f2 == 0.0
    assert score.chunks_recall == 0.0 and score.chunks_f2 == 0.0


def test_chunk_level_threshold_boundary_and_recall_accounting():
    from r2ai2026.core.scoring import score_query

    gold = "một hai ba bốn năm sáu bảy tám chín mười"  # 10 tokens
    # 4/10 = 0.40 -> counts as correct (rule is ">= 0.40")
    boundary_chunk = "một hai ba bốn"
    below_chunk = "một hai ba"
    score = score_query(
        predicted_docs=[7],
        gold_docs=[7],
        predicted_chunks=[(7, boundary_chunk), (7, below_chunk)],
        gold_chunks=[(7, gold)],
    )
    assert score.chunks_precision == 0.5  # one of two submitted chunks is correct
    assert score.chunks_recall == 1.0  # the single gold chunk was found by the boundary chunk
    assert round(score.chunks_f2, 4) == round(5 * 0.5 * 1.0 / (4 * 0.5 + 1.0), 4)


def test_whole_document_chunk_finds_all_gold_chunks():
    """A verbatim full document submitted as one chunk recovers every gold span in it."""
    from r2ai2026.core.scoring import score_query

    doc = " ".join(f"t{i}" for i in range(1, 500))
    gold_chunks = [
        (3, " ".join(f"t{i}" for i in range(10, 30))),
        (3, " ".join(f"t{i}" for i in range(200, 240))),
    ]
    score = score_query([3], [3], [(3, doc)], gold_chunks)
    assert score.chunks_precision == 1.0
    assert score.chunks_recall == 1.0


def test_mismatched_doc_id_never_matches():
    from r2ai2026.core.scoring import score_query

    score = score_query(
        [1], [1], [(2, "đau ngực khi gắng sức")], [(1, "đau ngực khi gắng sức")]
    )
    assert score.chunks_precision == 0.0
    assert score.chunks_recall == 0.0


def test_score_submission_macro_average_and_final_score():
    from r2ai2026.core.scoring import score_submission

    references = {
        1: {"relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "sốt cao co giật"}]},
        2: {"relevant_docs": [20, 21], "relevant_chunks": []},
    }
    predictions = {
        1: {"relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "sốt cao co giật"}]},
        2: {"relevant_docs": [], "relevant_chunks": []},
    }
    result = score_submission(predictions, references)
    assert result.docs_f2 == result.per_query["1"].docs_f2  # query 2 contributes 0
    assert round(result.final_score, 6) == round((result.docs_f2 + result.chunks_f2) / 2, 6)
    assert result.queries_scored == 1  # query 1 has perfect precision/recall, query 2 scores 0
    assert result.queries_total == 2


def test_f_beta_weights_recall_more_than_precision():
    from r2ai2026.core.scoring import f_beta

    assert abs(f_beta(1.0, 0.5, 2.0) - 5 * 0.5 / 4.5) < 1e-9
    assert f_beta(1.0, 0.0) == 0.0
    assert f_beta(0.0, 0.0) == 0.0
