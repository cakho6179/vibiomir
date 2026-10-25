"""decide.py on AnyJev's FakeBackend (vendored wheel): readout columns, shared path = Decider, batch independence."""
from __future__ import annotations

import math

import pytest

from r2ai2026.stages import decide as stage

stage.import_anyjev()
from anyjev.backends.fake import FakeBackend  # noqa: E402


def content(state: str, option: str) -> float:
    relevant = "viem da day" in state.split("Document:", 1)[-1]
    return (2.0 if relevant else -2.0) if option == "Yes" else 0.0


def scores(states, label_prior=None, prior="none"):
    backend = FakeBackend(content, label_prior=label_prior or {})
    decider, question = stage.make_decider(backend, "L0", prior)
    return stage.decide(decider, question, states), backend


def test_render_cuts_passage():
    s = stage.render("dau bung?", "x" * 5000, 1500)
    assert s.startswith("Patient question (Vietnamese):\ndau bung?") and s.endswith("x" * 1500)
    assert len(s) < 1600


def test_columns_and_ranking():
    states = [stage.render("q", "bai ve viem da day", 1500), stage.render("q", "bai ve gan", 1500)]
    df, backend = scores(states)
    assert list(df.columns) == ["score", "p_yes", "logodds", "p_yes_yes_first", "p_yes_no_first", "mass"]
    assert df["p_yes"][0] > 0.8 > 0.2 > df["p_yes"][1]
    assert df["logodds"][0] == pytest.approx(math.log(df["p_yes"][0] / (1 - df["p_yes"][0])))


def test_shared_path_matches_decider():
    """decide_shared = Decider(L0, prior none), with one shared-prefix call per chunk instead of two full prompts."""
    states = [stage.render("q", t, 1500) for t in ["bai ve viem da day", "bai ve gan", "viem da day cap"]]
    backend = FakeBackend(content, label_prior={"Yes": 0.7}, position_bias=[0.4, 0.0])
    decider, question = stage.make_decider(backend, "L0", "none")
    ref = stage.decide(decider, question, states)
    fast = stage.decide_shared(backend, question, states)
    assert backend.shared_groups == len(states)
    for col in ref.columns:
        assert fast[col].tolist() == pytest.approx(ref[col].tolist(), abs=1e-9)


def test_label_bias_keeps_ranking():
    """A planted preference for the token 'Yes' moves the raw phrasings but not the L0 log-odds ranking."""
    states = [stage.render("q", "bai ve viem da day", 1500), stage.render("q", "bai ve gan", 1500)]
    clean, _ = scores(states)
    biased, _ = scores(states, label_prior={"Yes": 1.5})
    assert (biased["p_yes"] > clean["p_yes"]).all()          # no prior correction: the bias stays
    assert biased["logodds"][0] > biased["logodds"][1]        # ranking unchanged


def test_score_independent_of_batch():
    """prior='none': a pair scores the same alone or among others (reproducible across shards)."""
    a = stage.render("q", "bai ve viem da day", 1500)
    others = [stage.render("q", f"bai {i} ve gan", 1500) for i in range(20)]
    alone, _ = scores([a])
    mixed, _ = scores(others + [a])
    assert mixed["p_yes"].iloc[-1] == pytest.approx(alone["p_yes"][0])


def test_graded_question_expected_level():
    def graded(state: str, option: str) -> float:
        relevant = "viem da day" in state.split("Document:", 1)[-1]
        return 3.0 if relevant and option.endswith("directly answers what is asked") else 0.0

    backend = FakeBackend(graded)
    decider, question = stage.make_decider(backend, "L0", "none", "graded")
    df = stage.decide(decider, question, [stage.render("q", "bai ve viem da day", 1500),
                                          stage.render("q", "bai ve gan", 1500)])
    assert list(df.columns) == ["score", "p_l0", "p_l1", "p_l2", "p_l3", "p_l4", "mass"]
    assert df["score"][0] > 3 > 2.5 > df["score"][1] > 1.5  # second state: uniform over 5 levels = 2.0
