import numpy as np
import pandas as pd
import pytest
from cg import governance as gv
from cg.nlp import add_concerns, extract_concerns
from cg.synth import generate

POP = {"student": .5, "teacher": .5}


def toy(ns_open=2, nt_restricted=8):
    rows = [{"group": "student", "stance": "open", "severity": 3, "text": "x", "subgroup": "none"}] * ns_open
    rows += [{"group": "teacher", "stance": "restricted", "severity": 3, "text": "x", "subgroup": "none"}] * nt_restricted
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.mark.parametrize("s", gv.SCHEMES)
def test_weights_sum_to_one(s):
    assert gv.weights(toy(), s, POP).sum() == pytest.approx(1)


def test_majority_vs_balanced_can_flip():
    df = toy(2, 8)
    assert gv.compare_schemes(df, POP).set_index("scheme").loc["majority", "winner"] == "restricted"
    sh, _ = gv.tally(df, gv.weights(df, "balanced", POP))
    assert sh["open"] == pytest.approx(.5) and sh["restricted"] == pytest.approx(.5)


def test_counterfactual_share_monotone_and_flip():
    df = toy(2, 8)
    assert gv.counterfactual(df, "student", 0.2)["winner"] == "restricted"
    assert gv.counterfactual(df, "student", 0.8)["winner"] == "open"
    f = gv.flip_threshold(df, "student")
    assert f["flips"] and 0.49 <= f["threshold_share"] <= 0.52 and f["new_winner"] == "open"


def test_audit_flags_underrepresentation():
    a = gv.representation_audit(toy(2, 8), POP).set_index("group")
    assert a.loc["student", "status"] == "under" and a.loc["teacher", "status"] == "over"


def test_extractor_and_subgroup_suppression():
    assert "privacy" in extract_concerns("Worried about student data") and extract_concerns("hello") == []
    df = add_concerns(generate(300, 1))
    assert gv.subgroup_audit(df, k=10_000)["suppressed"].all()


def test_bootstrap_and_pipeline_smoke():
    from cg.pipeline import run_all
    df = generate(200, 3)
    b = gv.bootstrap(add_concerns(df).reset_index(drop=True), "balanced", n=30)
    assert b["win_freq"].sum() == pytest.approx(1)
    res = run_all(df, n_boot=20)
    assert res["schemes"]["winner"].isin(gv.POLICIES).all()
    assert res["retriever"].evaluate()[1] >= 0.8
