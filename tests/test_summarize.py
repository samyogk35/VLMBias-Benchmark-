"""Synthetic fixture with known transitions (SPEC §12 analysis validation)."""
import pandas as pd
import pytest

from src.analysis.summarize import pair_table, condition_summary, transitions


def _rec(cond, seed, pair, variant, correct, valid=True):
    return {"condition": cond, "seed": seed, "pair_id": pair, "sub_domain": "X", "template_id": pair,
            "image_variant": variant, "is_correct": correct if valid else None, "is_bias_answer": (not correct) if valid else None,
            "parse_status": "valid" if valid else "invalid", "raw_output": "", "config_hash": "h", "model_revision": "m",
            "duration_ms": 1.0, "parsed_answer": None}


@pytest.fixture
def df():
    rows = []
    # pair p1: regular fails CF, vcd fixes CF, canonical stays correct -> pair_improvement + CF_correction
    rows += [_rec("regular", 1, "p1", "canonical", True), _rec("regular", 1, "p1", "counterfactual", False),
             _rec("vcd", 1, "p1", "canonical", True), _rec("vcd", 1, "p1", "counterfactual", True)]
    # pair p2: vcd fixes CF but breaks canonical -> tradeoff_only, CF_correction, C_regression
    rows += [_rec("regular", 1, "p2", "canonical", True), _rec("regular", 1, "p2", "counterfactual", False),
             _rec("vcd", 1, "p2", "canonical", False), _rec("vcd", 1, "p2", "counterfactual", True)]
    # pair p3: both fully correct under both -> no transitions
    rows += [_rec("regular", 1, "p3", "canonical", True), _rec("regular", 1, "p3", "counterfactual", True),
             _rec("vcd", 1, "p3", "canonical", True), _rec("vcd", 1, "p3", "counterfactual", True)]
    # pair p4: vcd invalid parse on CF -> counts as not correct (CF_regression), invalid_rate > 0
    rows += [_rec("regular", 1, "p4", "canonical", True), _rec("regular", 1, "p4", "counterfactual", True),
             _rec("vcd", 1, "p4", "canonical", True), _rec("vcd", 1, "p4", "counterfactual", False, valid=False)]
    d = pd.DataFrame(rows)
    d["valid"] = d.parse_status == "valid"
    d["correct"] = d.is_correct.fillna(False).astype(bool)
    d["bias"] = d.is_bias_answer.fillna(False).astype(bool)
    return d


def test_metrics(df):
    pt = pair_table(df)
    s = condition_summary(pt).set_index("condition")
    assert s.loc["regular", "pair_success"] == pytest.approx(2 / 4)
    assert s.loc["vcd", "pair_success"] == pytest.approx(2 / 4)       # p1, p3
    assert s.loc["regular", "counterfactual_acc"] == pytest.approx(2 / 4)
    assert s.loc["vcd", "counterfactual_acc"] == pytest.approx(3 / 4)
    assert s.loc["vcd", "canonical_acc"] == pytest.approx(3 / 4)
    assert s.loc["vcd", "any_invalid_pair_rate"] > 0 and s.loc["regular", "any_invalid_pair_rate"] == 0


def test_transitions(df):
    tr = transitions(pair_table(df))
    assert tr["n_pair_seed"] == 4
    assert tr["CF_correction"] == 2 and tr["CF_regression"] == 1
    assert tr["C_correction"] == 0 and tr["C_regression"] == 1
    assert tr["pair_improvement"] == 1 and tr["pair_regression"] == 1
    assert tr["tradeoff_only"] == 1
