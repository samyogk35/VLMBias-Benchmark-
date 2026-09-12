# small made-up dataset where I know what the transition counts should be
import pandas as pd
import pytest

from src.analysis.summarize import pair_table, condition_summary, transitions


def rec(cond, seed, pair, variant, correct, valid=True):
    return {"condition": cond, "seed": seed, "pair_id": pair, "sub_domain": "X", "template_id": pair,
            "image_variant": variant, "is_correct": correct if valid else None,
            "is_bias_answer": (not correct) if valid else None,
            "parse_status": "valid" if valid else "invalid", "raw_output": "", "config_hash": "h",
            "model_revision": "m", "duration_ms": 1.0, "parsed_answer": None}


@pytest.fixture
def df():
    rows = []
    # p1: vcd fixes the counterfactual, canonical stays right -> pair improves
    rows += [rec("regular", 1, "p1", "canonical", True), rec("regular", 1, "p1", "counterfactual", False),
             rec("vcd", 1, "p1", "canonical", True), rec("vcd", 1, "p1", "counterfactual", True)]
    # p2: vcd fixes the counterfactual but breaks the canonical -> tradeoff
    rows += [rec("regular", 1, "p2", "canonical", True), rec("regular", 1, "p2", "counterfactual", False),
             rec("vcd", 1, "p2", "canonical", False), rec("vcd", 1, "p2", "counterfactual", True)]
    # p3: all correct under both
    rows += [rec("regular", 1, "p3", "canonical", True), rec("regular", 1, "p3", "counterfactual", True),
             rec("vcd", 1, "p3", "canonical", True), rec("vcd", 1, "p3", "counterfactual", True)]
    # p4: vcd gives an unparseable answer on the counterfactual
    rows += [rec("regular", 1, "p4", "canonical", True), rec("regular", 1, "p4", "counterfactual", True),
             rec("vcd", 1, "p4", "canonical", True), rec("vcd", 1, "p4", "counterfactual", False, valid=False)]
    d = pd.DataFrame(rows)
    d["valid"] = d.parse_status == "valid"
    d["correct"] = d.is_correct.fillna(False).astype(bool)
    d["bias"] = d.is_bias_answer.fillna(False).astype(bool)
    return d


def test_metrics(df):
    s = condition_summary(pair_table(df)).set_index("condition")
    assert s.loc["regular", "pair_success"] == pytest.approx(2 / 4)
    assert s.loc["vcd", "pair_success"] == pytest.approx(2 / 4)
    assert s.loc["regular", "counterfactual_acc"] == pytest.approx(2 / 4)
    assert s.loc["vcd", "counterfactual_acc"] == pytest.approx(3 / 4)
    assert s.loc["vcd", "canonical_acc"] == pytest.approx(3 / 4)
    assert s.loc["vcd", "any_invalid_pair_rate"] > 0
    assert s.loc["regular", "any_invalid_pair_rate"] == 0


def test_transitions(df):
    tr = transitions(pair_table(df))
    assert tr["n_pair_seed"] == 4
    assert tr["CF_correction"] == 2 and tr["CF_regression"] == 1
    assert tr["C_correction"] == 0 and tr["C_regression"] == 1
    assert tr["pair_improvement"] == 1 and tr["pair_regression"] == 1
    assert tr["tradeoff_only"] == 1
