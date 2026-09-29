# manifest builder for the raw VLMBias split, checked against the tracked inventory csv
import os

import pandas as pd
import pytest

from src.data.build_benchmark_manifest import case_id, answer_type, rows_to_manifest, mark_smoke
from src.analysis.summarize_benchmark import ENAR_LLAVA, topic_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def inventory():
    return pd.read_csv(os.path.join(ROOT, "data", "inventory", "main_rows.csv"))


def test_case_id():
    assert case_id("x/images/Ebbinghaus_001_str3_diff0_notitle_px768.png") == "Ebbinghaus_001_str3_diff0"
    assert case_id("x/images/dice_001_remove_C3_notitle_px384.png") == "dice_001_remove_C3"
    assert case_id("x/images/nike-black-soccer-1_1152.png") == "nike-black-soccer-1"
    assert case_id("x/images/Flag of Cuba-stripes=4_384.png") == "Flag of Cuba-stripes=4"
    assert case_id("x/images/horse_2_2_384.png") == "horse_2_2"


def test_case_ids_collapse_to_464(inventory):
    cases = inventory.image_path.map(case_id)
    assert cases.nunique() == 464
    # every case has exactly Q1 and Q2 at each of 3 resolutions
    assert set(cases.value_counts()) == {6}


@pytest.mark.parametrize("pixel", [384, 768, 1152])
def test_manifest_matches_enar_denominators(inventory, pixel):
    rows = rows_to_manifest(inventory, pixel)
    assert len(rows) == 928
    per_topic = pd.Series([r["domain"] for r in rows]).value_counts()
    for topic, v in ENAR_LLAVA.items():
        assert per_topic[topic] == 2 * v["cases"], topic
    assert all(r["resolution"] == pixel for r in rows)
    # filenames carry the resolution as _px768 or a bare _768 depending on the topic's generator
    assert all(r["counterfactual_path"].endswith(("px%d.png" % pixel, "_%d.png" % pixel)) for r in rows)
    # all resolutions export into one directory, so filenames must not collide (Q1/Q2 share one image)
    files = {r["counterfactual_path"] for r in rows}
    assert len(files) == 464


def test_answer_types_and_gt_are_strings(inventory):
    rows = rows_to_manifest(inventory, 768)
    for r in rows:
        assert isinstance(r["counterfactual_gt"], str)
        if r["domain"] == "optical_illusion":
            assert r["answer_type"] == "yes_no" and r["counterfactual_gt"] in ("Yes", "No")
        else:
            assert r["answer_type"] == "number" and r["counterfactual_gt"].isdigit()
        # the familiar answer must differ from the truth, otherwise it isn't a counterfactual
        assert r["counterfactual_gt"] != r["expected_bias"]


def test_smoke_subset_is_stratified_q1(inventory):
    rows = mark_smoke(rows_to_manifest(inventory, 768), per_topic=15)
    smoke = [r for r in rows if r["smoke_subset"]]
    assert len(smoke) == 15 * 7
    assert all(r["question_form"] == "Q1" for r in smoke)
    assert len({r["template_id"] for r in smoke}) == len(smoke)  # one row per case
    assert pd.Series([r["domain"] for r in smoke]).value_counts().eq(15).all()


def test_topic_table_against_enar():
    # fake a run that reproduces EnAR exactly: rows correct in the published proportions
    recs = []
    for topic, v in ENAR_LLAVA.items():
        n = 2 * v["cases"]
        for c in ("regular", "vcd"):
            for i in range(n):
                recs.append({"domain": topic, "condition": c, "seed": 42, "pair_id": "%s_%d_Q1_" % (topic, i),
                             "correct": i < v[c], "bias": i >= v[c], "valid": True, "parse_status": "valid"})
    t = topic_table(pd.DataFrame(recs), ["regular", "vcd"]).set_index("topic")
    assert t.loc["total", "enar_regular"] == pytest.approx(16.92, abs=0.005)
    assert t.loc["total", "enar_vcd"] == pytest.approx(19.18, abs=0.005)
    for topic in ENAR_LLAVA:
        assert t.loc[topic, "acc_regular"] == pytest.approx(t.loc[topic, "enar_regular"])
        assert t.loc[topic, "acc_vcd"] == pytest.approx(t.loc[topic, "enar_vcd"])
    assert t.loc["total", "n_regular"] == 928
