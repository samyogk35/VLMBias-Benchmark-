import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "manifests", "optical_pairs_v0.1.jsonl")
REQUIRED = ["pair_id", "domain", "template_id", "prompt", "resolution", "canonical_path", "counterfactual_path",
            "canonical_gt", "counterfactual_gt", "expected_bias", "intervention", "generator_params",
            "source_revision", "qc_status", "qc_notes"]


@pytest.fixture(scope="module")
def rows():
    assert os.path.exists(MANIFEST), "run src/data/build_pairs.py first"
    return [json.loads(l) for l in open(MANIFEST)]


def test_required_fields(rows):
    for r in rows:
        for k in REQUIRED:
            assert k in r, (r.get("pair_id"), k)


def test_pair_ids_unique(rows):
    ids = [r["pair_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_targets_differ(rows):
    for r in rows:
        assert {r["canonical_gt"], r["counterfactual_gt"]} == {"Yes", "No"}, r["pair_id"]
        assert r["canonical_path"] != r["counterfactual_path"]


def test_one_canonical_per_template(rows):
    by = {}
    for r in rows:
        by.setdefault(r["template_id"], set()).add(r["canonical_path"])
    for t, s in by.items():
        assert len(s) == 1, (t, s)


def test_single_resolution_and_prompt(rows):
    assert {r["resolution"] for r in rows} == {768}
    assert {r["question_form"] for r in rows} == {"Q1"}


def test_smoke_subset(rows):
    assert len(rows) >= 10
    smoke = [r for r in rows if r["smoke_subset"]]
    assert 10 <= len(smoke) <= 25
    assert len({r["template_id"] for r in smoke}) == len(smoke)


def test_generator_params(rows):
    for r in rows:
        gp = r["generator_params"]
        assert gp["canonical_difference"] == 0.0
        assert gp["counterfactual_difference"] != 0.0
        assert r["sub_domain"] == gp["illusion_type"]


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "data", "images", "optical")), reason="images not exported")
def test_images_exist_and_heights_match(rows):
    from PIL import Image
    for r in rows:
        a = os.path.join(ROOT, r["canonical_path"])
        b = os.path.join(ROOT, r["counterfactual_path"])
        assert os.path.exists(a) and os.path.exists(b), r["pair_id"]
        assert Image.open(a).size[1] == Image.open(b).size[1] == r["resolution"]
