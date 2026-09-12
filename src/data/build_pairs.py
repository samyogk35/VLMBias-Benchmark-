"""Build the optical-illusion canonical/counterfactual pair manifest (SPEC §4).

Pairing rule (frozen for v0.1):
  * domain = Optical Illusion, prompt form Q1, resolution 768 px (SPEC §4 default).
  * canonical      = the `difference = 0` render (both elements genuinely equal; GT = Yes).
  * counterfactual = a `difference != 0` render (genuinely unequal; GT = No) with the SAME
    illusion type and SAME signed illusion_strength (or size_min for Vertical-Horizontal).
  * Several counterfactuals (different |difference|) share one canonical; they share a
    `template_id`, and analyses must cluster on it.
  * `smoke_subset` marks one pair per template (the largest |difference|) for pipeline smoke tests.

Images are exported from the parquet snapshot into data/images/optical/ (git-ignored),
flattened onto white. Provenance = dataset revision + original VLMBias row ID.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.common import load_split_metadata, iter_images, DATASET_REVISION, DATASET_REPO  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(ROOT, "data", "images", "optical")
MANIFEST_VERSION = "0.1"

INTERVENTIONS = {
    "Ebbinghaus": "inner circles made genuinely unequal (difference != 0)",
    "MullerLyer": "horizontal lines made genuinely unequal in length",
    "Ponzo": "horizontal lines made genuinely unequal in length",
    "Zollner": "main diagonal lines made genuinely non-parallel",
    "Poggendorff": "diagonal segments made genuinely non-collinear",
    "VerticalHorizontal": "horizontal and vertical lines made genuinely unequal",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Q1")
    ap.add_argument("--pixel", type=int, default=768)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "manifests", f"optical_pairs_v{MANIFEST_VERSION}.jsonl"))
    ap.add_argument("--no-export", action="store_true", help="skip image export (manifest only)")
    args = ap.parse_args()

    df = load_split_metadata("main")
    df = df[(df.topic == "Optical Illusion") & (df.type_of_question == args.question) & (df.pixel == args.pixel)].copy()
    df["meta"] = df.metadata.apply(json.loads)
    df["illusion"] = df.meta.apply(lambda m: m["illusion_type"])
    df["strength"] = df.meta.apply(lambda m: int(m.get("strength", 0)))
    df["size_min"] = df.meta.apply(lambda m: round(float(m.get("size_min", 0.0)), 2))
    df["diff"] = df.meta.apply(lambda m: round(float(m.get("diff", 0.0)), 2))
    df["is_zero"] = df.meta.apply(lambda m: bool(m.get("is_diff_zero", False)))
    df["template_key"] = df.apply(lambda r: (r.illusion, r.strength, r.size_min), axis=1)

    canon = df[df.is_zero]
    cf = df[~df.is_zero]
    # sanity: canonical rows have GT Yes, counterfactual rows GT No (per generator)
    assert set(canon.ground_truth) == {"Yes"} and set(cf.ground_truth) == {"No"}, "unexpected GT layout"

    canon_by_key = {r.template_key: r for r in canon.itertuples()}
    pairs = []
    for r in cf.itertuples():
        c = canon_by_key.get(r.template_key)
        if c is None:
            continue  # no canonical at this strength (should not happen for released params)
        assert c.prompt == r.prompt, "prompt mismatch within pair"
        ill = r.illusion
        if ill == "VerticalHorizontal":
            tmpl = f"{ill}_min{r.size_min:g}"
        else:
            tmpl = f"{ill}_str{r.strength}"
        pair_id = f"opt_{tmpl}_diff{r.diff:g}".replace("-", "neg").replace(".", "p")
        template_id = f"opt_{tmpl}".replace("-", "neg").replace(".", "p")
        pairs.append({
            "pair_id": pair_id,
            "domain": "optical_illusion",
            "sub_domain": ill,
            "template_id": template_id,
            "prompt": r.prompt,
            "answer_type": "yes_no",
            "question_form": args.question,
            "resolution": args.pixel,
            "canonical_path": f"data/images/optical/{c.image_path.split('/')[-1]}",
            "counterfactual_path": f"data/images/optical/{r.image_path.split('/')[-1]}",
            "canonical_gt": c.ground_truth,
            "counterfactual_gt": r.ground_truth,
            # expected_bias as released by VLMBias for each member (canonical: "No", counterfactual: "Yes")
            "canonical_expected_bias": c.expected_bias,
            "counterfactual_expected_bias": r.expected_bias,
            "expected_bias": r.expected_bias,  # SPEC field: the prior/canonical answer on the counterfactual
            "intervention": INTERVENTIONS[ill],
            "generator_params": {
                "illusion_type": ill, "illusion_strength": r.strength, "size_min": r.size_min,
                "canonical_difference": 0.0, "counterfactual_difference": r.diff,
                "renderer": "pyllusion via third_party/vlms-are-biased/generators/optical_illusion_generator.py",
            },
            "source_revision": {
                "dataset": DATASET_REPO, "revision": DATASET_REVISION, "split": "main",
                "canonical_row_id": c.ID, "counterfactual_row_id": r.ID,
            },
            "smoke_subset": False,
            "qc_status": "auto_matched",
            "qc_notes": "parameter-matched from VLMBias metadata; prompt/resolution identity asserted programmatically; manual visual review pending",
            "manifest_version": MANIFEST_VERSION,
        })

    # mark one pair per template (largest |difference|) for the smoke subset
    best = {}
    for p in pairs:
        d = abs(p["generator_params"]["counterfactual_difference"])
        if p["template_id"] not in best or d > best[p["template_id"]][0]:
            best[p["template_id"]] = (d, p["pair_id"])
    smoke_ids = {v[1] for v in best.values()}
    for p in pairs:
        p["smoke_subset"] = p["pair_id"] in smoke_ids

    pairs.sort(key=lambda p: p["pair_id"])
    ids = [p["pair_id"] for p in pairs]
    assert len(ids) == len(set(ids)), "duplicate pair_id"

    if not args.no_export:
        os.makedirs(IMG_DIR, exist_ok=True)
        want = {p["source_revision"]["canonical_row_id"] for p in pairs} | {p["source_revision"]["counterfactual_row_id"] for p in pairs}
        id2file = {r.ID: r.image_path.split("/")[-1] for r in df.itertuples()}
        n = 0
        for rid, im in iter_images("main", want):
            out = os.path.join(IMG_DIR, id2file[rid])
            if not os.path.exists(out):
                im.save(out)
            n += 1
        print(f"exported/verified {n} images to {IMG_DIR}")
        # record actual dimensions; pyllusion sizes the Ebbinghaus canvas from content, so widths
        # can differ by a few px within a pair. Height must match; width mismatch is flagged for QC.
        from PIL import Image
        for p in pairs:
            a = Image.open(os.path.join(ROOT, p["canonical_path"])).size
            b = Image.open(os.path.join(ROOT, p["counterfactual_path"])).size
            p["canonical_size"] = list(a)
            p["counterfactual_size"] = list(b)
            assert a[1] == b[1] == args.pixel, (p["pair_id"], a, b)
            if a != b:
                p["qc_status"] = "auto_matched_dim_warn"
                p["qc_notes"] += f"; WIDTH MISMATCH canonical={a[0]}px counterfactual={b[0]}px (renderer-determined canvas)"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs ({len(smoke_ids)} in smoke subset, {len(set(p['template_id'] for p in pairs))} templates) -> {args.out}")
    from collections import Counter
    print(Counter(p["sub_domain"] for p in pairs))


if __name__ == "__main__":
    main()
