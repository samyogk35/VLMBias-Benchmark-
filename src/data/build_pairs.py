# Build the optical illusion pair manifest.
#
# VLMBias renders each illusion twice: once with difference=0 (the two things really are equal,
# answer Yes) and once with difference != 0 (they really are different, answer No), at the same
# illusion strength. I treat the difference=0 one as the canonical image and the other as the
# counterfactual. Same prompt, same resolution, only the intervention differs.
#
# I use Q1 at 768px. Several counterfactuals (different |difference|) share one canonical, they
# get the same template_id. One pair per template is flagged smoke_subset for quick pipeline runs.
# Images are pulled out of the parquet into data/images/optical/ (gitignored).

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.common import load_split_metadata, iter_images, DATASET_REVISION, DATASET_REPO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(ROOT, "data", "images", "optical")
MANIFEST_VERSION = "0.1"

INTERVENTIONS = {
    "Ebbinghaus": "inner circles made genuinely unequal",
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
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "manifests", "optical_pairs_v%s.jsonl" % MANIFEST_VERSION))
    ap.add_argument("--no-export", action="store_true", help="don't export the images")
    args = ap.parse_args()

    df = load_split_metadata("main")
    df = df[(df.topic == "Optical Illusion") & (df.type_of_question == args.question) & (df.pixel == args.pixel)].copy()
    df["meta"] = df.metadata.apply(json.loads)
    df["illusion"] = df.meta.apply(lambda m: m["illusion_type"])
    df["strength"] = df.meta.apply(lambda m: int(m.get("strength", 0)))
    df["size_min"] = df.meta.apply(lambda m: round(float(m.get("size_min", 0.0)), 2))
    df["diff"] = df.meta.apply(lambda m: round(float(m.get("diff", 0.0)), 2))  # stored as 0.6000000000000001 etc
    df["is_zero"] = df.meta.apply(lambda m: bool(m.get("is_diff_zero", False)))
    df["key"] = df.apply(lambda r: (r.illusion, r.strength, r.size_min), axis=1)

    canon = df[df.is_zero]
    cf = df[~df.is_zero]
    assert set(canon.ground_truth) == {"Yes"} and set(cf.ground_truth) == {"No"}

    canon_by_key = {r.key: r for r in canon.itertuples()}
    pairs = []
    for r in cf.itertuples():
        c = canon_by_key.get(r.key)
        if c is None:
            # e.g. Ponzo strength 18 has counterfactuals but no difference=0 render
            continue
        assert c.prompt == r.prompt
        ill = r.illusion
        tmpl = "%s_min%g" % (ill, r.size_min) if ill == "VerticalHorizontal" else "%s_str%d" % (ill, r.strength)
        template_id = ("opt_" + tmpl).replace("-", "neg").replace(".", "p")
        pair_id = ("opt_%s_diff%g" % (tmpl, r.diff)).replace("-", "neg").replace(".", "p")
        pairs.append({
            "pair_id": pair_id,
            "domain": "optical_illusion",
            "sub_domain": ill,
            "template_id": template_id,
            "prompt": r.prompt,
            "answer_type": "yes_no",
            "question_form": args.question,
            "resolution": args.pixel,
            "canonical_path": "data/images/optical/" + c.image_path.split("/")[-1],
            "counterfactual_path": "data/images/optical/" + r.image_path.split("/")[-1],
            "canonical_gt": c.ground_truth,
            "counterfactual_gt": r.ground_truth,
            "canonical_expected_bias": c.expected_bias,
            "counterfactual_expected_bias": r.expected_bias,
            "expected_bias": r.expected_bias,
            "intervention": INTERVENTIONS[ill],
            "generator_params": {
                "illusion_type": ill, "illusion_strength": r.strength, "size_min": r.size_min,
                "canonical_difference": 0.0, "counterfactual_difference": r.diff,
                "renderer": "pyllusion, see third_party/vlms-are-biased/generators/optical_illusion_generator.py",
            },
            "source_revision": {
                "dataset": DATASET_REPO, "revision": DATASET_REVISION, "split": "main",
                "canonical_row_id": c.ID, "counterfactual_row_id": r.ID,
            },
            "smoke_subset": False,
            "qc_status": "auto_matched",
            "qc_notes": "matched on metadata, prompt and resolution checked in code, not yet looked at by eye",
            "manifest_version": MANIFEST_VERSION,
        })

    # smoke subset: the biggest |difference| per template
    best = {}
    for p in pairs:
        d = abs(p["generator_params"]["counterfactual_difference"])
        if p["template_id"] not in best or d > best[p["template_id"]][0]:
            best[p["template_id"]] = (d, p["pair_id"])
    smoke_ids = {v[1] for v in best.values()}
    for p in pairs:
        p["smoke_subset"] = p["pair_id"] in smoke_ids

    pairs.sort(key=lambda p: p["pair_id"])
    assert len({p["pair_id"] for p in pairs}) == len(pairs)

    if not args.no_export:
        os.makedirs(IMG_DIR, exist_ok=True)
        want = set()
        for p in pairs:
            want.add(p["source_revision"]["canonical_row_id"])
            want.add(p["source_revision"]["counterfactual_row_id"])
        id2file = {r.ID: r.image_path.split("/")[-1] for r in df.itertuples()}
        n = 0
        for rid, im in iter_images("main", want):
            out = os.path.join(IMG_DIR, id2file[rid])
            if not os.path.exists(out):
                im.save(out)
            n += 1
        print("exported %d images to %s" % (n, IMG_DIR))

        # pyllusion picks the Ebbinghaus canvas width from the content, so the two images in a
        # pair can differ by a few px in width. heights must match, widths get flagged.
        from PIL import Image
        for p in pairs:
            a = Image.open(os.path.join(ROOT, p["canonical_path"])).size
            b = Image.open(os.path.join(ROOT, p["counterfactual_path"])).size
            p["canonical_size"] = list(a)
            p["counterfactual_size"] = list(b)
            assert a[1] == b[1] == args.pixel, (p["pair_id"], a, b)
            if a != b:
                p["qc_status"] = "auto_matched_dim_warn"
                p["qc_notes"] += "; width differs: canonical %dpx vs counterfactual %dpx" % (a[0], b[0])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("wrote %d pairs (%d smoke, %d templates) -> %s"
          % (len(pairs), len(smoke_ids), len({p["template_id"] for p in pairs}), args.out))
    print(Counter(p["sub_domain"] for p in pairs))


if __name__ == "__main__":
    main()
