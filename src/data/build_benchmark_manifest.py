# Build a manifest over the raw VLMBias main split so run_inference.py can reproduce the
# published regular-vs-VCD numbers (EnAR Table 1: 16.92 / 19.18 on LLaVA-1.5-7B).
#
# Every VLMBias image is a counterfactual by construction (the familiar answer is always wrong),
# so each row becomes a single-variant manifest entry with only the counterfactual_* fields.
# Run it with --variants counterfactual. EnAR's 928-row denominator is one resolution with both
# Q1 and Q2 (see docs/enar_reproduction_notes.md), so the manifest is per resolution and keeps
# both prompt forms; template_id is the underlying case so statistics can dedup to one row per case.
#
# A stratified smoke_subset (--smoke-per-topic rows per topic, Q1 only) is flagged for pilots.
# Images are pulled out of the parquet into data/images/vlmbias_main/ (gitignored).

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.common import DATASET_REPO, DATASET_REVISION

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(ROOT, "data", "images", "vlmbias_main")
MANIFEST_VERSION = "0.1"

# The row IDs encode the prompt form differently per topic (_Q1, _prompt1, or a bare trailing _1 for
# flags), so the case is taken from the image filename instead: Q1 and Q2 share one image, and the
# filename carries the resolution as _px768 or a bare _768 plus an optional _notitle.
_CASE_STRIP = re.compile(r"_(px)?(384|768|1152)(?=_|$)|_notitle(?=_|$)")


def case_id(image_path):
    return _CASE_STRIP.sub("", os.path.basename(image_path).rsplit(".", 1)[0])


def answer_type(topic):
    return "yes_no" if topic == "Optical Illusion" else "number"


def domain_slug(topic):
    return topic.lower().replace(" ", "_")


def rows_to_manifest(df, pixel):
    """df has the inventory columns (ID, image_path, topic, sub_topic, prompt, ground_truth,
    expected_bias, type_of_question, pixel). Returns one manifest entry per row at `pixel`."""
    rows = []
    for r in df[df.pixel == pixel].itertuples():
        rows.append({
            "pair_id": r.ID,
            "domain": domain_slug(r.topic),
            "sub_domain": r.sub_topic,
            "template_id": case_id(r.image_path),
            "prompt": r.prompt,
            "answer_type": answer_type(r.topic),
            "question_form": r.type_of_question,
            "resolution": int(pixel),
            "counterfactual_path": "data/images/vlmbias_main/" + r.image_path.split("/")[-1],
            "counterfactual_gt": str(r.ground_truth),
            "counterfactual_expected_bias": str(r.expected_bias),
            "expected_bias": str(r.expected_bias),
            "source_revision": {
                "dataset": DATASET_REPO, "revision": DATASET_REVISION, "split": "main",
                "counterfactual_row_id": r.ID,
            },
            "smoke_subset": False,
            "manifest_version": MANIFEST_VERSION,
        })
    rows.sort(key=lambda p: p["pair_id"])
    assert len({p["pair_id"] for p in rows}) == len(rows)
    return rows


def mark_smoke(rows, per_topic):
    """Flag a stratified pilot: the first `per_topic` Q1 cases of every topic, in sorted order."""
    seen = Counter()
    for p in rows:
        if p["question_form"] != "Q1":
            continue
        if seen[p["domain"]] < per_topic:
            p["smoke_subset"] = True
            seen[p["domain"]] += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel", type=int, default=768)
    ap.add_argument("--smoke-per-topic", type=int, default=15, help="15 x 7 topics = 105-row pilot")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-export", action="store_true", help="don't export the images")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "data", "manifests", "vlmbias_main_px%d_v%s.jsonl" % (args.pixel, MANIFEST_VERSION))

    from src.data.common import load_split_metadata, iter_images
    df = load_split_metadata("main")
    rows = mark_smoke(rows_to_manifest(df, args.pixel), args.smoke_per_topic)
    assert len(rows) == 928, len(rows)  # 464 cases x {Q1, Q2}

    if not args.no_export:
        os.makedirs(IMG_DIR, exist_ok=True)
        want = {p["pair_id"] for p in rows}
        id2file = {r.ID: r.image_path.split("/")[-1] for r in df.itertuples()}
        n = 0
        for rid, im in iter_images("main", want):
            dst = os.path.join(IMG_DIR, id2file[rid])
            if not os.path.exists(dst):
                im.save(dst)
            n += 1
        print("exported %d images to %s" % (n, IMG_DIR))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for p in rows:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("wrote %d rows (%d smoke, %d cases) -> %s"
          % (len(rows), sum(p["smoke_subset"] for p in rows), len({p["template_id"] for p in rows}), out))
    print(Counter(p["domain"] for p in rows))


if __name__ == "__main__":
    main()
