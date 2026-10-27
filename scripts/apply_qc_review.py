"""Merge an exported qc_review.json into the pair manifest and write a frozen version.

Every pair must have a verdict, and fail/unsure verdicts must have a note.
Pairs keep their record; qc_status becomes pass / fail / unsure and the
reviewer note is appended to qc_notes. Only `pass` pairs enter the primary sample.

Usage:
    python scripts/apply_qc_review.py --review ~/Downloads/qc_review.json \
        --manifest data/manifests/optical_pairs_v0.1.jsonl \
        --out data/manifests/optical_pairs_v1.0.jsonl --version v1.0
"""
import argparse
import collections
import json
import sys

# Changes besides the target element, seen in the diff overlays during review.
# They come from how the generator draws each illusion, so they hold for every
# pair of that type. Recorded so the report can list them as a limitation.
SIDE_CHANGES = {
    "Ebbinghaus": "surrounding circles pushed outward to fit the larger target; canvas width changes",
    "MullerLyer": "arrow fins move with the line ends they are attached to",
    "Poggendorff": "",
    "Ponzo": "",
    "VerticalHorizontal": "whole figure re-centered when the vertical line grows",
    "Zollner": "hatch marks move with the tilted lines they are attached to",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()

    reviews = json.load(open(args.review))["reviews"]
    pairs = [json.loads(l) for l in open(args.manifest)]

    problems = []
    for p in pairs:
        r = reviews.get(p["pair_id"], {})
        if r.get("verdict") not in ("pass", "fail", "unsure"):
            problems.append(f"{p['pair_id']}: no verdict")
        elif r["verdict"] != "pass" and not r.get("note", "").strip():
            problems.append(f"{p['pair_id']}: {r['verdict']} without a note")
    unknown = set(reviews) - {p["pair_id"] for p in pairs}
    problems += [f"{pid}: in review but not in manifest" for pid in sorted(unknown)]
    if problems:
        sys.exit("review incomplete:\n  " + "\n  ".join(problems))

    with open(args.out, "w") as f:
        for p in pairs:
            r = reviews[p["pair_id"]]
            p["qc_status"] = r["verdict"]
            note = r.get("note", "").strip()
            p["qc_notes"] = f"{p['qc_notes']}; hand review {r['at'][:10]}: {r['verdict']}" + (f" ({note})" if note else "")
            if p["sub_domain"] in SIDE_CHANGES:
                p["side_changes"] = SIDE_CHANGES[p["sub_domain"]]
            p["size_mismatch"] = p["canonical_size"] != p["counterfactual_size"]
            p["manifest_version"] = args.version
            f.write(json.dumps(p) + "\n")

    print(f"wrote {args.out}:", dict(collections.Counter(reviews[p['pair_id']]['verdict'] for p in pairs)))


if __name__ == "__main__":
    main()
