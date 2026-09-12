# Check the parser against real model output: list every distinct raw string and what it
# parsed to, count valid/ambiguous/invalid per condition, and dump anything non-valid so I
# can look at it by hand.

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.eval.parse_answer import parse_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = []
    for f in args.records:
        recs += [json.loads(l) for l in open(os.path.join(ROOT, f))]

    lines = ["# Parser audit", "", "%d records from %s" % (len(recs), ", ".join(args.records)), "",
             "## distinct raw outputs (re-parsed with the current parser)", "",
             "| n | raw_output | parsed | status | method |", "|---:|---|---|---|---|"]
    for raw, n in Counter(r["raw_output"] for r in recs).most_common():
        pr = parse_answer(raw, "yes_no")
        lines.append("| %d | `%r` | %s | %s | %s |" % (n, raw, pr.parsed_answer, pr.parse_status, pr.method))

    lines += ["", "## parse status by condition", "", "| condition | valid | ambiguous | invalid | not-valid rate |", "|---|---:|---:|---:|---:|"]
    by = defaultdict(Counter)
    for r in recs:
        by[r["condition"]][r["parse_status"]] += 1
    for cond, c in sorted(by.items()):
        tot = sum(c.values())
        lines.append("| %s | %d | %d | %d | %.1f%% |" % (cond, c["valid"], c["ambiguous"], c["invalid"], 100 * (c["ambiguous"] + c["invalid"]) / tot))

    lines += ["", "records that hit max_new_tokens: %d" % sum(1 for r in recs if r.get("hit_max_new_tokens"))]

    bad = [r for r in recs if r["parse_status"] != "valid"]
    lines += ["", "## non-valid records to check by hand (%d)" % len(bad), ""]
    for r in bad:
        lines.append("- %s %s %s %s seed=%s: `%r` -> %s" % (r["run_id"], r["pair_id"], r["image_variant"], r["condition"], r["seed"], r["raw_output"], r["parse_status"]))

    mismatch = sum(1 for r in recs if parse_answer(r["raw_output"], "yes_no").parsed_answer != r["parsed_answer"])
    lines += ["", "stored vs re-parsed mismatches: %d" % mismatch]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
