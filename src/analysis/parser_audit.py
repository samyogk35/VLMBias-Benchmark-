"""Parser audit over real model outputs (SPEC §7).

Prints (and writes) a table of distinct raw outputs with their parse result, counts by
parse_status x condition, and dumps every ambiguous/invalid record for hand review.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.eval.parse_answer import parse_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="records.jsonl files")
    ap.add_argument("--out", default=None, help="markdown output path")
    args = ap.parse_args()

    recs = []
    for f in args.records:
        recs += [json.loads(l) for l in open(os.path.join(ROOT, f))]

    # 1. distinct raw outputs -> parse result (re-parsed live so the table reflects current parser code)
    distinct = collections.Counter(r["raw_output"] for r in recs)
    lines = [f"# Parser audit", "", f"Records: {len(recs)} from {', '.join(args.records)}", "",
             "## Distinct raw outputs", "", "| n | raw_output | parsed | status | method |", "|---:|---|---|---|---|"]
    for raw, n in distinct.most_common():
        pr = parse_answer(raw, "yes_no")
        lines.append(f"| {n} | `{raw!r}` | {pr.parsed_answer} | {pr.parse_status} | {pr.method} |")

    # 2. status by condition
    lines += ["", "## Parse status by condition", "", "| condition | valid | ambiguous | invalid | invalid+ambiguous rate |", "|---|---:|---:|---:|---:|"]
    by = collections.defaultdict(collections.Counter)
    for r in recs:
        by[r["condition"]][r["parse_status"]] += 1
    for cond, c in sorted(by.items()):
        tot = sum(c.values())
        lines.append(f"| {cond} | {c['valid']} | {c['ambiguous']} | {c['invalid']} | {(c['ambiguous'] + c['invalid']) / tot:.1%} |")

    # 3. hit_max_new_tokens (truncation) check
    trunc = [r for r in recs if r.get("hit_max_new_tokens")]
    lines += ["", f"Records that hit max_new_tokens: {len(trunc)}"]

    # 4. all non-valid for hand review
    bad = [r for r in recs if r["parse_status"] != "valid"]
    lines += ["", f"## Non-valid records for hand review ({len(bad)})", ""]
    for r in bad:
        lines.append(f"- `{r['run_id']}` {r['pair_id']} {r['image_variant']} {r['condition']} seed={r['seed']}: `{r['raw_output']!r}` -> {r['parse_status']}")

    # 5. stored parse vs live re-parse consistency
    mismatch = [r for r in recs if parse_answer(r["raw_output"], "yes_no").parsed_answer != r["parsed_answer"]]
    lines += ["", f"Stored-vs-reparsed mismatches: {len(mismatch)}"]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
