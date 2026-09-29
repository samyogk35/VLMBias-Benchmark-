# Re-parse the stored raw_output of a finished run with the current parser and write a new
# records file next to the old one (the original is never touched). The old parse fields are
# kept under *_v1 so before/after can be compared row by row. Also prints the before/after
# parse-status and accuracy tables.

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.eval.parse_answer import parse_answer, score
from src.analysis.summarize_benchmark import TOPIC_ORDER

PARSE_FIELDS = ("parsed_answer", "parse_status", "parse_method", "is_correct", "is_bias_answer")


def rescore_record(r, answer_type):
    out = dict(r)
    for k in PARSE_FIELDS:
        out[k + "_v1"] = r.get(k)
    pr = parse_answer(r["raw_output"], answer_type)
    is_correct, is_bias = score(pr, r["ground_truth"], r.get("expected_bias"))
    out.update(parsed_answer=pr.parsed_answer, parse_status=pr.parse_status, parse_method=pr.method,
               is_correct=is_correct, is_bias_answer=is_bias, answer_type=answer_type)
    return out


def status_table(df, col):
    t = df.groupby(["condition", col]).size().unstack(fill_value=0)
    return t.reindex(columns=["valid", "ambiguous", "invalid"], fill_value=0)


def acc_table(df, correct_col):
    ok = df[correct_col].eq(True)
    t = ok.groupby([df.domain, df.condition]).mean().unstack() * 100
    t = t.reindex([d for d in TOPIC_ORDER if d in t.index])
    t.loc["total"] = ok.groupby(df.condition).mean() * 100
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vlmbias_main_px768/records.jsonl")
    ap.add_argument("--manifest", default="data/manifests/vlmbias_main_px768_v0.1.jsonl")
    ap.add_argument("--out", default="outputs/vlmbias_main_px768/records_rescored_v2.jsonl")
    ap.add_argument("--tables", default="outputs/vlmbias_main_px768/rescore_v2_tables.md")
    args = ap.parse_args()

    if os.path.abspath(os.path.join(ROOT, args.out)) == os.path.abspath(os.path.join(ROOT, args.records)):
        sys.exit("refusing to overwrite the original records file")

    answer_type = {}
    for l in open(os.path.join(ROOT, args.manifest)):
        m = json.loads(l)
        answer_type[m["pair_id"]] = m["answer_type"]

    recs = [json.loads(l) for l in open(os.path.join(ROOT, args.records))]
    new = [rescore_record(r, answer_type[r["pair_id"]]) for r in recs]
    with open(os.path.join(ROOT, args.out), "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(new)
    changed = df[df.parse_status != df.parse_status_v1]
    before, after = status_table(df, "parse_status_v1"), status_table(df, "parse_status")
    acc_b, acc_a = acc_table(df, "is_correct_v1"), acc_table(df, "is_correct")
    acc = pd.concat({"before": acc_b, "after": acc_a}, axis=1).swaplevel(axis=1)
    acc = acc[[(c, w) for c in ("greedy", "regular", "vcd") for w in ("before", "after")]]
    acc.columns = ["%s %s" % c for c in acc.columns]

    moved = changed.groupby(["condition", "domain", "parse_status_v1", "parse_status"]).size()
    flips = int((df.is_correct.eq(True) != df.is_correct_v1.eq(True)).sum())

    lines = ["# Re-score with parser v2 (grid-cell labels no longer count as numbers)", "",
             "Source: %s -> %s, %d records" % (args.records, args.out, len(df)), "",
             "## parse status by condition, before (v1)", "", before.to_markdown(), "",
             "## parse status by condition, after (v2)", "", after.to_markdown(), "",
             "## rows whose parse status changed (%d)" % len(changed), "", moved.to_frame("n").to_markdown(), "",
             "## accuracy (%), before vs after; not-valid counts as wrong", "",
             acc.to_markdown(floatfmt=".2f"), "",
             "rows whose correctness changed: %d" % flips, ""]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(ROOT, args.tables), "w").write(text)


if __name__ == "__main__":
    main()
