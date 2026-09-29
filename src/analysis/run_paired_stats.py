# Significance of the reproduced VCD gain on the vlmbias_main run: for each baseline
# (greedy, regular sampling) vs VCD, the accuracy difference with a case-clustered bootstrap
# CI and the exact McNemar p-value.
#
# Rows are matched on pair_id (greedy used seed 0, regular/VCD seed 42, so the seed can't be
# part of the key). Each case (template_id) has two rows, Q1 and Q2 on the same image, so the
# bootstrap resamples cases, and McNemar is also given on Q1 only, where rows are independent.
#
# primary: an unparseable answer (parsed_answer None) counts as wrong.
# sensitivity: drop every row that is unparseable in either condition.

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.analysis.paired_stats import cluster_bootstrap_diff, mcnemar_exact, transition_table


def paired_frame(df, base, other):
    cols = ["pair_id", "template_id", "domain", "question_form", "is_correct", "parsed_answer"]
    a = df[df.condition == base][cols].set_index("pair_id")
    b = df[df.condition == other][cols].set_index("pair_id")
    m = a.join(b[["is_correct", "parsed_answer"]], rsuffix="_b", how="inner")
    m["a"] = m.is_correct.eq(True)
    m["b"] = m.is_correct_b.eq(True)
    m["unparsed"] = m.parsed_answer.isna() | m.parsed_answer_b.isna()
    return m


def compare(m, n_boot, seed):
    t = transition_table(m.a, m.b)
    q1 = m[m.question_form == "Q1"]
    t1 = transition_table(q1.a, q1.b)
    bs = cluster_bootstrap_diff(m.a, m.b, m.template_id, n=n_boot, seed=seed)
    return {
        "rows": bs["n_rows"], "cases": bs["n_clusters"],
        "acc_base": m.a.mean() * 100, "acc_vcd": m.b.mean() * 100,
        "diff_pp": bs["estimate"] * 100, "ci_low": bs["ci_low"] * 100, "ci_high": bs["ci_high"] * 100,
        "fixed": t["fixed"], "broken": t["broken"],
        "p_mcnemar_all": mcnemar_exact(t["fixed"], t["broken"]),
        "fixed_q1": t1["fixed"], "broken_q1": t1["broken"],
        "p_mcnemar_q1": mcnemar_exact(t1["fixed"], t1["broken"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vlmbias_main_px768/records_rescored_v2.jsonl")
    ap.add_argument("--out", default="outputs/vlmbias_main_px768/paired_stats_v2.md")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, args.records))])
    rows = []
    for base in ("regular", "greedy"):
        m = paired_frame(df, base, "vcd")
        for analysis, sub in (("primary (None = wrong)", m), ("sensitivity (drop None)", m[~m.unparsed])):
            rows.append({"comparison": "%s -> vcd" % base, "analysis": analysis, **compare(sub, args.n_boot, args.seed)})
    t = pd.DataFrame(rows)

    lines = ["# Paired significance of the VCD gain (parser v2)", "",
             "Source: %s. Bootstrap: %d resamples of cases (template_id), seed %d, 95%% percentile CI."
             % (args.records, args.n_boot, args.seed),
             "diff_pp = VCD accuracy minus baseline accuracy, in percentage points. p_mcnemar_all uses all",
             "rows (treats Q1 and Q2 as independent); p_mcnemar_q1 uses Q1 only (one row per case).", "",
             t.to_markdown(index=False, floatfmt=".3f"), ""]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
