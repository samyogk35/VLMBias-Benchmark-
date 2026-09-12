# Turn a records.jsonl into the results table: accuracy on canonical / counterfactual images,
# pair success (both right), bias rate, and the regular -> vcd transition counts.
# No confidence intervals here, this is just for looking at smoke runs.

import argparse
import json
import os
from collections import OrderedDict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, path))])
    df["valid"] = df.parse_status == "valid"
    # invalid parses count as not correct but are also reported on their own
    df["correct"] = df.is_correct.fillna(False).astype(bool)
    df["bias"] = df.is_bias_answer.fillna(False).astype(bool)
    return df


def pair_table(df):
    # one row per (condition, seed, pair) with both images side by side
    piv = df.pivot_table(index=["condition", "seed", "pair_id", "sub_domain", "template_id"], columns="image_variant",
                         values=["correct", "bias", "valid"], aggfunc="first")
    piv.columns = ["%s_%s" % (a, b) for a, b in piv.columns]
    piv = piv.reset_index()
    piv["pair_success"] = piv.correct_canonical & piv.correct_counterfactual
    return piv


def condition_summary(pt, by_seed=False):
    keys = ["condition", "seed"] if by_seed else ["condition"]
    rows = []
    for k, g in pt.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row.update({
            "pairs": g.pair_id.nunique(), "pair_runs": len(g),
            "canonical_acc": g.correct_canonical.mean(),
            "counterfactual_acc": g.correct_counterfactual.mean(),
            "balanced_acc": (g.correct_canonical.mean() + g.correct_counterfactual.mean()) / 2,
            "pair_success": g.pair_success.mean(),
            "cf_bias_rate": g.bias_counterfactual.mean(),
            # answered the two images differently (both right or both wrong = same answer)
            "response_change_rate": (g.correct_canonical != g.correct_counterfactual).mean(),
            "any_invalid_pair_rate": 1 - (g.valid_canonical & g.valid_counterfactual).mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def transitions(pt, a="regular", b="vcd"):
    # what changed between condition a and b on the same pair and seed
    A = pt[pt.condition == a].set_index(["seed", "pair_id"])
    B = pt[pt.condition == b].set_index(["seed", "pair_id"])
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    cf_fixed = ~A.correct_counterfactual & B.correct_counterfactual
    c_broken = A.correct_canonical & ~B.correct_canonical
    out = OrderedDict()
    out["n_pair_seed"] = len(idx)
    out["CF_correction"] = int(cf_fixed.sum())
    out["CF_regression"] = int((A.correct_counterfactual & ~B.correct_counterfactual).sum())
    out["C_correction"] = int((~A.correct_canonical & B.correct_canonical).sum())
    out["C_regression"] = int(c_broken.sum())
    out["pair_improvement"] = int((~A.pair_success & B.pair_success).sum())
    out["pair_regression"] = int((A.pair_success & ~B.pair_success).sum())
    out["tradeoff_only"] = int((cf_fixed & c_broken).sum())
    return out


def md(d):
    return d.to_markdown(index=False, floatfmt=".3f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = load(args.records)
    pt = pair_table(df)

    lines = ["# Smoke test results (not a finding)", "",
             "Source: %s, %d records, %d pairs, seeds %s, config_hash %s, model rev %s"
             % (args.records, len(df), df.pair_id.nunique(), sorted(df.seed.unique()), df.config_hash.iloc[0], df.model_revision.iloc[0][:8]), "",
             "Optical illusion pairs, Q1, 768px. Canonical GT = Yes (really equal), counterfactual GT = No.",
             "Invalid/ambiguous parses count as wrong here and are also listed in any_invalid_pair_rate.", "",
             "## by condition (all seeds pooled)", "", md(condition_summary(pt)), "",
             "## by condition and seed", "", md(condition_summary(pt, by_seed=True)), "",
             "## by illusion type (all seeds pooled)", ""]
    sub = []
    for (c, s), g in pt.groupby(["condition", "sub_domain"]):
        sub.append({"condition": c, "sub_domain": s, "pairs": g.pair_id.nunique(),
                    "canonical_acc": g.correct_canonical.mean(), "counterfactual_acc": g.correct_counterfactual.mean(),
                    "pair_success": g.pair_success.mean()})
    lines += [md(pd.DataFrame(sub)), ""]

    lines += ["## regular -> vcd, same pair and seed", "", "| metric | count |", "|---|---:|"]
    lines += ["| %s | %s |" % kv for kv in transitions(pt).items()] + [""]

    lines += ["## greedy -> vcd (greedy is one run, compared against every vcd seed)", "", "| metric | count |", "|---|---:|"]
    g = pt[pt.condition == "greedy"]
    gg = pd.concat([g.assign(seed=s) for s in sorted(pt[pt.condition == "vcd"].seed.unique())])
    lines += ["| %s | %s |" % kv for kv in transitions(pd.concat([gg, pt[pt.condition == "vcd"]]), "greedy", "vcd").items()] + [""]

    lines += ["## answer counts", ""]
    lines += [df.groupby(["condition", "image_variant", "parsed_answer"], dropna=False).size().unstack(fill_value=0).to_markdown(), ""]
    lines += ["## time per generation (ms)", ""]
    lines += [df.groupby("condition").duration_ms.agg(["count", "mean", "median"]).round(0).to_markdown(), ""]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
