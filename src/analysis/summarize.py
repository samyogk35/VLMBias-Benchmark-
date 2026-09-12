"""Summarise a records.jsonl into the smoke-test results table (SPEC §8 metrics, descriptive only).

For each condition (and seed): canonical accuracy, counterfactual accuracy, exact pair success,
counterfactual bias rate, invalid rate. Plus regular-vs-VCD transition counts per seed
(CF correction/regression, C correction/regression, pair improvement, tradeoff-only).
No confidence intervals: this is a pipeline smoke test, not a finding.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, path))])
    # invalid/ambiguous parses are not correct, not bias; keep a separate flag
    df["valid"] = df.parse_status == "valid"
    df["correct"] = df.is_correct.fillna(False).astype(bool)
    df["bias"] = df.is_bias_answer.fillna(False).astype(bool)
    return df


def pair_table(df):
    """One row per (condition, seed, pair) with canonical/counterfactual correctness."""
    piv = df.pivot_table(index=["condition", "seed", "pair_id", "sub_domain", "template_id"], columns="image_variant",
                         values=["correct", "bias", "valid"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["pair_success"] = piv.correct_canonical & piv.correct_counterfactual
    return piv


def condition_summary(pt, by_seed=False):
    keys = ["condition", "seed"] if by_seed else ["condition"]
    rows = []
    for k, g in pt.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        n_pairs = g.pair_id.nunique()
        n = len(g)
        rows.append({
            **dict(zip(keys, k)), "pairs": n_pairs, "pair_runs": n,
            "canonical_acc": g.correct_canonical.mean(), "counterfactual_acc": g.correct_counterfactual.mean(),
            "balanced_acc": (g.correct_canonical.mean() + g.correct_counterfactual.mean()) / 2,
            "pair_success": g.pair_success.mean(),
            "cf_bias_rate": g.bias_counterfactual.mean(),
            "response_change_rate": (g.correct_canonical != g.correct_counterfactual).mean(),  # both-correct or both-wrong => same answer
            "any_invalid_pair_rate": 1 - (g.valid_canonical & g.valid_counterfactual).mean(),
        })
    return pd.DataFrame(rows)


def transitions(pt, a="regular", b="vcd"):
    """Per-seed transition counts between two conditions on the same pair & seed."""
    A = pt[pt.condition == a].set_index(["seed", "pair_id"])
    B = pt[pt.condition == b].set_index(["seed", "pair_id"])
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    out = collections.OrderedDict()
    out["n_pair_seed"] = len(idx)
    out["CF_correction"] = int((~A.correct_counterfactual & B.correct_counterfactual).sum())
    out["CF_regression"] = int((A.correct_counterfactual & ~B.correct_counterfactual).sum())
    out["C_correction"] = int((~A.correct_canonical & B.correct_canonical).sum())
    out["C_regression"] = int((A.correct_canonical & ~B.correct_canonical).sum())
    out["pair_improvement"] = int((~A.pair_success & B.pair_success).sum())
    out["pair_regression"] = int((A.pair_success & ~B.pair_success).sum())
    out["tradeoff_only"] = int(((~A.correct_counterfactual & B.correct_counterfactual) & (A.correct_canonical & ~B.correct_canonical)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = load(args.records)
    pt = pair_table(df)
    fmt = lambda d: d.to_markdown(index=False, floatfmt=".3f")

    lines = ["# Smoke-test results (NOT a finding)", "",
             f"Source: `{args.records}` — {len(df)} records, {df.pair_id.nunique()} pairs, seeds {sorted(df.seed.unique())}, "
             f"config_hash `{df.config_hash.iloc[0]}`, model rev `{df.model_revision.iloc[0][:8]}`.", "",
             "Optical-illusion pairs (Q1, 768 px). Canonical GT = Yes (genuinely equal), counterfactual GT = No.",
             "Invalid/ambiguous parses are counted as not-correct and reported separately (any_invalid_pair_rate; record-level invalid counts in the parser audit).", "",
             "## By condition (pooled over seeds)", "", fmt(condition_summary(pt)), "",
             "## By condition and seed", "", fmt(condition_summary(pt, by_seed=True)), "",
             "## By sub-domain (pooled over seeds)", ""]
    sub = []
    for (c, s), g in pt.groupby(["condition", "sub_domain"]):
        sub.append({"condition": c, "sub_domain": s, "pairs": g.pair_id.nunique(), "canonical_acc": g.correct_canonical.mean(),
                    "counterfactual_acc": g.correct_counterfactual.mean(), "pair_success": g.pair_success.mean()})
    lines += [fmt(pd.DataFrame(sub)), ""]
    lines += ["## Transitions regular -> vcd (same pair, same seed)", ""]
    tr = transitions(pt)
    lines += ["| metric | count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in tr.items()] + [""]
    lines += ["## Transitions greedy -> vcd (greedy has a single run; compared against each vcd seed)", ""]
    g = pt[pt.condition == "greedy"].copy()
    gg = pd.concat([g.assign(seed=s) for s in sorted(pt[pt.condition == "vcd"].seed.unique())])
    tr2 = transitions(pd.concat([gg, pt[pt.condition == "vcd"]]), "greedy", "vcd")
    lines += ["| metric | count |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in tr2.items()] + [""]
    # answer distribution
    lines += ["## Raw answer distribution", ""]
    ad = df.groupby(["condition", "image_variant", "parsed_answer"], dropna=False).size().unstack(fill_value=0)
    lines += [ad.to_markdown(), ""]
    lines += ["## Throughput", ""]
    th = df.groupby("condition").duration_ms.agg(["count", "mean", "median"]).round(0)
    lines += [th.to_markdown(), ""]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
