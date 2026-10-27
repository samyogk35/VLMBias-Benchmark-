# Paired MVP analysis, following docs/analysis_plan_v1.txt section by section.
# Main test: pair score (mean pair success over seeds), VCD minus regular sampling, with a
# cluster bootstrap CI and a sign-flip p-value over templates (pairs that share a normal image).
#
#   python src/analysis/analyze_paired_mvp.py outputs/paired_mvp_v1/records.jsonl \
#       --manifest data/manifests/optical_pairs_v1.0.jsonl --out outputs/paired_mvp_v1/analysis.md

import argparse
import json
import os

import pandas as pd

from src.analysis.paired_stats import (cluster_bootstrap_diff, cluster_signflip_test, holm, mcnemar_exact,
                                       transition_table)
from src.analysis.summarize import ROOT, load, md, pair_table, transitions

N_BOOT = 10_000


def pair_scores(pt):
    # one row per (condition, pair): means over seeds of pair success and of each image being right
    return (pt.groupby(["condition", "pair_id", "template_id", "sub_domain"], as_index=False)
              .agg(pair=("pair_success", "mean"), C=("correct_canonical", "mean"), CF=("correct_counterfactual", "mean")))


def compare(ps, base, other, metric, keep=None):
    a = ps[ps.condition == base].set_index("pair_id")
    b = ps[ps.condition == other].set_index("pair_id")
    idx = a.index.intersection(b.index)
    if keep is not None:
        idx = idx[idx.isin(keep)]
    a, b = a.loc[idx], b.loc[idx]
    boot = cluster_bootstrap_diff(a[metric], b[metric], a.template_id, n=N_BOOT, seed=0)
    flip = cluster_signflip_test(a[metric], b[metric], a.template_id, n=N_BOOT, seed=0)
    return {"comparison": "%s - %s" % (other, base), "metric": metric, "pairs": len(idx),
            "templates": boot["n_clusters"], "base": a[metric].mean(), "other": b[metric].mean(),
            "diff": boot["estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "p": flip["p_value"], "p_method": flip["method"]}


def condition_table(df, pt):
    rows = []
    for c, g in df.groupby("condition"):
        p = pt[pt.condition == c]
        ans = g.parsed_answer.fillna("<none>")
        top_share = ans.value_counts(normalize=True).iloc[0]
        rows.append({
            "condition": c, "records": len(g), "seeds": g.seed.nunique(),
            "normal_acc": p.correct_canonical.mean(), "edited_acc": p.correct_counterfactual.mean(),
            "pair_success": p.pair_success.mean(),
            "yes_rate_normal": (g[g.image_variant == "canonical"].parsed_answer == "Yes").mean(),
            "yes_rate_edited": (g[g.image_variant == "counterfactual"].parsed_answer == "Yes").mean(),
            "invalid_or_ambiguous": 1 - g.valid.mean(),
            "most_common_answer": ans.value_counts().index[0], "its_share": top_share,
            "fixed_answer_flag": top_share >= 0.9,
        })
    return pd.DataFrame(rows)


def mcnemar_by_seed(pt, base="regular", other="vcd"):
    rows = []
    for seed in sorted(pt[pt.condition == other].seed.unique()):
        a = pt[(pt.condition == base) & (pt.seed == seed)].set_index("pair_id")
        b = pt[(pt.condition == other) & (pt.seed == seed)].set_index("pair_id").loc[a.index]
        for col, name in (("correct_canonical", "normal"), ("correct_counterfactual", "edited")):
            t = transition_table(a[col], b[col])
            rows.append({"seed": seed, "image": name, "fixed": t["fixed"], "broken": t["broken"],
                         "p_exact": mcnemar_exact(t["fixed"], t["broken"])})
    return pd.DataFrame(rows)


def greedy_match_of_vcd_changes(df):
    # of the answers VCD changed relative to regular sampling (same pair, image, seed), how many
    # land on greedy's answer for that image
    key = ["pair_id", "image_variant"]
    reg = df[df.condition == "regular"].set_index(key + ["seed"]).parsed_answer
    vcd = df[df.condition == "vcd"].set_index(key + ["seed"]).parsed_answer
    greedy = df[df.condition == "greedy"].set_index(key).parsed_answer
    idx = reg.index.intersection(vcd.index)
    changed = [i for i in idx if reg[i] != vcd[i]]
    match = sum(vcd[i] == greedy.get(i[:2]) for i in changed)
    return len(changed), match


def edit_size_level(manifest):
    # rank of |counterfactual_difference| within each illusion type: 1 = smallest edit
    m = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, manifest))])
    m["size"] = m.generator_params.map(lambda g: abs(g["counterfactual_difference"]))
    m["edit_level"] = m.groupby("sub_domain")["size"].rank(method="dense").astype(int)
    return m.set_index("pair_id")[["edit_level", "size", "size_mismatch"]]


def reading(primary, sec, trans, by_type_d, d_by_seed, cond):
    r, dC, dCF = primary, sec["C"], sec["CF"]
    tradeoff_share = trans["tradeoff_only"] / trans["CF_correction"] if trans["CF_correction"] else float("nan")
    inval = cond.set_index("condition").invalid_or_ambiguous
    checks = {
        "D interval above zero": r["ci_low"] > 0,
        "trade-off-only < half of edited fixes": tradeoff_share < 0.5,
        "D positive in >= 4 of 6 types": int((by_type_d > 0).sum()) >= 4,
        "dCF > 0": dCF["diff"] > 0,
        "dC < 0": dC["diff"] < 0,
        "D interval includes or below zero": r["ci_low"] <= 0,
        "trade-off-only >= half of edited fixes": tradeoff_share >= 0.5,
        "invalid rate differs > 5 pts (regular vs vcd)": abs(inval.get("vcd", 0) - inval.get("regular", 0)) > 0.05,
        "D changes sign between seeds": (d_by_seed > 0).any() and (d_by_seed < 0).any(),
    }
    grounding = checks["D interval above zero"] and checks["trade-off-only < half of edited fixes"] and checks["D positive in >= 4 of 6 types"]
    avoid = (checks["dCF > 0"] and checks["dC < 0"] and checks["D interval includes or below zero"]) or checks["trade-off-only >= half of edited fixes"]
    if checks["invalid rate differs > 5 pts (regular vs vcd)"] or checks["D changes sign between seeds"]:
        verdict = "inconclusive"
    elif grounding:
        verdict = "better grounding"
    elif avoid:
        verdict = "avoiding the familiar answer"
    else:
        verdict = "inconclusive (decoding-artifact reading waits for the APC-only / contrast-only run)"
    return checks, tradeoff_share, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--manifest", default="data/manifests/optical_pairs_v1.0.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load(args.records)
    pt = pair_table(df)
    ps = pair_scores(pt)
    levels = edit_size_level(args.manifest)
    cond = condition_table(df, pt)

    primary = compare(ps, "regular", "vcd", "pair")
    sec_rows = [compare(ps, "regular", "vcd", "C"), compare(ps, "regular", "vcd", "CF"), compare(ps, "greedy", "vcd", "pair")]
    for row, p_adj in zip(sec_rows, holm([r["p"] for r in sec_rows])):
        row["p_holm"] = p_adj
    sec = {"C": sec_rows[0], "CF": sec_rows[1]}

    trans = transitions(pt)
    d_by_seed = pd.Series({int(s): pt[(pt.condition == "vcd") & (pt.seed == s)].pair_success.mean()
                           - pt[(pt.condition == "regular") & (pt.seed == s)].pair_success.mean()
                           for s in sorted(pt[pt.condition == "vcd"].seed.unique())})

    wide = ps.pivot_table(index=["pair_id", "sub_domain"], columns="condition", values=["pair", "C", "CF"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide = wide.join(levels, on="pair_id")
    by_type = wide.groupby("sub_domain").agg(pairs=("pair_id", "count"), **{
        "%s_%s" % (m, c): ("%s_%s" % (m, c), "mean") for m in ("C", "CF", "pair") for c in ("greedy", "regular", "vcd")})
    by_type["D"] = by_type.pair_vcd - by_type.pair_regular
    by_size = wide.groupby("edit_level").agg(pairs=("pair_id", "count"), **{
        "%s_%s" % (m, c): ("%s_%s" % (m, c), "mean") for m in ("C", "CF", "pair") for c in ("greedy", "regular", "vcd")})

    keep = levels.index[~levels.size_mismatch]
    robust = compare(ps, "regular", "vcd", "pair", keep=keep)
    n_changed, n_match = greedy_match_of_vcd_changes(df)
    checks, tradeoff_share, verdict = reading(primary, sec, trans, by_type.D, d_by_seed, cond)

    f = lambda x: "%+.3f [%+.3f, %+.3f], p = %.4f" % (x["diff"], x["ci_low"], x["ci_high"], x["p"])
    lines = [
        "# Paired MVP results", "",
        "Source: %s (%d records, %d pairs from %d templates). Plan: docs/analysis_plan_v1.txt." % (
            args.records, len(df), df.pair_id.nunique(), df.template_id.nunique()),
        "Normal image correct answer = Yes, edited image correct answer = No.", "",
        "## Main test (plan section 5)", "",
        "Pair score VCD - regular: **%s** (%s, %d templates)" % (f(primary), primary["p_method"], primary["templates"]),
        "Regular %.3f, VCD %.3f." % (primary["base"], primary["other"]), "",
        "## Secondary tests (Holm across these three)", "",
        md(pd.DataFrame(sec_rows)[["comparison", "metric", "base", "other", "diff", "ci_low", "ci_high", "p", "p_holm"]]), "",
        "## Each condition", "", md(cond), "",
        "## Regular -> VCD transitions (same pair and seed)", "",
        "| metric | count |", "|---|---:|"] + ["| %s | %s |" % kv for kv in trans.items()] + [
        "", "Trade-off-only share of edited-image fixes: %.2f" % tradeoff_share, "",
        "## McNemar per seed, regular vs VCD (descriptive)", "", md(mcnemar_by_seed(pt)), "",
        "## D per seed", "", md(d_by_seed.rename("D").reset_index().rename(columns={"index": "seed"})), "",
        "## By illusion type (means of pair-level scores)", "", by_type.round(3).to_markdown(), "",
        "## By edit size (1 = smallest edit within its type)", "", by_size.round(3).to_markdown(), "",
        "## VCD changes that match greedy", "",
        "VCD changed %d answers relative to regular sampling (same image and seed); %d of them equal greedy's answer." % (n_changed, n_match), "",
        "## Robustness preview: without the 12 size-mismatch pairs", "", "Pair score VCD - regular: %s (%d pairs)" % (f(robust), robust["pairs"]), "",
        "## Reading rule (plan section 6)", "",
        "| check | holds |", "|---|---|"] + ["| %s | %s |" % (k, v) for k, v in checks.items()] + [
        "", "**Reading: %s.** The decoding-artifact reading needs the APC-only and contrast-only run." % verdict, ""]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
