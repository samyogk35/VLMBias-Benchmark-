# Full paired analysis, following docs/analysis_plan_v2.txt. Per category (illusions, boards):
# main test VCD vs regular (Holm across categories), the ablation family (APC-only and
# contrast-only vs regular), the secondary family, chance baselines, transitions, and the
# section 6 reading rules applied in order.
#
#   python -m src.analysis.analyze_full outputs/paired_full_v1/records.jsonl --out outputs/paired_full_v1/analysis.md

import argparse
import json
import os

import pandas as pd

from src.analysis.analyze_paired_mvp import compare, condition_table, pair_scores
from src.analysis.paired_stats import holm
from src.analysis.summarize import ROOT, load, md, pair_table, transitions

MANIFESTS = ("data/manifests/optical_pairs_v1.0.jsonl", "data/manifests/board_pairs_v1.0.jsonl")
CATEGORY = {"optical_illusion": "illusions", "game_boards": "boards"}
SAMPLED = ("regular", "apc_only", "contrast_only", "vcd")


def manifest_frame():
    rows = []
    for m in MANIFESTS:
        for l in open(os.path.join(ROOT, m)):
            r = json.loads(l)
            rows.append({"pair_id": r["pair_id"], "category": CATEGORY[r["domain"]],
                         "size_mismatch": bool(r.get("size_mismatch", False)),
                         "edit_direction": r.get("edit_direction", "n/a")})
    return pd.DataFrame(rows).set_index("pair_id")


def chance_baseline(pt, cond):
    # pair success expected if the answers on the two images were independent draws
    g = pt[pt.condition == cond].groupby("pair_id")
    return float((g.correct_canonical.mean() * g.correct_counterfactual.mean()).mean())


def same_answer_share(df, cond="greedy"):
    g = df[df.condition == cond].pivot_table(index="pair_id", columns="image_variant", values="parsed_answer", aggfunc="first")
    return float((g.canonical == g.counterfactual).mean())


def changed_answers(df, a, b):
    key = ["pair_id", "image_variant", "seed"]
    A = df[df.condition == a].set_index(key).parsed_answer
    B = df[df.condition == b].set_index(key).parsed_answer
    idx = A.index.intersection(B.index)
    return [i for i in idx if A[i] != B[i]], B


def greedy_match_share(df, changed, vcd_answers):
    greedy = df[df.condition == "greedy"].set_index(["pair_id", "image_variant"]).parsed_answer
    if not changed:
        return float("nan")
    return sum(vcd_answers[i] == greedy.get(i[:2]) for i in changed) / len(changed)


def error_types(df):
    # boards: wrong numeric answers split into familiar count / off by one / other
    rows = []
    for c, g in df[(df.is_correct == False) & (df.parse_status == "valid")].groupby("condition"):  # noqa: E712
        gt = pd.to_numeric(g.ground_truth, errors="coerce")
        pred = pd.to_numeric(g.parsed_answer, errors="coerce")
        fam = g.parsed_answer == g.expected_bias
        off1 = ~fam & ((pred - gt).abs() == 1)
        rows.append({"condition": c, "wrong": len(g), "familiar_count": int(fam.sum()),
                     "off_by_one": int(off1.sum()), "other": int((~fam & ~off1).sum())})
    return pd.DataFrame(rows)


def reading(r):
    # plan v2 section 6, first matching line wins
    if r["n_changed"] < 10 or r["invalid_gap"] > 0.05 or r["seed_sign_change"]:
        return "inconclusive (rule 1)"
    d_vcd, d_apc, d_con = r["dCF_vcd"], r["dCF_apc"], r["dCF_contrast"]
    if (d_vcd != 0 and d_apc * d_vcd > 0 and abs(d_apc) >= 0.5 * abs(d_vcd) and abs(d_con) < 0.5 * abs(d_vcd)) \
            or r["greedy_match"] >= 0.5:
        return "decoding artifact (rule 2)"
    if r["D_ci_low"] > 0 and r["tradeoff_share"] < 0.5 and r["excess_vcd"] > r["excess_regular"]:
        return "better grounding (rule 3)"
    if (r["dCF_vcd"] > 0 and r["dC_vcd"] < 0 and r["D_ci_low"] <= 0) or r["tradeoff_share"] >= 0.5:
        return "avoiding the familiar answer (rule 4)"
    return "inconclusive (rule 5)"


def fmt(x):
    return "%+.3f [%+.3f, %+.3f], p = %.4f" % (x["diff"], x["ci_low"], x["ci_high"], x["p"])


def analyze_category(name, df, lines):
    pt = pair_table(df)
    ps = pair_scores(pt)
    conds = set(df.condition)
    lines += ["## %s: %d pairs from %d templates, %d records" % (name, df.pair_id.nunique(), df.template_id.nunique(), len(df)), ""]

    main = compare(ps, "regular", "vcd", "pair")
    abl = [compare(ps, "regular", c, m) for c in ("apc_only", "contrast_only") if c in conds for m in ("CF", "pair")]
    sec = [compare(ps, "regular", "vcd", "C"), compare(ps, "regular", "vcd", "CF"), compare(ps, "greedy", "vcd", "pair")]
    for fam in (abl, sec):
        for row, p in zip(fam, holm([r["p"] for r in fam]) if fam else []):
            row["p_holm"] = p

    cond = condition_table(df, pt)
    cond["chance_pair_success"] = [chance_baseline(pt, c) if c in SAMPLED else float("nan") for c in cond.condition]
    cond["excess_over_chance"] = cond.pair_success - cond.chance_pair_success
    cond["familiar_rate_edited"] = [df[(df.condition == c) & (df.image_variant == "counterfactual")].bias.mean() for c in cond.condition]

    changed, vcd_ans = changed_answers(df, "regular", "vcd")
    trans = transitions(pt)
    tradeoff = trans["tradeoff_only"] / trans["CF_correction"] if trans["CF_correction"] else 0.0
    d_seed = [pt[(pt.condition == "vcd") & (pt.seed == s)].pair_success.mean() - pt[(pt.condition == "regular") & (pt.seed == s)].pair_success.mean()
              for s in sorted(pt[pt.condition == "vcd"].seed.unique())]
    inval = cond.set_index("condition").invalid_or_ambiguous
    ex = cond.set_index("condition").excess_over_chance

    def dcf(c):
        return compare(ps, "regular", c, "CF")["diff"] if c in conds else float("nan")

    r = {"n_changed": len(changed), "invalid_gap": abs(inval.get("vcd", 0) - inval.get("regular", 0)),
         "seed_sign_change": any(d > 0 for d in d_seed) and any(d < 0 for d in d_seed),
         "dCF_vcd": sec[1]["diff"], "dC_vcd": sec[0]["diff"], "dCF_apc": dcf("apc_only"), "dCF_contrast": dcf("contrast_only"),
         "greedy_match": greedy_match_share(df, changed, vcd_ans), "D_ci_low": main["ci_low"], "tradeoff_share": tradeoff,
         "excess_vcd": ex.get("vcd", float("nan")), "excess_regular": ex.get("regular", float("nan"))}
    verdict = reading(r)

    lines += ["Main test, pair score VCD - regular: **%s** (%s, %d templates)" % (fmt(main), main["p_method"], main["templates"]), "",
              "### Ablation family (Holm within)", "", md(pd.DataFrame(abl)[["comparison", "metric", "base", "other", "diff", "ci_low", "ci_high", "p", "p_holm"]]) if abl else "(no ablation records)", "",
              "### Secondary family (Holm within)", "", md(pd.DataFrame(sec)[["comparison", "metric", "base", "other", "diff", "ci_low", "ci_high", "p", "p_holm"]]), "",
              "### Each setting", "", md(cond), "",
              "Greedy gave the same answer to both images in %.1f%% of pairs." % (100 * same_answer_share(df)), "",
              "### Regular -> VCD transitions", "", "| metric | count |", "|---|---:|"] + ["| %s | %s |" % kv for kv in trans.items()]
    for c in ("apc_only", "contrast_only"):
        if c in conds:
            lines += ["", "### Regular -> %s transitions" % c, "", "| metric | count |", "|---|---:|"] + \
                     ["| %s | %s |" % kv for kv in transitions(pt, "regular", c).items()]
    lines += ["", "D per seed: %s" % ", ".join("%+.3f" % d for d in d_seed), "",
              "VCD changed %d answers vs regular; %.0f%% of them equal greedy's answer." % (len(changed), 100 * r["greedy_match"]), ""]

    by = ps.join(manifest_frame(), on="pair_id")
    for col in ("sub_domain", "edit_direction"):
        t = by.pivot_table(index=col, columns="condition", values="pair", aggfunc="mean")
        lines += ["### Pair score by %s" % col, "", t.round(3).to_markdown(), ""]
    if name == "boards":
        lines += ["### Error types on wrong answers", "", md(error_types(df)), ""]

    rob = []
    if by.size_mismatch.any():
        rob.append(("without size-mismatch pairs", pd.Index(by.pair_id[~by.size_mismatch].unique())))
    if (by.sub_domain == "go").any():
        rob.append(("without go", pd.Index(by.pair_id[by.sub_domain != "go"].unique())))
    for label, keep in rob:
        lines += ["Robustness preview, %s: %s" % (label, fmt(compare(ps, "regular", "vcd", "pair", keep=keep))), ""]

    lines += ["### Reading (plan v2 section 6)", "", "| input | value |", "|---|---|"] + \
             ["| %s | %s |" % (k, ("%.3f" % v) if isinstance(v, float) else v) for k, v in r.items()] + ["", "**%s: %s**" % (name, verdict), ""]
    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = load(args.records)
    df["category"] = df.domain.map(CATEGORY)

    lines = ["# Full paired results", "", "Source: %s, %d records. Plan: docs/analysis_plan_v2.txt." % (args.records, len(df)), ""]
    mains = {name: analyze_category(name, g, lines) for name, g in df.groupby("category")}
    adj = holm([m["p"] for m in mains.values()])
    lines += ["## Main tests, Holm across categories", "", "| category | D | p | p_holm |", "|---|---|---:|---:|"] + \
             ["| %s | %+.3f [%+.3f, %+.3f] | %.4f | %.4f |" % (n, m["diff"], m["ci_low"], m["ci_high"], m["p"], p) for (n, m), p in zip(mains.items(), adj)]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
