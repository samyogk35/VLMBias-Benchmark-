# Turn a records.jsonl from a vlmbias_main run into an accuracy-by-topic table next to the
# EnAR Table 1 numbers for LLaVA-1.5-7B. Single-variant records (image_variant=counterfactual),
# so there's no pair logic here, just accuracy, bias rate and invalid rate per topic.
#
# EnAR's denominators are 2 x cases per topic (Q1 + Q2 at one resolution), which is exactly what
# the manifest holds, so a full run is directly comparable. The dedup table keeps Q1 only,
# one row per case, since Q1 and Q2 on the same image are not independent.

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# EnAR (CVPR 2026) Table 1, LLaVA-v1.5-7B row, as (correct, denominator); the percentages in the
# paper are exact integer counts over these denominators (docs/enar_reproduction_notes.md)
ENAR_LLAVA = {
    "animals":          {"cases": 91,  "regular": 0,   "vcd": 2},
    "chess_pieces":     {"cases": 48,  "regular": 0,   "vcd": 0},
    "flags":            {"cases": 40,  "regular": 7,   "vcd": 6},
    "game_boards":      {"cases": 28,  "regular": 6,   "vcd": 3},
    "logos":            {"cases": 69,  "regular": 12,  "vcd": 19},
    "optical_illusion": {"cases": 132, "regular": 132, "vcd": 127},
    "patterned_grid":   {"cases": 56,  "regular": 0,   "vcd": 21},
}
TOPIC_ORDER = list(ENAR_LLAVA)


def load(path):
    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, path))])
    df["valid"] = df.parse_status == "valid"
    df["correct"] = df.is_correct.fillna(False).astype(bool)
    df["bias"] = df.is_bias_answer.fillna(False).astype(bool)
    return df


def topic_table(df, conditions):
    """One row per topic (+ total): n, accuracy per condition, EnAR accuracy per condition."""
    rows = []
    groups = [(t, df[df.domain == t]) for t in TOPIC_ORDER if (df.domain == t).any()]
    groups.append(("total", df))
    for topic, g in groups:
        row = {"topic": topic}
        for c in conditions:
            gc = g[g.condition == c]
            row["n_" + c] = len(gc)
            row["acc_" + c] = gc.correct.mean() * 100 if len(gc) else float("nan")
            row["bias_" + c] = gc.bias.mean() * 100 if len(gc) else float("nan")
            row["invalid_" + c] = (1 - gc.valid.mean()) * 100 if len(gc) else float("nan")
        for c in ("regular", "vcd"):
            if topic == "total":
                num = sum(v[c] for v in ENAR_LLAVA.values())
                den = sum(2 * v["cases"] for v in ENAR_LLAVA.values())
            else:
                num, den = ENAR_LLAVA[topic][c], 2 * ENAR_LLAVA[topic]["cases"]
            row["enar_" + c] = num / den * 100
        rows.append(row)
    return pd.DataFrame(rows)


def md(d):
    return d.to_markdown(index=False, floatfmt=".2f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = load(args.records)
    conditions = [c for c in ("greedy", "regular", "vcd") if (df.condition == c).any()]
    res = sorted(df.resolution.dropna().astype(int).unique().tolist())

    lines = ["# VLMBias main split, regular vs VCD (reproduction of EnAR Table 1, LLaVA-1.5-7B)", "",
             "Source: %s, %d records, %d rows, seeds %s, resolution %s, config_hash %s, model rev %s"
             % (args.records, len(df), df.pair_id.nunique(), sorted(df.seed.unique()), res,
                df.config_hash.iloc[0], df.model_revision.iloc[0][:8]), "",
             "Accuracy in %. `enar_*` = published numbers (16.92 regular / 19.18 VCD overall). EnAR's Regular",
             "row is greedy decoding (docs/enar_reproduction_notes.md), so compare `acc_greedy` to `enar_regular`;",
             "`acc_regular` is temperature-1 sampling, the VCD paper's own baseline. Invalid and ambiguous parses",
             "count as wrong in acc and are also listed on their own. bias = answered the familiar (wrong) answer.", ""]

    lines += ["## by topic, Q1 + Q2 (EnAR's denominator)", "", md(topic_table(df, conditions)), ""]

    q1 = df[df.question_form == "Q1"]
    if len(q1):
        lines += ["## by topic, Q1 only (one row per case, for statistics)", "", md(topic_table(q1, conditions)), ""]

    if "regular" in conditions and "vcd" in conditions:
        A = df[df.condition == "regular"].set_index(["seed", "pair_id"]).correct
        B = df[df.condition == "vcd"].set_index(["seed", "pair_id"]).correct
        idx = A.index.intersection(B.index)
        A, B = A.loc[idx], B.loc[idx]
        lines += ["## regular -> vcd, same row and seed", "", "| metric | count |", "|---|---:|",
                  "| n | %d |" % len(idx),
                  "| fixed (wrong -> right) | %d |" % int((~A & B).sum()),
                  "| broken (right -> wrong) | %d |" % int((A & ~B).sum()),
                  "| net | %+d |" % int(B.sum() - A.sum()), ""]

    lines += ["## parse status by condition", ""]
    lines += [df.groupby(["condition", "parse_status"]).size().unstack(fill_value=0).to_markdown(), ""]
    lines += ["## time per generation (ms)", ""]
    lines += [df.groupby("condition").duration_ms.agg(["count", "mean", "median"]).round(0).to_markdown(), ""]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(os.path.join(ROOT, args.out), "w").write(text)


if __name__ == "__main__":
    main()
