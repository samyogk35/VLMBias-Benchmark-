# Which way does VCD move answers? For every row where VCD's parsed answer differs from
# regular sampling's (same row, same seed), label the change:
#   toward prior  VCD now gives expected_bias (the memorized answer)
#   away          regular gave expected_bias and VCD does not
#   other         neither answer is the prior, or one side is unparseable (parse status changed)
# In the VLMBias main split expected_bias is never the ground truth, so this shows the direction
# of VCD's push, not whether it trades familiar-image accuracy for counterfactual accuracy.
#
#   python src/analysis/vcd_direction.py

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.analysis.make_figures import BLUE, INK, INK2, ORANGE, PRETTY, _title, plt  # noqa: E402
from src.analysis.summarize_benchmark import TOPIC_ORDER  # noqa: E402

GRAY = "#8a8882"
LABELS = ["toward prior", "away from prior", "other"]


def _norm(x):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip().lower()


def change_direction(reg_answer, vcd_answer, bias):
    """'toward prior' / 'away from prior' / 'other', or None if the answer did not change."""
    reg, vcd, bias = _norm(reg_answer), _norm(vcd_answer), _norm(bias)
    if reg == vcd:
        return None
    if reg is None or vcd is None:
        return "other"
    if vcd == bias:
        return "toward prior"
    if reg == bias:
        return "away from prior"
    return "other"


def direction_frame(df):
    cols = ["pair_id", "domain", "parsed_answer", "expected_bias", "ground_truth"]
    r = df[df.condition == "regular"][cols].set_index("pair_id")
    v = df[df.condition == "vcd"][["pair_id", "parsed_answer"]].set_index("pair_id")
    m = r.join(v, rsuffix="_vcd", how="inner")
    m["direction"] = [change_direction(a, b, e) for a, b, e in zip(m.parsed_answer, m.parsed_answer_vcd, m.expected_bias)]
    m["vcd_correct"] = [_norm(b) == _norm(g) for b, g in zip(m.parsed_answer_vcd, m.ground_truth)]
    return m[m.direction.notna()]


def counts(m):
    t = pd.crosstab(m.domain, m.direction).reindex(columns=LABELS, fill_value=0)
    t = t.reindex([d for d in TOPIC_ORDER if d in t.index])
    t.loc["total"] = t.sum()
    t["changed"] = t.sum(axis=1)
    return t


def fig_direction(t, out):
    t = t.drop(index="total")
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    n, h = len(t), 0.24
    ys = list(range(n))[::-1]
    series = [("toward prior (VCD now gives the memorized answer)", ORANGE),
              ("away from prior (regular gave it, VCD does not)", BLUE),
              ("other (neither is the prior, or unparseable)", GRAY)]
    top = t[LABELS].to_numpy().max()
    for i, (name, color) in enumerate(series):
        vals = t[LABELS[i]].tolist()
        y = [yy + (1 - i) * (h + 0.03) for yy in ys]
        ax.barh(y, vals, height=h, color=color, label=name)
        for yy, v in zip(y, vals):
            ax.text(v + top * 0.01, yy, str(v), va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY[d] for d in t.index], color=INK)
    ax.set_xlim(0, top * 1.12)
    ax.set_xlabel("rows where VCD's answer differs from regular sampling's")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), frameon=False, fontsize=9)
    tot = t[LABELS].sum()
    _title(ax, "Which way does VCD move LLaVA-1.5-7B's answers?",
           "VLMBias main, 768 px, regular sampling vs VCD on the same row and seed (parser v2).\n"
           "All domains: %d toward the prior, %d away, %d other, of %d changed rows."
           % (tot["toward prior"], tot["away from prior"], tot["other"], tot.sum()))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vlmbias_main_px768/records_rescored_v2.jsonl")
    ap.add_argument("--out", default="outputs/vlmbias_main_px768/vcd_direction_v2.md")
    ap.add_argument("--fig", default="figures/png/vcd_change_direction.png")
    args = ap.parse_args()

    df = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, args.records))])
    m = direction_frame(df)
    t = counts(m)
    other = m[m.direction == "other"]
    away = m[m.direction == "away from prior"]
    lines = ["# Direction of VCD's changes (regular -> vcd, parser v2)", "",
             "Source: %s" % args.records, "", t.to_markdown(), "",
             "other, split: %d with an unparseable answer on one side, %d where both are numbers/yes-no but neither is the prior"
             % (int((other.parsed_answer.isna() | other.parsed_answer_vcd.isna()).sum()),
                int((other.parsed_answer.notna() & other.parsed_answer_vcd.notna()).sum())),
             "away from prior, split: %d where VCD's new answer is correct, %d where it is another wrong answer"
             % (int(away.vcd_correct.sum()), int((~away.vcd_correct).sum())),
             "other rows where VCD's new answer is correct: %d" % int(other.vcd_correct.sum()), ""]

    # does VCD's new answer look like greedy decoding? (bears on the APC / greedy-artifact question)
    greedy = df[df.condition == "greedy"].set_index("pair_id").parsed_answer.reindex(m.index)
    same = pd.Series([_norm(a) == _norm(b) for a, b in zip(m.parsed_answer_vcd, greedy)], index=m.index)
    prior_rate = df.is_bias_answer.eq(True).groupby(df.condition).mean() * 100
    lines += ["## VCD vs greedy", "",
              "changed rows where VCD's new answer equals greedy's answer on the same row:", ""]
    lines += ["- %s: %d of %d" % (d, int(same[m.direction == d].sum()), int((m.direction == d).sum())) for d in LABELS]
    lines += ["", "prior-answer rate (%% of all rows answering expected_bias): %s" %
              ", ".join("%s %.2f" % (c, prior_rate[c]) for c in ("greedy", "regular", "vcd")), ""]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(ROOT, args.out), "w").write(text)
    fig_direction(t, os.path.join(ROOT, args.fig))


if __name__ == "__main__":
    main()
