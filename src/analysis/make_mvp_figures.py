# Figures for the paired MVP (outputs/paired_mvp_v1). Same style as make_figures.py.
#
#   python src/analysis/make_mvp_figures.py --out outputs/paired_mvp_v1/figs
#
#   mvp_scores.png      normal-image accuracy, edited-image accuracy and pair success per setting
#   pair_check_ponzo.png  one frozen pair: normal, edited, and the changed pixels, with line lengths

import argparse
import json
import os

from src.analysis.make_figures import BLUE, GRID, INK, INK2, ORANGE, SURFACE, _title, plt
from src.analysis.summarize import ROOT, load, pair_table
from PIL import Image, ImageChops

SETTINGS = [("greedy", "greedy", BLUE, "////"), ("regular", "regular sampling", BLUE, None), ("vcd", "VCD", ORANGE, None)]


def fig_scores(records, out):
    df = load(records)
    pt = pair_table(df)
    panels = [("correct_canonical", "Normal image right (Yes)"), ("correct_counterfactual", "Edited image right (No)"),
              ("pair_success", "Both right (pair success)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, (col, name) in zip(axes, panels):
        for i, (cond, label, color, hatch) in enumerate(SETTINGS):
            v = 100 * pt[pt.condition == cond][col].mean()
            y = len(SETTINGS) - 1 - i
            ax.barh(y, v, height=0.62, color=SURFACE if hatch else color, edgecolor=color,
                    linewidth=1.0 if hatch else 0, hatch=hatch)
            ax.text(v + 1.5, y, "%.1f%%" % v, va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(SETTINGS)))
        ax.set_yticklabels([s[1] for s in SETTINGS][::-1], color=INK)
        ax.set_xlim(0, 100)
        ax.set_title(name, loc="left", fontsize=10, color=INK, fontweight="bold")
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[1].set_xlabel("% of answers (greedy: 1 run per image; sampling and VCD: 3 seeds)")
    fig.suptitle("LLaVA-1.5-7B on 60 frozen illusion pairs (22 templates), 840 answers",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "mvp_scores.png"))
    plt.close(fig)


def _line_lengths(path):
    # lengths of the two horizontal target lines: longest dark run in each band of rows
    im = Image.open(path).convert("L")
    w, h = im.size
    px = im.load()
    runs = []
    for y in range(h):
        best = cur = 0
        for x in range(w):
            cur = cur + 1 if px[x, y] < 235 else 0
            best = max(best, cur)
        runs.append(best)
    groups, cur = [], []
    for y, r in enumerate(runs):
        if r > 60:
            cur.append((r, y))
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return [max(g)[0] for g in groups]


def fig_pair_check(manifest, pair_id, out):
    pair = next(json.loads(l) for l in open(os.path.join(ROOT, manifest)) if json.loads(l)["pair_id"] == pair_id)
    a = Image.open(os.path.join(ROOT, pair["canonical_path"])).convert("RGB")
    b = Image.open(os.path.join(ROOT, pair["counterfactual_path"])).convert("RGB")
    mask = ImageChops.difference(a, b).convert("L").point(lambda v: 255 if v > 20 else 0)
    faded = Image.blend(a, Image.new("RGB", a.size, "white"), 0.7)
    diff = Image.composite(Image.new("RGB", a.size, (230, 0, 0)), faded, mask)
    la, lb = _line_lengths(os.path.join(ROOT, pair["canonical_path"])), _line_lengths(os.path.join(ROOT, pair["counterfactual_path"]))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
    caps = ["Normal: lines %d px and %d px\ncorrect answer Yes" % tuple(la),
            "Edited: lines %d px and %d px\ncorrect answer No" % tuple(lb),
            "Changed pixels (red)\nonly the ends of one line"]
    for ax, im, cap in zip(axes, (a, b, diff), caps):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
        ax.grid(False)
        ax.set_xlabel(cap, fontsize=9, color=INK2, linespacing=1.4)
    fig.suptitle('Pair check, Ponzo illusion: "%s"' % pair["prompt"].split(" Answer")[0],
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "pair_check_ponzo.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/paired_mvp_v1/records.jsonl")
    ap.add_argument("--manifest", default="data/manifests/optical_pairs_v1.0.jsonl")
    ap.add_argument("--pair", default="opt_Ponzo_strneg15_diffneg0p25")
    ap.add_argument("--out", default="outputs/paired_mvp_v1/figs")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    fig_scores(args.records, out)
    fig_pair_check(args.manifest, args.pair, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
