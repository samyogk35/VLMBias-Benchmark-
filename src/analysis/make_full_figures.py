# Figures for the full five-setting run (outputs/paired_full_v1). Same style as make_figures.py.
#
#   python -m src.analysis.make_full_figures --out outputs/paired_full_v1/figs
#
#   full_scores.png   normal accuracy, edited accuracy and pair success per setting, illusions and boards
#   boards_nine.png   a standard board and its edit with greedy's answers, and the share of "9" answers

import argparse
import os

from PIL import Image

from src.analysis.make_figures import BLUE, GRID, INK, INK2, ORANGE, SURFACE, plt
from src.analysis.summarize import ROOT, load, pair_table

# blue = no noisy image, orange = uses the noisy image; hatching marks greedy and the two ablations
SETTINGS = [("greedy", "greedy", BLUE, "////"), ("regular", "regular sampling", BLUE, None),
            ("apc_only", "APC filter only", ORANGE, "\\\\\\\\"), ("contrast_only", "contrast only", ORANGE, "...."),
            ("vcd", "full VCD", ORANGE, None)]
CATEGORY = {"optical_illusion": "Illusions (60 pairs, 22 templates)", "game_boards": "Boards (20 pairs, 6 templates)"}


def _bars(ax, values, xmax=100):
    n = len(SETTINGS)
    for i, ((cond, label, color, hatch), v) in enumerate(zip(SETTINGS, values)):
        y = n - 1 - i
        ax.barh(y, v, height=0.62, color=SURFACE if hatch else color, edgecolor=color,
                linewidth=1.0 if hatch else 0, hatch=hatch)
        ax.text(v + 1.5, y, "%.1f%%" % v, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[1] for s in SETTINGS][::-1], color=INK)
    ax.set_xlim(0, xmax)


def fig_scores(df, out):
    pt = pair_table(df)
    panels = [("correct_canonical", "Normal image right"), ("correct_counterfactual", "Edited image right"),
              ("pair_success", "Both right (pair success)")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharex=True)
    for row, dom in enumerate(("optical_illusion", "game_boards")):
        p = pt[pt.pair_id.isin(df[df.domain == dom].pair_id.unique())]
        for col, (metric, name) in enumerate(panels):
            ax = axes[row, col]
            _bars(ax, [100 * p[p.condition == c][metric].mean() for c, *_ in SETTINGS])
            ax.set_title(name if row == 0 else "", loc="left", fontsize=10, color=INK, fontweight="bold")
            if col:
                ax.tick_params(labelleft=False)
        axes[row, 0].set_ylabel(CATEGORY[dom], fontsize=9.5, color=INK, labelpad=10)
    axes[1, 1].set_xlabel("% of answers (greedy: 1 run per image; other settings: 3 seeds)")
    fig.suptitle("LLaVA-1.5-7B under five decoding settings, 2,080 answers",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "full_scores.png"))
    plt.close(fig)


def fig_boards(df, out):
    b = df[df.domain == "game_boards"]
    fig = plt.figure(figsize=(11.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.75)
    for i, (path, cap) in enumerate((
            ("data/images/boards/chess_standard_px768.png", "Normal: 8 rows\ngreedy answers 9"),
            ("data/images/boards/chess_remove_first_row_px768.png", "Edited: 7 rows\ngreedy answers 9"))):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(Image.open(os.path.join(ROOT, path)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
        ax.set_xlabel(cap, fontsize=9, color=INK2, linespacing=1.4)
    ax = fig.add_subplot(gs[0, 2])
    _bars(ax, [100 * (b[b.condition == c].parsed_answer == "9").mean() for c, *_ in SETTINGS], xmax=115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title('Board answers that are "9"', loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("% of answers on all 40 board images")
    fig.suptitle('Prompt: "How many rows are there on this board? Answer with a number in curly brackets, e.g., {9}."',
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(os.path.join(out, "boards_nine.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/paired_full_v1/records.jsonl")
    ap.add_argument("--out", default="outputs/paired_full_v1/figs")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    df = load(args.records)
    fig_scores(df, out)
    fig_boards(df, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
