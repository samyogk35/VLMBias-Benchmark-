# Render the report figures (PNG) from a vlmbias_main records.jsonl plus the optical smoke run.
# Numbers come from the same functions summarize_benchmark.py uses for the markdown tables, so
# the PNGs and the tables cannot disagree. Runs on CPU, no torch.
#
#   python src/analysis/make_figures.py                      # defaults below
#   python src/analysis/make_figures.py --out figures/png
#
# Figures:
#   acc_by_domain.png          ours (greedy / vcd) vs EnAR Table 1 (regular / vcd), per topic + total
#   transitions_by_domain.png  regular -> vcd on the same row and seed: fixed vs broken, per topic
#   timing.png                 median ms per generation per condition
#   pair_ebbinghaus.png        one canonical / counterfactual pair with the three answers under each
#   examples_fixed_broken.png  2 rows VCD fixed, 2 rows VCD broke: image, question, regular vs vcd

import argparse
import json
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.analysis.summarize_benchmark import TOPIC_ORDER, load, topic_table  # noqa: E402

# palette: blue = regular/greedy decoding, orange = VCD; hatched = published EnAR number.
# fixed/broken use the status pair (green / red). Text never wears a series color.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#008300", "#e34948"
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 1,
    "axes.axisbelow": True, "hatch.linewidth": 0.8, "savefig.dpi": 200,
})

PRETTY = {"animals": "animals", "chess_pieces": "chess pieces", "flags": "flags", "game_boards": "game boards",
          "logos": "logos", "optical_illusion": "optical illusions", "patterned_grid": "patterned grids",
          "total": "TOTAL (928 rows)"}


def _title(ax, title, subtitle=None):
    n_sub = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=8 + 13 * n_sub)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom", linespacing=1.4)


def fig_acc_by_domain(df, out):
    t = topic_table(df, ["greedy", "vcd"])
    labels = [PRETTY[x] for x in t.topic]
    series = [("greedy (ours)", t.acc_greedy, BLUE, None), ("EnAR \"Regular\" (published)", t.enar_regular, BLUE, "////"),
              ("VCD (ours)", t.acc_vcd, ORANGE, None), ("EnAR VCD (published)", t.enar_vcd, ORANGE, "////")]
    n, h = len(t), 0.19
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ys = list(range(n))[::-1]
    for i, (name, vals, color, hatch) in enumerate(series):
        y = [yy + (1.5 - i) * (h + 0.02) for yy in ys]
        ax.barh(y, vals, height=h, color=color if hatch is None else SURFACE, edgecolor=color,
                linewidth=1.0 if hatch else 0, hatch=hatch, label=name)
        for yy, v in zip(y, vals):
            ax.text(v + 0.7, yy, "%.2f" % v, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, 62)
    ax.set_xlabel("accuracy (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _title(ax, "LLaVA-1.5-7B on VLMBias main split: our reproduction vs EnAR Table 1",
           "928 rows (464 cases x Q1/Q2) at 768 px, one seed. Solid = ours, hatched = published.\n"
           "Greedy matches EnAR exactly on 5 of 7 topics.")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def transitions(df):
    """regular -> vcd on the same (seed, pair_id): fixed / broken counts per topic and total."""
    A = df[df.condition == "regular"].set_index(["seed", "pair_id"])
    B = df[df.condition == "vcd"].set_index(["seed", "pair_id"])
    idx = A.index.intersection(B.index)
    a, b, dom = A.loc[idx].correct, B.loc[idx].correct, A.loc[idx].domain
    rows = []
    for t in TOPIC_ORDER + ["total"]:
        m = (dom == t) if t != "total" else (dom == dom)
        rows.append((t, int((~a[m] & b[m]).sum()), int((a[m] & ~b[m]).sum())))
    return rows


def fig_transitions(df, out):
    rows = transitions(df)
    labels = [PRETTY[t] for t, _, _ in rows]
    fixed = [f for _, f, _ in rows]
    broken = [-b for _, _, b in rows]
    ys = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(ys, fixed, height=0.55, color=GREEN, label="fixed (wrong -> right)")
    ax.barh(ys, broken, height=0.55, color=RED, label="broken (right -> wrong)")
    for y, f, b in zip(ys, fixed, broken):
        if f:
            ax.text(f + 0.3, y, str(f), va="center", fontsize=9, color=INK2)
        if b:
            ax.text(b - 0.3, y, str(-b), va="center", ha="right", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK)
    lim = max(max(fixed), -min(broken)) + 4
    ax.set_xlim(-lim, lim)
    ticks = [-x for x in range(5, int(lim) + 1, 5)][::-1] + list(range(0, int(lim) + 1, 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(x)) for x in ticks])
    ax.set_xlabel("rows whose correctness changed, regular sampling -> VCD (same image, prompt and seed)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    tot = rows[-1]
    _title(ax, "What VCD changes: %d answers fixed, %d broken, net %+d of 928" % (tot[1], tot[2], tot[1] - tot[2]),
           "Same-seed paired comparison. The other %d rows are unchanged in correctness." % (928 - tot[1] - tot[2]))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_timing(df, out):
    med = df.groupby("condition").duration_ms.median()
    order = [c for c in ("greedy", "regular", "vcd") if c in med.index]
    names = {"greedy": "greedy", "regular": "regular sampling", "vcd": "VCD"}
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ys = list(range(len(order)))[::-1]
    ax.barh(ys, [med[c] for c in order], height=0.5, color=[ORANGE if c == "vcd" else BLUE for c in order])
    for y, c in zip(ys, order):
        ax.text(med[c] + 8, y, "%d ms" % med[c], va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[c] for c in order], color=INK)
    ax.set_xlim(0, max(med) * 1.25)
    ax.set_xlabel("median time per generation (ms), max_new_tokens=16, RTX A5000")
    _title(ax, "VCD costs one extra forward pass per token: ~2x wall time",
           "%d generations per condition. Full 928 x 3 run = about 20 min." % int(df.groupby('condition').size().iloc[0]))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------- image + text composites (PIL) ----------

def _font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def _fit(img, w, h=None):
    r = w / img.width
    if h is not None:
        r = min(r, h / img.height)
    return img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.LANCZOS)


def _wrap_lines(draw, x, y, text, font, width, fill, gap=4):
    for line in textwrap.wrap(text, width=width) or [""]:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + gap
    return y


def _answer_line(draw, x, y, label, raw, correct, font):
    mark = "correct" if correct else "wrong"
    color = GREEN if correct else RED
    draw.rectangle([x, y + 3, x + 10, y + 13], fill=color)
    draw.text((x + 16, y), "%s: %r  (%s)" % (label, raw, mark), font=font, fill=INK)
    return y + font.size + 6


def fig_pair(smoke_records, pair_id, out, seed=42):
    """Canonical and counterfactual member side by side with greedy / regular / vcd answers."""
    recs = [json.loads(l) for l in open(os.path.join(ROOT, smoke_records))]
    recs = [r for r in recs if r["pair_id"] == pair_id and (r["seed"] == seed or r["condition"] == "greedy")]
    W, PAD = 420, 20
    f_h, f_t, f_s = _font(15, True), _font(12), _font(11)
    cols = []
    for variant in ("canonical", "counterfactual"):
        rs = {r["condition"]: r for r in recs if r["image_variant"] == variant}
        any_r = next(iter(rs.values()))
        img = _fit(Image.open(os.path.join(ROOT, any_r["image_path"])).convert("RGB"), W)
        cols.append((variant, rs, img))
    img_h = max(im.height for _, _, im in cols)
    H = PAD + 22 + 8 + img_h + 14 + 5 * 18 + 3 * 20 + PAD
    canvas = Image.new("RGB", (2 * W + 3 * PAD, H), SURFACE)
    d = ImageDraw.Draw(canvas)
    for i, (variant, rs, im) in enumerate(cols):
        x = PAD + i * (W + PAD)
        gt = next(iter(rs.values()))["ground_truth"]
        d.text((x, PAD), "%s   (correct answer: %s)" % (variant.upper(), gt), font=f_h, fill=INK)
        canvas.paste(im, (x, PAD + 30))
        y = PAD + 30 + img_h + 12
        y = _wrap_lines(d, x, y, "Q: " + next(iter(rs.values()))["question"], f_s, 62, INK2)
        y += 4
        for c in ("greedy", "regular", "vcd"):
            if c in rs:
                y = _answer_line(d, x, y, c, rs[c]["raw_output"], rs[c]["is_correct"], f_t)
    canvas = canvas.crop((0, 0, canvas.width, min(H, y + PAD)))
    canvas.save(os.path.join(ROOT, out))


def fig_examples(df, out, pair_ids):
    """One row per example: image left, question + regular vs vcd answers right."""
    W_IMG, W_TXT, PAD = 300, 560, 18
    f_h, f_t, f_s = _font(14, True), _font(12), _font(11)
    rows = []
    for pid, tag in pair_ids:
        g = df[(df.pair_id == pid) & (df.condition.isin(["regular", "vcd"]))]
        rs = {r.condition: r for r in g.itertuples()}
        img = _fit(Image.open(os.path.join(ROOT, rs["vcd"].image_path)).convert("RGB"), W_IMG, 220)
        rows.append((pid, tag, rs, img))
    heights = [max(im.height, 150) + PAD for *_, im in rows]
    canvas = Image.new("RGB", (W_IMG + W_TXT + 3 * PAD, PAD + sum(heights)), SURFACE)
    d = ImageDraw.Draw(canvas)
    y0 = PAD
    for i, (pid, tag, rs, im) in enumerate(rows):
        row_h = heights[i]
        canvas.paste(im, (PAD + (W_IMG - im.width) // 2, y0))
        x = 2 * PAD + W_IMG
        color = GREEN if tag == "FIXED" else RED
        d.rectangle([x, y0 + 2, x + 8, y0 + 16], fill=color)
        d.text((x + 14, y0), "VCD %s it   -   %s  (%s)" % (tag, pid, rs["vcd"].domain), font=f_h, fill=INK)
        y = y0 + 26
        y = _wrap_lines(d, x, y, "Q: " + rs["vcd"].question, f_s, 78, INK2)
        y += 2
        d.text((x, y), "correct answer: %s      familiar (biased) answer: %s"
               % (rs["vcd"].ground_truth, rs["vcd"].expected_bias), font=f_t, fill=INK)
        y += 24
        for c in ("regular", "vcd"):
            y = _answer_line(d, x, y, c, rs[c].raw_output, bool(rs[c].correct), f_t)
        if i < len(rows) - 1:
            d.line([PAD, y0 + row_h - PAD // 2, canvas.width - PAD, y0 + row_h - PAD // 2], fill=GRID, width=1)
        y0 += row_h
    canvas.save(os.path.join(ROOT, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vlmbias_main_px768/records.jsonl")
    ap.add_argument("--smoke-records", default="outputs/smoke_optical_22pairs/records.jsonl")
    ap.add_argument("--out", default="figures/png")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    df = load(args.records)

    fig_acc_by_domain(df, os.path.join(out, "acc_by_domain.png"))
    fig_transitions(df, os.path.join(out, "transitions_by_domain.png"))
    fig_timing(df, os.path.join(out, "timing.png"))
    fig_pair(args.smoke_records, "opt_Ebbinghaus_str3_diff0p7", os.path.join(args.out, "pair_ebbinghaus.png"))
    fig_examples(df, os.path.join(args.out, "examples_fixed_broken.png"), [
        ("Flag_of_Malaysia_stripes_13_768", "FIXED"),
        ("MullerLyer_013_Q1_notitle_px768", "FIXED"),
        ("animal_038_caracal_notitle_Q2_px768", "BROKEN"),
        ("Flag_of_Greece_stripes_8_768_1", "BROKEN"),
    ])
    for t, f, b in transitions(df):
        print("%-18s fixed=%-3d broken=%d" % (t, f, b))
    print("wrote 5 figures to", out)


if __name__ == "__main__":
    main()
