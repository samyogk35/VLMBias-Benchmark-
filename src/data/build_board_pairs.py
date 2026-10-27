# Build canonical / counterfactual pairs for the game-board domain (chess, sudoku, go) by
# rendering BOTH images with the VLMBias generator code, on the same machine, at 768 px.
#
# The counterfactual images are the generator's own row/column edits (the same variations as the
# benchmark's main split). The canonical image is the standard board drawn by the same function.
# Rendering both locally keeps fonts and the cairo version identical inside a pair. Each
# counterfactual render is also compared with the released benchmark image: they do NOT match
# (the released chess boards have a thick frame and fill the canvas, the released Go boards
# place stones differently), so the released images came from a different generator version
# than the released code, and a canonical matching them cannot be rendered. Pairs therefore
# use local renders for both members; the comparison is kept in benchmark_check.
# Xiangqi is left out: its river text needs a CJK font, so renders are machine-dependent.
#
# Every image gets an automatic count check (count_cells) that must match its ground truth.
#
# Needs cairosvg + svgwrite (and libcairo):
#   python src/data/build_board_pairs.py --out data/manifests/board_pairs_v0.1.jsonl

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VB = os.path.join(ROOT, "third_party", "vlms-are-biased")
for p in (VB, os.path.join(VB, "generators")):
    if p not in sys.path:
        sys.path.insert(0, p)

import grid_utils  # noqa: E402
import chess_board_generator as chess  # noqa: E402
import go_board_generator as go  # noqa: E402
import sudoku_board_generator as sudoku  # noqa: E402

PX = 768
SVG_SIZE = 800
QUALITY = 5.0
IMG_DIR = "data/images/boards"
BENCH_DIR = "data/images/vlmbias_main"

ROW_Q = "How many rows are there on this board? Answer with a number in curly brackets, e.g., {9}."
COL_Q = "How many columns are there on this board? Answer with a number in curly brackets, e.g., {9}."
SUDOKU_ROW_Q = "How many rows are there on this puzzle? Answer with a number in curly brackets, e.g., {9}."
SUDOKU_COL_Q = "How many columns are there on this puzzle? Answer with a number in curly brackets, e.g., {9}."

# (variation name, action, benchmark image basename); order and names follow each generator
BOARDS = {
    "chess": {
        "module": chess, "cls": chess.ChessBoard, "draw": chess.draw_chess_board_svg,
        "std": (8, 8), "q": (ROW_Q, COL_Q), "kind": "squares",
        "side_changes": "board keeps its outer size, so every square is resized when a row or column is added or removed",
        "variations": [
            ("remove_first_row", lambda b: b.remove_row("first"), "chess_grid_01_row_remove_first_row_notitle"),
            ("remove_last_row", lambda b: b.remove_row("last"), "chess_grid_02_row_remove_last_row_notitle"),
            ("add_first_row", lambda b: b.add_row("first"), "chess_grid_03_row_add_first_row_notitle"),
            ("add_last_row", lambda b: b.add_row("last"), "chess_grid_04_row_add_last_row_notitle"),
            ("remove_first_col", lambda b: b.remove_column("first"), "chess_grid_05_col_remove_first_col_notitle"),
            ("remove_last_col", lambda b: b.remove_column("last"), "chess_grid_06_col_remove_last_col_notitle"),
            ("add_first_col", lambda b: b.add_column("first"), "chess_grid_07_col_add_first_col_notitle"),
            ("add_last_col", lambda b: b.add_column("last"), "chess_grid_08_col_add_last_col_notitle"),
        ],
    },
    "sudoku": {
        "module": sudoku, "cls": sudoku.SudokuGrid, "draw": sudoku.draw_sudoku_board_svg,
        "std": (9, 9), "q": (SUDOKU_ROW_Q, SUDOKU_COL_Q), "kind": "lines_minus_one",
        "side_changes": "grid keeps its outer size, so every cell (and its digit) is resized",
        "variations": [
            ("add_first_row", lambda g: g.add_row("first"), "sudoku_grid_01_row_add_first_row_notitle"),
            ("add_last_row", lambda g: g.add_row("last"), "sudoku_grid_02_row_add_last_row_notitle"),
            ("add_first_col", lambda g: g.add_column("first"), "sudoku_grid_03_col_add_first_col_notitle"),
            ("add_last_col", lambda g: g.add_column("last"), "sudoku_grid_04_col_add_last_col_notitle"),
            ("remove_first_row", lambda g: g.remove_row("first"), "sudoku_grid_05_row_remove_first_row_notitle"),
            ("remove_last_row", lambda g: g.remove_row("last"), "sudoku_grid_06_row_remove_last_row_notitle"),
            ("remove_first_col", lambda g: g.remove_column("first"), "sudoku_grid_07_col_remove_first_col_notitle"),
            ("remove_last_col", lambda g: g.remove_column("last"), "sudoku_grid_08_col_remove_last_col_notitle"),
        ],
    },
    "go": {
        "module": go, "cls": go.GoBoard, "draw": go.draw_go_board_svg,
        "std": (19, 19), "q": (ROW_Q, COL_Q), "kind": "lines",
        "side_changes": "star points (hoshi) are drawn only on the standard 19x19 board, so every edited board loses them; spacing is resized",
        "variations": [
            ("remove_row_last", lambda b: b.remove_row("last"), "go_grid_01_remove_row"),
            ("remove_col_last", lambda b: b.remove_column("last"), "go_grid_02_remove_col"),
            ("add_row_last", lambda b: b.add_row("last"), "go_grid_03_add_row"),
            ("add_col_last", lambda b: b.add_column("last"), "go_grid_04_add_col"),
        ],
    },
}


def render(spec, board, path, details=None):
    kw = {}
    if spec["module"] is sudoku and details:
        kw = {"orig_grid_start_col_in_new": details.get("orig_start_x_in_new", 0),
              "orig_grid_start_row_in_new": details.get("orig_start_y_in_new", 0)}
    svg = spec["draw"](board, render_size=SVG_SIZE, **kw)
    ok = grid_utils.svg_to_png_direct(svg, path, scale=QUALITY * PX / float(SVG_SIZE), output_size=PX)
    if not ok:
        raise RuntimeError("render failed: " + path)


def _runs(mask_1d, min_len=2):
    runs, cur = 0, 0
    for v in mask_1d:
        cur = cur + 1 if v else 0
        if cur == min_len:
            runs += 1
    return runs


def count_cells(path, kind, axis):
    """Count rows (axis='rows') or columns (axis='cols') from pixels, mode over many scanlines.

    squares: alternating light/dark squares, counted as colour runs along the scanline.
    lines: dark grid lines, counted as dark runs.  lines_minus_one: cells = lines - 1.
    """
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    if axis == "rows":
        a = a.transpose(1, 0, 2)  # scan down columns instead of along rows
    h = a.shape[0]
    counts = []
    for y in np.linspace(0.2 * h, 0.8 * h, 41).astype(int):
        line = a[y]
        if kind == "squares":
            light = np.abs(line - [0xf0, 0xd9, 0xb5]).sum(1) < 40
            dark = np.abs(line - [0xb5, 0x88, 0x63]).sum(1) < 40
            counts.append(_runs(light, 3) + _runs(dark, 3))
        else:
            # go: black lines on wood (#DCB35C, mean ~164) need a strict threshold; sudoku's thin
            # lines on white antialias to mid-grey and need a lighter one
            n = _runs(line.sum(1) < 3 * (100 if kind == "lines" else 200), 1)
            counts.append(n - 1 if kind == "lines_minus_one" else n)
    return Counter(counts).most_common(1)[0][0]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifests/board_pairs_v0.1.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, IMG_DIR), exist_ok=True)
    bench = {}
    for l in open(os.path.join(ROOT, "data/manifests/vlmbias_main_px768_v0.1.jsonl")):
        r = json.loads(l)
        if r["domain"] == "game_boards" and r["question_form"] == "Q1":
            bench[os.path.basename(r["counterfactual_path"])] = r

    rows, problems = [], []
    for name, spec in BOARDS.items():
        std_rows, std_cols = spec["std"]
        canon_path = "%s/%s_standard_px%d.png" % (IMG_DIR, name, PX)
        render(spec, spec["cls"](std_rows, std_cols), os.path.join(ROOT, canon_path))
        for dim, q in zip(("rows", "cols"), spec["q"]):
            got = count_cells(os.path.join(ROOT, canon_path), spec["kind"], dim)
            want = std_rows if dim == "rows" else std_cols
            if got != want:
                problems.append("%s canonical %s: counted %d, expected %d" % (name, dim, got, want))

        for var, action, bench_base in spec["variations"]:
            board = spec["cls"](std_rows, std_cols)
            details = action(board)
            dim = "rows" if details["dimension"] == "row" else "cols"
            cf_path = "%s/%s_%s_px%d.png" % (IMG_DIR, name, var, PX)
            render(spec, board, os.path.join(ROOT, cf_path), details)
            cf_gt = board.rows if dim == "rows" else board.cols
            c_gt = std_rows if dim == "rows" else std_cols

            counted = {v: count_cells(os.path.join(ROOT, p), spec["kind"], dim) for v, p in (("canonical", canon_path), ("counterfactual", cf_path))}
            if counted != {"canonical": c_gt, "counterfactual": cf_gt}:
                problems.append("%s %s: counted %s, expected %d / %d" % (name, var, counted, c_gt, cf_gt))

            # pipeline check against the released benchmark image and its label
            bench_file = "%s_px%d.png" % (bench_base, PX)
            b = bench.get(bench_file)
            bench_note = "no benchmark row"
            if b is not None:
                if str(cf_gt) != str(b["counterfactual_gt"]):
                    problems.append("%s %s: label %s vs benchmark %s" % (name, var, cf_gt, b["counterfactual_gt"]))
                ours = np.asarray(Image.open(os.path.join(ROOT, cf_path)).convert("L")).astype(int)
                theirs = np.asarray(Image.open(os.path.join(ROOT, BENCH_DIR, bench_file)).convert("L")).astype(int)
                if ours.shape != theirs.shape:
                    bench_note = "size differs %s vs %s" % (ours.shape, theirs.shape)
                else:
                    d = np.abs(ours - theirs)
                    bench_note = "max px diff %d, %.3f%% pixels differ by >32" % (d.max(), 100 * (d > 32).mean())

            rows.append({
                "pair_id": "board_%s_%s" % (name, var), "domain": "game_boards", "sub_domain": name,
                "template_id": "board_%s_%s" % (name, dim),
                "prompt": spec["q"][0] if dim == "rows" else spec["q"][1],
                "answer_type": "number", "question_form": "Q1", "resolution": PX,
                "canonical_path": canon_path, "counterfactual_path": cf_path,
                "canonical_gt": str(c_gt), "counterfactual_gt": str(cf_gt),
                "canonical_expected_bias": str(c_gt), "counterfactual_expected_bias": str(c_gt), "expected_bias": str(c_gt),
                "intervention": var.replace("_", " "), "edit_direction": "adds" if cf_gt > c_gt else "removes",
                "generator_params": {"board": name, "variation": var, "standard": "%dx%d" % (std_rows, std_cols),
                                     "edited": "%dx%d" % (board.rows, board.cols), "svg_size": SVG_SIZE,
                                     "quality_scale": QUALITY, "renderer": "third_party/vlms-are-biased/generators/%s_board_generator.py" % name},
                "measured": counted, "benchmark_image": bench_file, "benchmark_check": bench_note,
                "side_changes": spec["side_changes"],
                "canonical_md5": md5(os.path.join(ROOT, canon_path)), "counterfactual_md5": md5(os.path.join(ROOT, cf_path)),
                "canonical_size": list(Image.open(os.path.join(ROOT, canon_path)).size),
                "counterfactual_size": list(Image.open(os.path.join(ROOT, cf_path)).size),
                "qc_status": "auto_checked", "qc_notes": "both members rendered locally (released benchmark images come from a different generator version); counts verified from pixels; not yet reviewed by eye",
                "manifest_version": "v0.1",
            })

    for r in rows:
        print("%-32s %-4s gt %s->%s counted %s | %s" % (r["pair_id"], r["edit_direction"], r["canonical_gt"],
              r["counterfactual_gt"], r["measured"], r["benchmark_check"]))
    if problems:
        sys.exit("problems:\n  " + "\n  ".join(problems))
    with open(os.path.join(ROOT, args.out), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote %d pairs (%d templates) -> %s" % (len(rows), len({r["template_id"] for r in rows}), args.out))


if __name__ == "__main__":
    main()
