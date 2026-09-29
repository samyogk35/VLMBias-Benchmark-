# Build a single self-contained HTML page for browsing a vlmbias_main run: every row's image,
# question, and the greedy / regular / VCD answers side by side, with filters for the rows VCD
# fixed or broke. Opens from file:// (no CDN, no server); images are referenced relative to the
# repo, so keep the page under figures/.
#
#   python src/analysis/build_explorer.py                    # -> figures/explorer.html
#   open figures/explorer.html

import argparse
import html
import json
import os
import sys
import urllib.parse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.analysis.summarize_benchmark import TOPIC_ORDER, load, topic_table  # noqa: E402

CONDS = ["greedy", "regular", "vcd"]
PRETTY = {"animals": "animals", "chess_pieces": "chess pieces", "flags": "flags", "game_boards": "game boards",
          "logos": "logos", "optical_illusion": "optical illusions", "patterned_grid": "patterned grids"}


def rows_for_page(path):
    """One entry per (pair_id, sampling seed) with the three conditions nested."""
    by = defaultdict(dict)
    meta = {}
    for line in open(os.path.join(ROOT, path)):
        r = json.loads(line)
        key = r["pair_id"]
        meta.setdefault(key, {
            "id": r["pair_id"], "domain": r["domain"], "sub": r["sub_domain"], "q": r["question"],
            "qf": r["question_form"], "img": r["image_path"], "gt": str(r["ground_truth"]),
            "bias": str(r["expected_bias"]),
        })
        by[key][r["condition"]] = {
            "raw": r["raw_output"], "parsed": None if r["parsed_answer"] is None else str(r["parsed_answer"]),
            "status": r["parse_status"], "ok": bool(r["is_correct"]), "biased": bool(r["is_bias_answer"]),
            "cd": r["n_cd_forward_calls"], "ms": round(r["duration_ms"]), "seed": r["seed"],
            # APC masking sets pruned tokens to -1e30; those are not candidates, drop them
            "topk": [[t.replace("▁", " ").strip() or "␣", round(v, 1)]
                     for t, v in r.get("first_step_topk", []) if v > -1e9][:3],
        }
    out = []
    for key, m in meta.items():
        c = by[key]
        if "regular" in c and "vcd" in c:
            m["flip"] = ("fixed" if (not c["regular"]["ok"] and c["vcd"]["ok"])
                         else "broken" if (c["regular"]["ok"] and not c["vcd"]["ok"]) else "same")
            m["differ"] = c["regular"]["raw"] != c["vcd"]["raw"]
        else:
            m["flip"], m["differ"] = "same", False
        m["c"] = c
        out.append(m)
    # order: domain, then pair id
    out.sort(key=lambda m: (TOPIC_ORDER.index(m["domain"]) if m["domain"] in TOPIC_ORDER else 99, m["id"]))
    return out


def svg_bars(table, conds):
    """Inline SVG: accuracy per topic, one thin bar per condition (blue greedy, orange vcd)."""
    colors = {"greedy": "#2a78d6", "regular": "#7fa7dc", "vcd": "#eb6834"}
    topics = [t for t in table.topic if t != "total"] + ["total"]
    row_h, bar_h, gap, left, width = 30, 7, 2, 130, 520
    h = 18 + row_h * len(topics)
    parts = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="accuracy by topic">' % (left + width + 60, h)]
    for x in range(0, 61, 10):
        px = left + x / 60 * width
        parts.append('<line x1="%.1f" y1="8" x2="%.1f" y2="%d" stroke="#e6e5e1" stroke-width="1"/>' % (px, px, h - 10))
        parts.append('<text x="%.1f" y="%d" font-size="10" fill="#52514e" text-anchor="middle">%d</text>' % (px, h - 1, x))
    for i, t in enumerate(topics):
        y0 = 12 + i * row_h
        label = "TOTAL" if t == "total" else PRETTY.get(t, t)
        parts.append('<text x="%d" y="%d" font-size="11" fill="#0b0b0b" text-anchor="end">%s</text>' % (left - 8, y0 + 12, html.escape(label)))
        row = table[table.topic == t].iloc[0]
        for j, c in enumerate(conds):
            v = float(row["acc_" + c])
            y = y0 + j * (bar_h + gap)
            parts.append('<rect x="%d" y="%d" width="%.1f" height="%d" rx="2" fill="%s"><title>%s %s: %.2f%%</title></rect>'
                         % (left, y, max(v / 60 * width, 1), bar_h, colors[c], label, c, v))
            parts.append('<text x="%.1f" y="%d" font-size="9" fill="#52514e">%.2f</text>' % (left + max(v / 60 * width, 1) + 4, y + bar_h, v))
    parts.append("</svg>")
    legend = " ".join('<span class="lg"><i style="background:%s"></i>%s</span>' % (colors[c], c) for c in conds)
    return "".join(parts), legend


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>VLMBias px768 explorer</title>
<style>
  :root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e6e5e1; --card:#ffffff;
          --blue:#2a78d6; --orange:#eb6834; --green:#008300; --red:#e34948; --grey:#8a8983; }
  * { box-sizing: border-box; }
  body { margin:0; padding:20px 24px 60px; background:var(--bg); color:var(--ink);
         font: 14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  h1 { font-size:20px; margin:0 0 4px; }
  .src { color:var(--ink2); font-size:12px; margin-bottom:16px; }
  .top { display:grid; grid-template-columns: 1fr 1.3fr; gap:24px; align-items:start; margin-bottom:18px; }
  .tiles { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .tile .l { font-size:11px; color:var(--ink2); }
  .tile .v { font-size:24px; font-weight:600; }
  .tile .v small { font-size:12px; font-weight:400; color:var(--ink2); }
  .tile.fixed .v { color:var(--green); } .tile.broken .v { color:var(--red); }
  .chart { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .chart .t { font-size:12px; font-weight:600; margin-bottom:4px; }
  .lg { font-size:11px; color:var(--ink2); margin-right:12px; }
  .lg i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:0 0 12px; position:sticky; top:0;
         background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line); z-index:2; }
  .chip { border:1px solid var(--line); background:var(--card); border-radius:999px; padding:4px 12px;
          cursor:pointer; font-size:13px; }
  .chip.on { background:var(--ink); color:#fff; border-color:var(--ink); }
  select, input[type=search] { font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); }
  input[type=search] { width:220px; }
  .count { color:var(--ink2); font-size:13px; margin-left:auto; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; display:grid;
          grid-template-columns: 170px 1fr; }
  .card img { width:170px; height:170px; object-fit:contain; background:#f3f2ee; display:block; }
  .card .b { padding:10px 12px; min-width:0; }
  .card .h { font-size:11px; color:var(--ink2); display:flex; gap:8px; align-items:center; margin-bottom:2px; }
  .card .h b { color:var(--ink); font-weight:600; flex:1; }
  .card .pid { font-size:10.5px; color:var(--ink2); font-family: ui-monospace, Menlo, monospace; margin-bottom:5px;
               overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .flag { font-size:10px; font-weight:600; padding:1px 7px; border-radius:999px; color:#fff; }
  .flag.fixed { background:var(--green); } .flag.broken { background:var(--red); }
  .card .q { font-size:12.5px; margin-bottom:6px; }
  .card .gt { font-size:12px; color:var(--ink2); margin-bottom:6px; }
  .ans { display:grid; grid-template-columns: 58px 1fr auto; gap:6px 8px; align-items:center; font-size:12.5px; }
  .ans .c { color:var(--ink2); }
  .ans .r { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; word-break:break-word; }
  .ans .r.vcd { font-weight:600; }
  .badge { font-size:10px; padding:1px 6px; border-radius:4px; color:#fff; white-space:nowrap; }
  .badge.ok { background:var(--green); } .badge.no { background:var(--red); } .badge.inv { background:var(--grey); }
  .meta { font-size:10.5px; color:var(--ink2); margin-top:6px; }
  .meta code { font-family: ui-monospace, Menlo, monospace; }
  .empty { color:var(--ink2); padding:40px; text-align:center; }
</style>

<h1>VLMBias main split at 768 px &mdash; LLaVA-1.5-7B, greedy vs regular sampling vs VCD</h1>
<div class="src">__SRC__</div>

<div class="top">
  <div class="tiles">__TILES__</div>
  <div class="chart"><div class="t">Accuracy by topic (%)</div>__SVG__<div>__LEGEND__</div></div>
</div>

<div class="bar">
  <span id="chips"></span>
  <select id="domain"><option value="">all topics</option>__DOMOPTS__</select>
  <input id="search" type="search" placeholder="search pair id / sub-topic / answer">
  <span class="count" id="count"></span>
</div>
<div class="grid" id="grid"></div>

<script>
const ROWS = __ROWS__;
const CONDS = ["greedy", "regular", "vcd"];
const FILTERS = [
  ["fixed",   "VCD fixed it",        r => r.flip === "fixed"],
  ["broken",  "VCD broke it",        r => r.flip === "broken"],
  ["differ",  "regular ≠ VCD",  r => r.differ],
  ["biased",  "answered the familiar (wrong) answer", r => CONDS.some(c => r.c[c] && r.c[c].biased)],
  ["invalid", "unparseable output",  r => CONDS.some(c => r.c[c] && r.c[c].status !== "valid")],
  ["all",     "all rows",            r => true],
];
let active = "fixed";
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));

function badge(a) {
  if (!a) return "";
  if (a.status !== "valid") return '<span class="badge inv">' + a.status + '</span>';
  return a.ok ? '<span class="badge ok">correct</span>' : '<span class="badge no">wrong' + (a.biased ? " · familiar" : "") + '</span>';
}
function card(r) {
  const ans = CONDS.map(c => {
    const a = r.c[c]; if (!a) return "";
    return '<div class="c">' + c + '</div><div class="r ' + c + '" title="' + esc(a.raw) + '">' + esc(JSON.stringify(a.raw)) + '</div>' + badge(a);
  }).join("");
  const meta = CONDS.filter(c => r.c[c]).map(c => c + ": " + r.c[c].ms + " ms, cd_calls=" + r.c[c].cd).join(" · ");
  const topk = CONDS.filter(c => r.c[c] && r.c[c].topk.length).map(c =>
    c + " first token: " + r.c[c].topk.map(t => "<code>" + esc(t[0]) + "</code>&thinsp;" + t[1]).join(", ")).join("<br>");
  return '<div class="card">' +
    '<img loading="lazy" src="' + esc(r.src) + '" alt="">' +
    '<div class="b"><div class="h"><b>' + esc(r.sub) + '</b>' +
      (r.flip !== "same" ? '<span class="flag ' + r.flip + '">' + (r.flip === "fixed" ? "VCD fixed" : "VCD broke") + '</span>' : "") + '</div>' +
    '<div class="pid" title="' + esc(r.id) + '">' + esc(r.id) + '</div>' +
    '<div class="q">' + esc(r.q) + '</div>' +
    '<div class="gt">correct answer <b>' + esc(r.gt) + '</b> &nbsp;·&nbsp; familiar (biased) answer <b>' + esc(r.bias) + '</b></div>' +
    '<div class="ans">' + ans + '</div>' +
    '<div class="meta">' + meta + '<br>' + topk + '</div></div></div>';
}
function render() {
  const f = FILTERS.find(x => x[0] === active)[2];
  const dom = $("domain").value;
  const q = $("search").value.trim().toLowerCase();
  const rows = ROWS.filter(r => f(r) && (!dom || r.domain === dom) &&
    (!q || (r.id + " " + r.sub + " " + CONDS.map(c => r.c[c] ? r.c[c].raw : "").join(" ")).toLowerCase().includes(q)));
  $("count").textContent = rows.length + " of " + ROWS.length + " rows";
  $("grid").innerHTML = rows.length ? rows.slice(0, 400).map(card).join("") : '<div class="empty">no rows match</div>';
}
$("chips").innerHTML = FILTERS.map(([k, label]) =>
  '<button class="chip' + (k === active ? " on" : "") + '" data-k="' + k + '">' + label +
  ' <small>(' + ROWS.filter(FILTERS.find(x => x[0] === k)[2]).length + ')</small></button>').join(" ");
$("chips").addEventListener("click", e => {
  const b = e.target.closest(".chip"); if (!b) return;
  active = b.dataset.k;
  document.querySelectorAll(".chip").forEach(x => x.classList.toggle("on", x === b));
  render();
});
$("domain").addEventListener("change", render);
$("search").addEventListener("input", render);
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/vlmbias_main_px768/records.jsonl")
    ap.add_argument("--out", default="figures/explorer.html")
    args = ap.parse_args()

    rows = rows_for_page(args.records)
    out_dir = os.path.dirname(os.path.join(ROOT, args.out))
    for r in rows:
        rel = os.path.relpath(os.path.join(ROOT, r["img"]), out_dir)
        r["src"] = "/".join(urllib.parse.quote(p) for p in rel.split(os.sep))

    df = load(args.records)
    conds = [c for c in CONDS if (df.condition == c).any()]
    table = topic_table(df, conds)
    tot = table[table.topic == "total"].iloc[0]
    n_fixed = sum(r["flip"] == "fixed" for r in rows)
    n_broken = sum(r["flip"] == "broken" for r in rows)
    first = json.loads(open(os.path.join(ROOT, args.records)).readline())

    tiles = "".join([
        '<div class="tile"><div class="l">greedy accuracy</div><div class="v">%.2f<small> %% &nbsp;EnAR 16.92</small></div></div>' % tot.acc_greedy,
        '<div class="tile"><div class="l">regular sampling</div><div class="v">%.2f<small> %%</small></div></div>' % tot.acc_regular,
        '<div class="tile"><div class="l">VCD accuracy</div><div class="v">%.2f<small> %% &nbsp;EnAR 19.18</small></div></div>' % tot.acc_vcd,
        '<div class="tile fixed"><div class="l">VCD fixed (wrong &rarr; right)</div><div class="v">%d</div></div>' % n_fixed,
        '<div class="tile broken"><div class="l">VCD broke (right &rarr; wrong)</div><div class="v">%d</div></div>' % n_broken,
        '<div class="tile"><div class="l">rows &times; conditions</div><div class="v">%d<small> &times; %d</small></div></div>' % (len(rows), len(conds)),
    ])
    svg, legend = svg_bars(table, [c for c in conds if c != "regular"] if len(conds) == 3 else conds)
    src = ("%s &middot; %d records &middot; seeds greedy=0 / sampling=%s &middot; config_hash %s &middot; git %s &middot; model rev %s"
           % (html.escape(args.records), len(df), sorted(df[df.condition != "greedy"].seed.unique().tolist()),
              first["config_hash"], first["git_sha"], first["model_revision"][:8]))
    domopts = "".join('<option value="%s">%s</option>' % (t, PRETTY.get(t, t)) for t in TOPIC_ORDER if (df.domain == t).any())

    page = (PAGE.replace("__SRC__", src).replace("__TILES__", tiles).replace("__SVG__", svg)
            .replace("__LEGEND__", legend).replace("__DOMOPTS__", domopts)
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")))
    with open(os.path.join(ROOT, args.out), "w") as f:
        f.write(page)
    print("wrote %s: %d rows, fixed=%d broken=%d" % (args.out, len(rows), n_fixed, n_broken))


if __name__ == "__main__":
    main()
