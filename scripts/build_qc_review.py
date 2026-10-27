"""Build a local HTML page for hand-checking the paired-image manifest.

For each pair it shows the canonical image, the counterfactual image, and a
diff overlay (changed pixels in red over a faded canonical). Reviewer marks
pass / fail / unsure with notes; results are exported as JSON and merged back
into the manifest with scripts/apply_qc_review.py.

Usage:
    python scripts/build_qc_review.py --manifest data/manifests/optical_pairs_v0.1.jsonl
    open outputs/qc_review/index.html
"""
import argparse
import html
import json
import os
import shutil

from PIL import Image, ImageChops

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pad_to(img, w, h):
    """Center img on a white canvas of size (w, h)."""
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def diff_overlay(a, b, threshold=20):
    w, h = max(a.width, b.width), max(a.height, b.height)
    a, b = pad_to(a, w, h), pad_to(b, w, h)
    mask = ImageChops.difference(a, b).convert("L").point(lambda v: 255 if v > threshold else 0)
    faded = Image.blend(a, Image.new("RGB", a.size, "white"), 0.7)
    red = Image.new("RGB", a.size, (230, 0, 0))
    out = Image.composite(red, faded, mask)
    n_changed = sum(1 for v in mask.getdata() if v)
    return out, n_changed, mask.getbbox()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/optical_pairs_v0.1.jsonl")
    ap.add_argument("--out-dir", default="outputs/qc_review")
    args = ap.parse_args()

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(os.path.join(out_dir, "diffs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "imgs"), exist_ok=True)
    pairs = [json.loads(l) for l in open(os.path.join(REPO, args.manifest))]

    cards = []
    for i, p in enumerate(pairs):
        a = Image.open(os.path.join(REPO, p["canonical_path"])).convert("RGB")
        b = Image.open(os.path.join(REPO, p["counterfactual_path"])).convert("RGB")
        overlay, n_changed, bbox = diff_overlay(a, b)
        diff_name = f"diffs/{p['pair_id']}.png"
        overlay.save(os.path.join(out_dir, diff_name))
        # copy source images next to the page: browsers block file:// pages from
        # loading images outside their own folder
        def rel(path):
            name = f"imgs/{os.path.basename(path)}"
            shutil.copy2(os.path.join(REPO, path), os.path.join(out_dir, name))
            return name
        size_note = "" if a.size == b.size else f"<span class=warn>size differs: {a.size[0]}x{a.size[1]} vs {b.size[0]}x{b.size[1]}</span>"
        e = html.escape
        cards.append(f"""
<section class=pair id="p{i}" data-id="{e(p['pair_id'])}">
  <header><b>{i + 1}/{len(pairs)}</b> <code>{e(p['pair_id'])}</code> · {e(p['sub_domain'])} {size_note}</header>
  <p class=prompt>{e(p['prompt'])}</p>
  <p class=meta>Intervention: {e(p.get('intervention', ''))} · changed pixels: {n_changed} · diff box: {bbox}</p>
  <div class=imgs>
    <figure><img src="{e(rel(p['canonical_path']))}" loading=lazy><figcaption>Normal → correct answer <b>{e(p['canonical_gt'])}</b></figcaption></figure>
    <figure><img src="{e(rel(p['counterfactual_path']))}" loading=lazy><figcaption>Edited → correct answer <b>{e(p['counterfactual_gt'])}</b></figcaption></figure>
    <figure><img src="{e(diff_name)}" loading=lazy><figcaption>What changed (red)</figcaption></figure>
  </div>
  <div class=controls>
    <button data-v=pass>Pass (P)</button><button data-v=fail>Fail (F)</button><button data-v=unsure>Unsure (U)</button>
    <input class=note placeholder="Note (required for fail/unsure)">
  </div>
</section>""")

    page = PAGE.replace("{{CARDS}}", "\n".join(cards)).replace("{{MANIFEST}}", html.escape(args.manifest))
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)
    print(f"wrote {out_dir}/index.html with {len(pairs)} pairs")


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Pair QC Review</title>
<style>
:root{--bg:#fafafa;--fg:#222;--card:#fff;--line:#ddd;--pass:#1a7f37;--fail:#c62828;--unsure:#b26a00}
body{margin:0;font:14px/1.4 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
#bar{position:sticky;top:0;z-index:2;background:#222;color:#fff;padding:8px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
#bar button{padding:4px 10px}
main{max-width:1400px;margin:0 auto;padding:16px}
.pair{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:12px;margin-bottom:16px}
.pair.cur{box-shadow:0 0 0 3px #4a8cff}
.pair[data-v=pass]{border-color:var(--pass)}.pair[data-v=fail]{border-color:var(--fail)}.pair[data-v=unsure]{border-color:var(--unsure)}
.prompt{font-weight:600;margin:6px 0}.meta{color:#666;margin:0 0 8px}.warn{color:var(--fail);font-weight:600}
.imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
figure{margin:0}figure img{width:100%;border:1px solid var(--line);background:#fff}
body.model figure img{width:336px;max-width:100%;image-rendering:auto}
figcaption{text-align:center;margin-top:4px}
.controls{display:flex;gap:8px;margin-top:8px}.controls .note{flex:1;padding:4px}
.controls button.on[data-v=pass]{background:var(--pass);color:#fff}.controls button.on[data-v=fail]{background:var(--fail);color:#fff}.controls button.on[data-v=unsure]{background:var(--unsure);color:#fff}
</style></head><body>
<div id=bar><b>Pair QC</b><span id=count></span>
<label><input type=checkbox id=model> Model-size view (336 px)</label>
<span>Keys: J/K next/prev · P/F/U mark · N note</span>
<button id=export>Export JSON</button></div>
<main>{{CARDS}}</main>
<script>
const KEY='qc_review::{{MANIFEST}}';
let state={};try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(state))}catch(e){}};
const cards=[...document.querySelectorAll('.pair')];let cur=0;
function paint(c){const s=state[c.dataset.id]||{};c.dataset.v=s.verdict||'';
 c.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.v===s.verdict));
 c.querySelector('.note').value=s.note||'';}
function count(){const v=Object.values(state);const n=k=>v.filter(s=>s.verdict===k).length;
 document.getElementById('count').textContent=`done ${v.filter(s=>s.verdict).length}/${cards.length} · pass ${n('pass')} · fail ${n('fail')} · unsure ${n('unsure')}`;}
function mark(c,v){const id=c.dataset.id;state[id]={...(state[id]||{}),verdict:v,at:new Date().toISOString()};save();paint(c);count();}
function focus(i){cur=Math.max(0,Math.min(cards.length-1,i));cards.forEach((c,j)=>c.classList.toggle('cur',j===cur));cards[cur].scrollIntoView({block:'start',behavior:'smooth'});}
cards.forEach((c,i)=>{paint(c);
 c.querySelectorAll('button').forEach(b=>b.onclick=()=>{mark(c,b.dataset.v);focus(i)});
 c.querySelector('.note').oninput=e=>{const id=c.dataset.id;state[id]={...(state[id]||{}),note:e.target.value};save();};
 c.onclick=()=>{cur=i;cards.forEach((d,j)=>d.classList.toggle('cur',j===cur));};});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();
 if(k==='j')focus(cur+1);else if(k==='k')focus(cur-1);
 else if(k==='p'){mark(cards[cur],'pass');focus(cur+1)}else if(k==='f')mark(cards[cur],'fail');else if(k==='u')mark(cards[cur],'unsure');
 else if(k==='n'){e.preventDefault();cards[cur].querySelector('.note').focus();}});
document.getElementById('model').onchange=e=>document.body.classList.toggle('model',e.target.checked);
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify({manifest:'{{MANIFEST}}',reviewed_at:new Date().toISOString(),reviews:state},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qc_review.json';a.click();};
count();focus(0);
</script></body></html>"""

if __name__ == "__main__":
    main()
