"""Run LLaVA-1.5 over a pair manifest under one or more decoding conditions.

Writes one append-only JSONL record per (pair, image_variant, condition, seed) following the
SPEC §7 output contract, and resumes by skipping run_ids already present in the output file.

Example (single counterfactual image, plain LLaVA):
  python src/eval/run_inference.py --config configs/smoke_optical.yaml --run-name step5_plain \
      --conditions greedy --variants counterfactual --limit-pairs 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.eval.parse_answer import parse_answer, score  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_manifest(path: str, smoke_only: bool, limit: int | None, pair_ids: list[str] | None):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path))]
    if pair_ids:
        rows = [r for r in rows if r["pair_id"] in set(pair_ids)]
    if smoke_only:
        rows = [r for r in rows if r.get("smoke_subset")]
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--conditions", default="regular,vcd")
    ap.add_argument("--variants", default="canonical,counterfactual")
    ap.add_argument("--seeds", default=None, help="comma list; overrides config")
    ap.add_argument("--smoke-only", action="store_true", help="only pairs with smoke_subset=true")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--pair-ids", default=None, help="comma list of pair_ids")
    ap.add_argument("--diagnostics", action="store_true", help="also store first-step clean/noised/VCD diagnostics")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    conditions = args.conditions.split(",")
    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["seeds"])
    pairs = load_manifest(cfg["manifest"], args.smoke_only, args.limit_pairs, args.pair_ids.split(",") if args.pair_ids else None)

    out_dir = os.path.join(ROOT, "outputs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "records.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["run_id"])
            except Exception:
                pass

    # The generation config that defines a run (everything that changes the output)
    gen_common = {
        "conv_mode": cfg["conv_mode"], "max_new_tokens": cfg["max_new_tokens"],
        "image_aspect_ratio": "pad", "sampling": cfg["sampling"], "vcd": cfg["vcd"],
        "model_revision": cfg["model_revision"], "dataset_revision": cfg["dataset_revision"],
        "seeding": "per_item_derived_v1",
    }
    config_hash = hashlib.sha1(json.dumps(gen_common, sort_keys=True).encode()).hexdigest()[:12]
    json.dump({"config": cfg, "gen_common": gen_common, "config_hash": config_hash, "argv": sys.argv},
              open(os.path.join(out_dir, "run_config.json"), "w"), indent=1)

    from src.eval.vcd_adapter import LlavaRunner, install_patch, patch_status, resolve_snapshot
    import torch
    install_patch()
    ps = patch_status()
    print("patch status:", ps, flush=True)
    if "vcd" in conditions and not ps["sample_is_vcd"]:
        raise SystemExit("VCD monkeypatch not attached; refusing to run vcd condition")
    runner = LlavaRunner(resolve_snapshot(cfg["model_path"], cfg["model_revision"]), conv_mode=cfg["conv_mode"])
    gpu_name = torch.cuda.get_device_name(0)
    sha = git_sha()

    todo = [(p, v, c, s) for p in pairs for v in variants for c in conditions for s in (seeds if c != "greedy" else [0])]
    print(f"{len(todo)} generations planned ({len(pairs)} pairs x {variants} x {conditions} x seeds={seeds}); {len(done)} already done", flush=True)

    fout = open(out_path, "a")
    n_run = 0
    t_start = time.time()
    for p, variant, cond, seed in todo:
        run_id = hashlib.sha1(f"{p['pair_id']}|{variant}|{cond}|{seed}|{config_hash}".encode()).hexdigest()[:16]
        if run_id in done:
            continue
        img_path = os.path.join(ROOT, p[f"{variant}_path"])
        gt = p[f"{variant}_gt"]
        eb = p[f"{variant}_expected_bias"]
        image = Image.open(img_path)
        started = datetime.now(timezone.utc).isoformat()
        # Per-item seed derived from (run seed, pair, variant). Re-seeding with the *same* value before
        # every generation would make torch.multinomial draw the same quantile for every image, which
        # with a near-binary Yes/No distribution collapses "3 seeds" into 3 fixed thresholds.
        item_seed = (seed * 1_000_003 + zlib.crc32(f"{p['pair_id']}|{variant}".encode())) % (2 ** 31)
        res = runner.generate_once(
            p["prompt"], image, cond, item_seed, max_new_tokens=cfg["max_new_tokens"],
            temperature=cfg["sampling"]["temperature"], top_p=cfg["sampling"]["top_p"], top_k=cfg["sampling"]["top_k"],
            cd_alpha=cfg["vcd"]["cd_alpha"], cd_beta=cfg["vcd"]["cd_beta"], noise_step=cfg["vcd"]["noise_step"])
        parsed = parse_answer(res["raw_output"], p["answer_type"])
        is_correct, is_bias = score(parsed, gt, eb)
        rec = {
            "run_id": run_id, "pair_id": p["pair_id"], "image_variant": variant, "domain": p["domain"],
            "sub_domain": p["sub_domain"], "template_id": p["template_id"],
            "model_revision": cfg["model_revision"], "dataset_revision": cfg["dataset_revision"],
            "image_path": p[f"{variant}_path"], "prompt": res["prompt"], "question": p["prompt"],
            "raw_output": res["raw_output"], "parsed_answer": parsed.parsed_answer, "parse_status": parsed.parse_status,
            "parse_method": parsed.method,
            "ground_truth": gt, "expected_bias": eb, "is_correct": is_correct, "is_bias_answer": is_bias,
            "condition": cond, "seed": seed, "item_seed": item_seed, "generation_config": gen_common, "config_hash": config_hash,
            "n_new_tokens": res["n_new_tokens"], "hit_max_new_tokens": res["hit_max_new_tokens"],
            "n_cd_forward_calls": res["n_cd_forward_calls"], "first_step_topk": res["first_step_topk"],
            "first_step_n_unmasked": res["first_step_n_unmasked"],
            "git_sha": sha, "gpu_name": gpu_name, "hostname": platform.node(),
            "started_at": started, "duration_ms": round(res["duration_ms"], 1),
        }
        if args.diagnostics and cond == "vcd":
            rec["diagnostics"] = runner.first_step_diagnostics(
                p["prompt"], image, seed, cfg["vcd"]["cd_alpha"], cfg["vcd"]["cd_beta"], cfg["vcd"]["noise_step"])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_run += 1
        print(f"[{n_run}/{len(todo) - len(done)}] {p['pair_id']:38s} {variant:14s} {cond:8s} seed={seed:<3d} "
              f"{res['duration_ms']:7.0f}ms cd_calls={res['n_cd_forward_calls']:<3d} gt={gt:<3s} -> {res['raw_output']!r} "
              f"[{parsed.parse_status}{'' if is_correct is None else (' correct' if is_correct else ' WRONG')}]", flush=True)
    fout.close()
    print(f"done: {n_run} new records in {time.time() - t_start:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
