# Run LLaVA over the pair manifest and write one jsonl line per generation.
# Re-running with the same run name skips what's already done.
#
#   python src/eval/run_inference.py --config configs/smoke_optical.yaml --run-name test \
#       --conditions greedy --variants counterfactual --limit-pairs 1

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone

import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.eval.parse_answer import parse_answer, score


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_manifest(path, smoke_only, limit, pair_ids):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path))]
    if pair_ids:
        rows = [r for r in rows if r["pair_id"] in pair_ids]
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
    ap.add_argument("--seeds", default=None, help="comma separated, overrides the config")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--pair-ids", default=None, help="comma separated")
    ap.add_argument("--diagnostics", action="store_true", help="also store first-step clean/noisy logit info for vcd")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config)))
    conditions = args.conditions.split(",")
    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["seeds"])
    pair_ids = set(args.pair_ids.split(",")) if args.pair_ids else None
    pairs = load_manifest(cfg["manifest"], args.smoke_only, args.limit_pairs, pair_ids)

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

    # everything that affects the output goes into the config hash
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
        sys.exit("VCD patch not attached, not running the vcd condition")
    runner = LlavaRunner(resolve_snapshot(cfg["model_path"], cfg["model_revision"]), conv_mode=cfg["conv_mode"])
    gpu_name = torch.cuda.get_device_name(0)
    sha = git_sha()

    todo = [(p, v, c, s) for p in pairs for v in variants for c in conditions
            for s in (seeds if c != "greedy" else [0])]
    print("%d generations planned (%d pairs x %s x %s x seeds=%s), %d already done"
          % (len(todo), len(pairs), variants, conditions, seeds, len(done)), flush=True)

    fout = open(out_path, "a")
    n_run = 0
    t_start = time.time()
    for p, variant, cond, seed in todo:
        run_id = hashlib.sha1(("%s|%s|%s|%s|%s" % (p["pair_id"], variant, cond, seed, config_hash)).encode()).hexdigest()[:16]
        if run_id in done:
            continue
        gt = p[variant + "_gt"]
        eb = p[variant + "_expected_bias"]
        image = Image.open(os.path.join(ROOT, p[variant + "_path"]))
        started = datetime.now(timezone.utc).isoformat()
        # seed per image, derived from the run seed. Calling set_seed(seed) with the same value
        # before every image made torch.multinomial draw the same random number every time, so
        # with a 2-token Yes/No distribution one "seed" answered Yes to every single image.
        item_seed = (seed * 1_000_003 + zlib.crc32((p["pair_id"] + "|" + variant).encode())) % (2 ** 31)
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
            "image_path": p[variant + "_path"], "prompt": res["prompt"], "question": p["prompt"],
            "raw_output": res["raw_output"], "parsed_answer": parsed.parsed_answer, "parse_status": parsed.parse_status,
            "parse_method": parsed.method,
            "ground_truth": gt, "expected_bias": eb, "is_correct": is_correct, "is_bias_answer": is_bias,
            "condition": cond, "seed": seed, "item_seed": item_seed, "generation_config": gen_common,
            "config_hash": config_hash,
            "n_new_tokens": res["n_new_tokens"], "hit_max_new_tokens": res["hit_max_new_tokens"],
            "n_cd_forward_calls": res["n_cd_forward_calls"], "first_step_topk": res["first_step_topk"],
            "first_step_n_unmasked": res["first_step_n_unmasked"],
            "git_sha": sha, "gpu_name": gpu_name, "hostname": platform.node(),
            "started_at": started, "duration_ms": round(res["duration_ms"], 1),
        }
        if args.diagnostics and cond == "vcd":
            rec["diagnostics"] = runner.first_step_diagnostics(
                p["prompt"], image, item_seed, cfg["vcd"]["cd_alpha"], cfg["vcd"]["cd_beta"], cfg["vcd"]["noise_step"])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_run += 1
        status = parsed.parse_status
        if is_correct is not None:
            status += " correct" if is_correct else " WRONG"
        print("[%d/%d] %-38s %-14s %-8s seed=%-3d %6.0fms cd_calls=%-3d gt=%-3s -> %r [%s]"
              % (n_run, len(todo) - len(done), p["pair_id"], variant, cond, seed, res["duration_ms"],
                 res["n_cd_forward_calls"], gt, res["raw_output"], status), flush=True)
    fout.close()
    print("done: %d new records in %.1fs -> %s" % (n_run, time.time() - t_start, out_path))


if __name__ == "__main__":
    main()
