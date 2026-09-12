# Auditing Visual Contrastive Decoding with Canonical–Counterfactual Pairs

Senior seminar project (Fall 2026). **Research question:** when Visual Contrastive Decoding (VCD)
improves LLaVA-1.5-7B accuracy on the counterfactual images of the VLMBias benchmark, does it also
improve correct discrimination of *matched* canonical–counterfactual pairs — or does it merely shift the
answer distribution away from the familiar answer (which happens to score well on a counterfactual-only
test set)?

Full design: [SPEC.md](SPEC.md). Pinned revisions of every dependency: [VERSIONS.md](VERSIONS.md).

## Status (2026-09-12) — progress-demo scope

| Item | State |
|---|---|
| Pinned Python 3.10 env (torch 2.0.1+cu118, transformers 4.31.0) | done, `environment.lock.txt` |
| Exact VCD / VLMBias / model / dataset revisions | done, [VERSIONS.md](VERSIONS.md) |
| Dataset inventory | done, [data/inventory/summary.md](data/inventory/summary.md) |
| Optical-illusion pair manifest | done, 60 pairs / 22 templates, [data/manifests/optical_pairs_v0.1.jsonl](data/manifests/optical_pairs_v0.1.jsonl) |
| Plain LLaVA inference | done |
| VCD inference with verified contrastive path | done (see below) |
| Raw JSONL outputs + parser audit | done, [outputs/README.md](outputs/README.md), [figures/parser_audit.md](figures/parser_audit.md) |
| 22-pair smoke test, greedy / regular / VCD, 3 seeds | done, [figures/smoke_optical_22pairs.md](figures/smoke_optical_22pairs.md) — **smoke test, not a finding** |

## Setup

```bash
bash scripts/setup_env.sh              # conda env "vcd" with the pinned stack (~10 min)
conda activate vcd
bash scripts/clone_third_party.sh      # VCD + VLMBias at pinned commits -> third_party/
python scripts/download_assets.py      # ~16 GB: LLaVA-1.5-7B, CLIP ViT-L/14-336, VLMBias parquet
python src/data/build_inventory.py     # -> data/inventory/
python src/data/build_pairs.py         # -> data/manifests/optical_pairs_v0.1.jsonl + data/images/optical/
python -m pytest -q tests
```

## Running inference

```bash
export CUDA_VISIBLE_DEVICES=0          # one RTX A5000, FP16, batch size 1
python src/eval/run_inference.py --config configs/smoke_optical.yaml --run-name smoke_optical_22pairs \
    --conditions greedy,regular,vcd --variants canonical,counterfactual --smoke-only
python src/analysis/summarize.py outputs/smoke_optical_22pairs/records.jsonl --out figures/smoke_optical_22pairs.md
python src/analysis/parser_audit.py outputs/smoke_optical_22pairs/records.jsonl --out figures/parser_audit.md
```

Runs are append-only JSONL (`outputs/<run>/records.jsonl`) keyed by a `run_id` hash of
(pair, variant, condition, seed, config hash); re-running resumes without duplicates.

## How the VCD path is verified (not assumed)

`src/eval/vcd_adapter.py` wraps the official code without modifying it:

1. `patch_status()` asserts `transformers.GenerationMixin.sample` *is* `vcd_utils.vcd_sample.sample`.
2. `prepare_inputs_for_generation_cd` (only reachable from the contrastive branch) is wrapped with a call
   counter; a `vcd` run with zero calls raises, and a `regular` run with nonzero calls raises. Every record
   stores `n_cd_forward_calls`.
3. `first_step_diagnostics()` recomputes clean vs. noised next-token logits and the VCD combination
   outside the generation loop and stores top-k tokens, L1 differences, and how many tokens survive APC.

Upstream issue found: `vcd_sample.py` returns `SampleDecoderOnlyOutput` without importing it (only hit with
`return_dict_in_generate=True`). The adapter injects the name; `third_party/` stays pristine.

Preprocessing choice: LLaVA-1.5's config says `image_aspect_ratio: pad`; the VCD eval script bypasses this
and centre-crops. We follow the model config (`process_images`) because Ebbinghaus renders are ~1.5:1 and a
centre crop would remove the outer circles. Recorded in each record's `generation_config`.

## Smoke-test headline (22 pairs, 3 seeds; pipeline check only)

From [figures/smoke_optical_22pairs.md](figures/smoke_optical_22pairs.md): regular sampling and full VCD
give near-identical results (pair success 0.258 vs 0.273; one CF correction, no regressions, over 66
pair-seed comparisons). Greedy never discriminates within a pair (pair success 0.000: it answers the same
way to both members of every pair). Under VCD the adaptive plausibility constraint leaves exactly **2**
tokens (`No`, `Yes`) alive at the first step on every image. 22 pairs is far too few to conclude anything;
these numbers exist to prove the pipeline, parser, and metrics code run end to end.

## Challenges hit so far (and what was done)

- **Python 3.13 vs transformers 4.31.0** — the machine default cannot build the VCD stack; solved with a
  dedicated Python 3.10 conda env and the cu118 torch 2.0.1 wheel (no cu121 build exists for 2.0.1).
- **Silent VCD no-op risk** — addressed by instrumentation (see above) rather than trusting outputs.
- **Upstream `NameError`** in `vcd_sample.py` when `return_dict_in_generate=True`; patched by name injection.
- **Seeding flaw found in the first smoke run** — re-seeding with the same value before every generation
  made `torch.multinomial` draw the same quantile for every image; with a two-token distribution, "seeds"
  43/44 answered `Yes` to all 44 images. Fixed with a per-item seed derived from (seed, pair, variant);
  the flawed run is kept in `outputs/…_SUPERSEDED_perrun_reseed/` for the record.
- **Common random numbers between conditions** — regular and VCD share `item_seed`; VCD's image noise is
  drawn on the CPU generator, the token draw on the CUDA generator, so the two conditions see the same
  sampling threshold. This makes the paired comparison isolate the distribution shift, and explains why
  65/66 parsed answers coincide.
- **Ebbinghaus canvases differ in width** between pair members (renderer-determined, e.g. 1164 vs 1170 px);
  flagged `auto_matched_dim_warn` in the manifest, pending manual QC. Ponzo `strength=18` counterfactuals
  have no canonical counterpart in the release and are excluded.
- **Released prompts differ from generator HEAD** (e.g. "two red circles" vs "two inner circles");
  the HF dataset revision is treated as the source of truth.

## Layout

```
configs/            run configs (generation + VCD hyperparameters)
data/inventory/     dataset inventory (tracked)
data/manifests/     versioned pair manifests (tracked)
data/images/        exported images (ignored; rebuilt by build_pairs.py)
src/data/           inventory + pair-manifest builders
src/eval/           vcd_adapter.py, run_inference.py, parse_answer.py
src/analysis/       summarize.py (metrics + transitions), parser_audit.py
tests/              parser, manifest, and synthetic-metrics tests
outputs/            raw JSONL (ignored except README)
figures/            tracked tables/figures
third_party/        VCD and VLMBias clones (ignored; pinned in VERSIONS.md)
```
