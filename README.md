# Does VCD actually make LLaVA look at the image?

Senior seminar project, fall 2026.

VLMBias (ICLR 2026) shows that vision-language models answer from memory instead of from the image:
show LLaVA a dog with five legs and it says four. The EnAR paper (CVPR 2026) reports that Visual
Contrastive Decoding (VCD) lifts LLaVA-1.5-7B on VLMBias from 16.92% to 19.18%. My question is whether
that gain means the model is really more sensitive to what's in the image, or whether VCD just makes it
less likely to give the "familiar" answer, which happens to score better on a benchmark where the
familiar answer is always wrong.

To tell those apart I evaluate on *pairs*: a canonical image (familiar answer is correct) and a
minimally changed counterfactual (familiar answer is wrong), same prompt, same rendering. A decoder that
sees better should get both right more often. A decoder that just avoids the familiar answer will gain
on the counterfactual and lose on the canonical.

The full plan is in SPEC.md, the schedule in TIMELINE.md, and every pinned version in VERSIONS.md.

## Where things are (Sept 12)

Environment works (python 3.10, torch 2.0.1, transformers 4.31.0 - the versions VCD's code needs).
Model, CLIP tower and dataset are downloaded at fixed revisions. I built an inventory of the VLMBias
main split (2784 rows, which is really 464 cases x 3 resolutions x 2 prompt wordings) and a first pair
manifest: 60 optical illusion pairs from images already in VLMBias, 22 distinct canonicals. LLaVA runs,
VCD runs, and I checked that the VCD branch is really executing rather than silently falling back to
plain sampling. There is a 22-pair smoke run (greedy / regular sampling / VCD, 3 seeds) in
figures/smoke_optical_22pairs.md. It's a pipeline check, not a result.

## Setup

```bash
bash scripts/setup_env.sh              # conda env "vcd"
conda activate vcd
bash scripts/clone_third_party.sh      # VCD + VLMBias generators -> third_party/
python scripts/download_assets.py      # ~16 GB into the HF cache
python src/data/build_inventory.py
python src/data/build_pairs.py         # manifest + exports the pair images
python -m pytest -q tests
```

Running:

```bash
export CUDA_VISIBLE_DEVICES=0
python src/eval/run_inference.py --config configs/smoke_optical.yaml --run-name smoke_optical_22pairs \
    --conditions greedy,regular,vcd --variants canonical,counterfactual --smoke-only
python src/analysis/summarize.py outputs/smoke_optical_22pairs/records.jsonl --out figures/smoke_optical_22pairs.md
python src/analysis/parser_audit.py outputs/smoke_optical_22pairs/records.jsonl --out figures/parser_audit.md
```

Every generation is one line in `outputs/<run>/records.jsonl`. Re-running the same run name picks up
where it left off.

## Checking VCD really runs

VCD works by monkeypatching `transformers`' sampling loop, and if the patch doesn't attach you just
get plain sampling with no error. So in `src/eval/vcd_adapter.py` I (1) check that
`GenerationMixin.sample` is VCD's function, (2) count how many times the noisy-image branch is called
during each generation and raise if a vcd run has zero calls, and (3) for a few examples compute the
clean and noisy logits by hand to see that the contrastive term is nonzero. Every record stores the
call count. On the smoke run every vcd generation had 2 calls (one per token) and took twice as long.



## Layout

```
configs/          run configs
data/inventory/   dataset inventory
data/manifests/   pair manifests
data/images/      exported images (gitignored)
src/data/         inventory + manifest builders
src/eval/         vcd_adapter.py, run_inference.py, parse_answer.py
src/analysis/     summarize.py, parser_audit.py
tests/
outputs/          raw jsonl (gitignored)
figures/          tables that get tracked
third_party/      VCD and VLMBias clones (gitignored, commits in VERSIONS.md)
```
