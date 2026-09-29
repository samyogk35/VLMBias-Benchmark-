#!/usr/bin/env bash
# Step 1: reproduce EnAR regular-vs-VCD on VLMBias main split. Pilot first, then full 928 at px768.
set -euo pipefail
export HF_HOME=/workspace/hf CUDA_VISIBLE_DEVICES=0
source /workspace/venv/bin/activate
cd /workspace/Senior-seminar-/senior-seminar-VLMBias-Benchmark
for px in 384 768 1152; do python src/data/build_benchmark_manifest.py --pixel $px; done
python src/eval/run_inference.py --config configs/vlmbias_main.yaml --run-name vlmbias_pilot_px768 \
    --conditions greedy,regular,vcd --variants counterfactual --smoke-only
python src/analysis/summarize_benchmark.py outputs/vlmbias_pilot_px768/records.jsonl --out figures/vlmbias_pilot_px768.md
echo PILOT_OK
python src/eval/run_inference.py --config configs/vlmbias_main.yaml --run-name vlmbias_main_px768 \
    --conditions greedy,regular,vcd --variants counterfactual
python src/analysis/summarize_benchmark.py outputs/vlmbias_main_px768/records.jsonl --out figures/vlmbias_main_px768.md
echo STEP1_PX768_OK
