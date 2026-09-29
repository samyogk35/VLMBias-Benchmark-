#!/usr/bin/env bash
# Resume the full px768 run after a crash (run_inference skips finished run_ids), then summarize.
set -euo pipefail
export HF_HOME=/workspace/hf CUDA_VISIBLE_DEVICES=0
source /workspace/venv/bin/activate
cd /workspace/Senior-seminar-/senior-seminar-VLMBias-Benchmark
python src/eval/run_inference.py --config configs/vlmbias_main.yaml --run-name vlmbias_main_px768 \
    --conditions greedy,regular,vcd --variants counterfactual
python src/analysis/summarize_benchmark.py outputs/vlmbias_main_px768/records.jsonl --out figures/vlmbias_main_px768.md
echo STEP1_PX768_OK
