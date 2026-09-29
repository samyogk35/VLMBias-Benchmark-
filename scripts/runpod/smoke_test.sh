#!/usr/bin/env bash
# Gate: 2 pairs x {greedy, regular, vcd} x {canonical, counterfactual}, 1 seed. Proves CUDA + LLaVA + VCD branch.
set -euo pipefail
export HF_HOME=/workspace/hf CUDA_VISIBLE_DEVICES=0
source /workspace/venv/bin/activate
cd /workspace/Senior-seminar-/senior-seminar-VLMBias-Benchmark
python src/eval/run_inference.py --config configs/smoke_optical.yaml --run-name runpod_smoke_2pairs \
    --conditions greedy,regular,vcd --variants canonical,counterfactual --seeds 42 --limit-pairs 2 --diagnostics
python - <<'PY'
import json
rows=[json.loads(l) for l in open("outputs/runpod_smoke_2pairs/records.jsonl")]
print("records:", len(rows))
for r in rows:
    print(f'{r["condition"]:8s} {r["image_variant"]:15s} {r["pair_id"]:20s} cd_calls={r["n_cd_forward_calls"]} tok={r["n_new_tokens"]} {r["duration_ms"]:.0f}ms correct={r["is_correct"]} out={r["raw_output"]!r}')
vcd=[r for r in rows if r["condition"]=="vcd"]
assert vcd and all(r["n_cd_forward_calls"]>0 for r in vcd), "VCD branch never called!"
assert all(r["n_cd_forward_calls"]==0 for r in rows if r["condition"]!="vcd"), "non-vcd condition hit cd branch"
PY
echo SMOKE_OK
