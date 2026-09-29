#!/usr/bin/env bash
# One-time pod setup. Everything lives under /workspace (persistent disk).
set -euo pipefail
export HF_HOME=/workspace/hf
export PIP_CACHE_DIR=/workspace/pip-cache
mkdir -p /workspace/hf /workspace/pip-cache

cd /workspace
# repo is private: cloned from a git bundle scp'd to /workspace/repo.bundle
if [ ! -d Senior-seminar- ]; then
  git clone -b main /workspace/repo.bundle Senior-seminar-
else
  git -C Senior-seminar- pull /workspace/repo.bundle main
fi
cd Senior-seminar-/senior-seminar-VLMBias-Benchmark
git rev-parse --short HEAD

# venv instead of conda (base image already python 3.10)
if [ ! -x /workspace/venv/bin/python ]; then
  python3 -m venv /workspace/venv
fi
source /workspace/venv/bin/activate
python --version
pip install --upgrade "pip<25" -q
pip install -r requirements.txt -q
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"

bash scripts/clone_third_party.sh
python scripts/download_assets.py
python src/data/build_inventory.py
python src/data/build_pairs.py
python -m pytest -q tests
echo BOOTSTRAP_OK
