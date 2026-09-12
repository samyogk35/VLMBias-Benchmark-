#!/usr/bin/env bash
# Create the pinned Python 3.10 conda env "vcd" and install the VCD-compatible stack.
set -euo pipefail
cd "$(dirname "$0")/.."
conda create -y -n vcd python=3.10
conda run -n vcd --no-capture-output pip install --upgrade "pip<25"
conda run -n vcd --no-capture-output pip install -r requirements.txt
conda run -n vcd --no-capture-output python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__)"
