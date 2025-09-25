#!/usr/bin/env bash
set -euo pipefail

PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1 \
torchrun --nproc-per-node 8 \
  --log-dir logs \
  --redirects 3 --tee 3 \
  --max-restarts 0 \
  --standalone \
  train_gpt.py 