#!/bin/bash
# Step 2: full fine-tune of the SANA-Sprint teacher on MAGICK gray composites (8 GPUs).
# Usage: ./train_adapter.sh <run_name> [extra --flags for src/sana_alpha/train_adapter.py]
# Output: models/adapter/<run_name>/ (fp32 checkpoints, samples/); log: agents/.scratchpad/logs/train_adapter_<run_name>.log
# Re-running the same command resumes from models/adapter/<run_name>/resume.pt.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:?usage: ./train_adapter.sh <run_name> [--flag value ...]}
shift
GPUS=$(nvidia-smi --list-gpus | wc -l)
PYTHONPATH=src exec uv run torchrun --standalone --nproc_per_node "$GPUS" -m sana_alpha.train_adapter \
    --run "$RUN" --source lance --batch_size 32 --grad_checkpointing True --lr 2e-5 --warmup_steps 250 \
    --max_steps 3000 --log_every 10 --sample_every 250 --save_every 500 "$@"
