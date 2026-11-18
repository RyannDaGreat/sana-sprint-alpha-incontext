#!/bin/bash
# Step 3: GateControl matte control (LoRA + gates) on top of a frozen adapter checkpoint (8 GPUs).
# Usage: ./train_control.sh <run_name> <adapter_step_dir> [extra --flags for src/sana_alpha/train_control.py]
# NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 ./train_control.sh ... trains on a subset (default: all GPUs).
# Output: models/control/<run_name>/ (checkpoints, previews/); log: agents/.scratchpad/logs/train_control_<run_name>.log
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:?usage: ./train_control.sh <run_name> <adapter_step_dir> [--flag value ...]}
ADAPTER=${2:?usage: ./train_control.sh <run_name> <adapter_step_dir> [--flag value ...]}
shift 2
GPUS=${NGPU:-$(nvidia-smi --list-gpus | wc -l)}
PYTHONPATH=src exec uv run torchrun --standalone --nproc_per_node "$GPUS" -m sana_alpha.train_control train \
    --run "$RUN" --data lance --adapter "$ADAPTER" --rank 64 --batch_size 16 --grad_checkpointing True \
    --max_steps 4000 --warmup 500 --save_every 500 --preview_every 250 "$@"
