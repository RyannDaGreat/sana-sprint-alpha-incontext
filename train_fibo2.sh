#!/bin/bash
# FIBO2 trimask model (manifest §14), on all GPUs. Step 1: adapter. Step 2: control on a frozen adapter checkpoint.
# Usage: ./train_fibo2.sh adapter <run> [--flag value ...]
#        ./train_fibo2.sh control <run> --adapter models/fibo2/<adapter run>/step_XXXXXX [--flag value ...]
#        ./train_fibo2.sh splitter <run> --adapter ... --control models/fibo2/<control run>/step_XXXXXX [--flag value ...]   (manifest §17)
# NGPU=1 CUDA_VISIBLE_DEVICES=0 ./train_fibo2.sh ... runs on a subset (smoke tests).
# Output: models/fibo2/<run>/ (LoRA checkpoints every 500 steps, previews/); log: agents/.scratchpad/logs/train_fibo2_<run>.log
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:?usage: ./train_fibo2.sh adapter|control <run> [--flag value ...]}
RUN=${2:?usage: ./train_fibo2.sh adapter|control <run> [--flag value ...]}
shift 2
GPUS=${NGPU:-$(nvidia-smi --list-gpus | wc -l)}
PYTHONPATH=src exec uv run torchrun --standalone --nproc_per_node "$GPUS" -m sana_alpha.train_fibo2 train \
    --mode "$MODE" --run "$RUN" --rank 64 --save_every 500 --preview_every 500 "$@" \
    2>&1 | tee -a "agents/.scratchpad/logs/train_fibo2_$RUN.log"
