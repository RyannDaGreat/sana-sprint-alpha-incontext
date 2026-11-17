#!/bin/bash
# CLI: ./run_diffusion.sh MATTE.png "caption" OUT_PREFIX [--checkpoint models/control/RUN/final_step_N] [--seed N]
# Writes OUT_PREFIX_gray.png (RGB on 50% gray) and OUT_PREFIX_rgba.png (RGBA cutout).
set -euo pipefail
MATTE=$(realpath "${1:?usage: ./run_diffusion.sh MATTE.png caption OUT_PREFIX}") ; shift
CAPTION=${1:?caption} ; shift
OUT=$(realpath -m "${1:?OUT_PREFIX}") ; shift
cd "$(dirname "$0")"
PYTHONPATH=src exec uv run python -m sana_alpha.inference "$MATTE" "$CAPTION" "$OUT" "$@"
