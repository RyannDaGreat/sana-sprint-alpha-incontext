#!/bin/bash
# Web UI: draw or upload a matte, type a caption, get the image. Usage: ./run_server.sh [--checkpoint models/control/RUN/final_step_N] [--port P]
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=src exec uv run python -m sana_alpha.server "$@"
