#!/bin/bash
# Browser app (WebGPU): builds web/ into journal/webgpu/ and serves the journal + models/web/ over HTTPS.
# Open https://HOST:PORT/webgpu/ (self-signed certificate: accept it once). Usage: ./run_web.sh [PORT]
set -euo pipefail
cd "$(dirname "$0")/web"
npm install --no-audit --no-fund
npm run build
exec node serve.js "$@"
