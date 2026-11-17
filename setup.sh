#!/usr/bin/env bash
# Recreates what this dump needs outside git after a container reboot. Safe to run again
# - fish: the data/ launch scripts are fish
# - data/Aviary: corp/Aviary branch bluejay2 (gitignored), built into one wheel that data/aviary.sh runs
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

AVIARY_REPO=https://github.netflix.net/corp/Aviary
AVIARY_BRANCH=bluejay2
WHEEL_DIR=data/Aviary/.wheel  # data/aviary.sh runs the one wheel in this folder

command -v fish || { apt-get update && apt-get install -y fish; }

test -d data/Aviary || git clone -b $AVIARY_BRANCH $AVIARY_REPO data/Aviary

# Aviary's install_local.sh builds the same wheel, but installs it as a uv tool outside the dump
source ~/.nvm/nvm.sh  # npm builds Sparrow and Albatross
commit=$(git -C data/Aviary rev-parse --short HEAD)
if ! ls $WHEEL_DIR/*+$commit-*.whl; then
    rm -rf $WHEEL_DIR
    ( source data/Aviary/distribution/stage.sh
      uv version --project "$distribution" --frozen "0.0.0+$commit"
      uv build --wheel --out-dir "$PWD/$WHEEL_DIR" --index https://pypi.netflix.net/simple --default-index https://pypi.org/simple "$distribution" )
fi
data/aviary.sh --version
