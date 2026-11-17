#!/usr/bin/env fish
# Encodes every MAGICK image on its own random solid color (encode_colorbg, adapter_v2's data). Safe to run again
# Usage: fish run_proc_colorbg.fish [loops per GPU, default 1: more slows the training on the same GPUs 4x]

cd (status dirname); or exit
source uv.env; or exit
set -gx FAIL_FLAG (mktemp -u)
./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first
fish recipes/loops/stage.fish project.dtab encode_colorbg (math (set -q argv[1]; and echo $argv[1]; or echo 1) x (nvidia-smi -L | count)) (mktemp -u)
not test -e $FAIL_FLAG; or begin; echo "A loop failed. Its error is above" >&2; exit 1; end
