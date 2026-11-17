#!/usr/bin/env fish
# Generates the eval images of every eval_models row: one model server and one generate_eval loop per GPU
# The servers keep their models loaded between runs, and stop when the loops end. Safe to run again: only rows not done yet run
# Usage: fish run_proc_eval.fish [GPU...]    (default: every GPU)

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set aviary ./aviary.sh
set loop recipes/loops/until_empty.fish
set eval_py env -u UV_PROJECT_ENVIRONMENT PYTHONPATH=../src uv run --project .. python processors/generate_eval.py

set gpus $argv
test (count $gpus) -gt 0; or set gpus (seq 0 (math (nvidia-smi -L | count) - 1))

set -gx FAIL_FLAG (mktemp -u)

$aviary bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

fish $loop $dtab plan_eval; or exit

set servers
for gpu in $gpus
    set -a servers (CUDA_VISIBLE_DEVICES=$gpu $eval_py start); or exit
    CUDA_VISIBLE_DEVICES=$gpu fish $loop $dtab generate_eval &
end
wait
kill $servers  # frees the GPUs for training
test -e $FAIL_FLAG; or exit 0
echo "A generate_eval loop failed. Its error is above" >&2
exit 1
