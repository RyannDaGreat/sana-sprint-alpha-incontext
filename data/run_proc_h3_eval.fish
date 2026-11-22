#!/usr/bin/env fish
# H3 Magic eval inference (manifest §22): plans h3_eval_generations, then one h3_generate worker per GPU that keeps H3 loaded
# and fills every planned row of every h3_eval_models row. Safe to run again: only rows with no output run
# Usage: fish run_proc_h3_eval.fish GPU...   e.g. fish run_proc_h3_eval.fish 6 7

cd (status dirname); or exit
source h3_env.fish; or exit
set gpus $argv
test (count $gpus) -gt 0; or begin; echo "Usage: fish run_proc_h3_eval.fish GPU..." >&2; exit 1; end

./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first
$H3_ENV python processors/h3_generate.py plan; or exit

set log ../agents/.scratchpad/logs
set worker 0
for gpu in $gpus
    env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV python processors/h3_generate.py work $worker (count $gpus) 2>&1 | tee $log/h3_generate_$worker.log &
    set worker (math $worker + 1)
end
wait
