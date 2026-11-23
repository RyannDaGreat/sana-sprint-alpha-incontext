#!/usr/bin/env fish
# Is H3's encoder the bottleneck? (manifest §25): one h3_enc_opt.py worker on one GPU, committed with Bluejay's commit.
# Safe to run again: a worker does only the samples its run has no result for yet
# Usage: fish run_proc_h3_encoder.fish GPU COMMAND ARGS...   e.g. fish run_proc_h3_encoder.fish 6 images l1_vgg --objective l1_vgg
#        GPU - (a dash) runs on the CPU, for register

cd (status dirname); or exit
source h3_env.fish; or exit
test (count $argv) -ge 2; or begin; echo "Usage: fish run_proc_h3_encoder.fish GPU COMMAND ARGS..." >&2; exit 1; end
set gpu $argv[1]
test $gpu = -; and set gpu ''

./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first
set log ../agents/.scratchpad/logs
set name (string join _ h3_enc $argv[2..3])
env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV --with lpips --with DISTS_pytorch --with pytorch-msssim python processors/h3_enc_opt.py $argv[2..] 2>&1 | tee $log/$name.log
