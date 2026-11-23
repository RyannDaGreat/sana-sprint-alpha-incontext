#!/usr/bin/env fish
# Is H3's encoder the bottleneck? (manifest §25): one worker on one GPU, committed with Bluejay's commit.
# COMMAND register / images / video runs h3_enc_opt.py; train / evaluate runs h3_enc_train.py.
# Safe to run again: a worker does only the samples its run has no result for yet
# Usage: fish run_proc_h3_encoder.fish GPU COMMAND ARGS...   e.g. fish run_proc_h3_encoder.fish 6 images img_l1vgg --objective l1_vgg
#        GPU - (a dash) runs on the CPU, for register

cd (status dirname); or exit
source h3_env.fish; or exit
test (count $argv) -ge 2; or begin; echo "Usage: fish run_proc_h3_encoder.fish GPU COMMAND ARGS..." >&2; exit 1; end
set gpu $argv[1]
test $gpu = -; and set gpu ''
set script processors/h3_enc_opt.py
contains -- $argv[2] train evaluate; and set script processors/h3_enc_train.py

./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first
set log ../agents/.scratchpad/logs
set name h3_enc_$argv[2]
test (count $argv) -ge 3; and not string match -q -- '-*' $argv[3]; and set name $name"_"$argv[3]  # the run name, when given
env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV --with lpips --with DISTS_pytorch --with pytorch-msssim python $script $argv[2..] 2>&1 | tee $log/$name.log
