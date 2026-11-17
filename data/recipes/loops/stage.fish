#!/usr/bin/env fish
# Runs <loops> copies of until_empty.fish for one processor, then touches <done_file>. Loop i sees only GPU i mod the GPU count
# Usage: fish stage.fish <dtab> <processor> <loops> <done_file> [<upstream_done_file>]

set gpus (nvidia-smi -L | count)
for i in (seq 0 (math $argv[3] - 1))
    CUDA_VISIBLE_DEVICES=(math $i % $gpus) fish (status dirname)/until_empty.fish $argv[1] $argv[2] $argv[5..] &
end
wait
touch $argv[4]
