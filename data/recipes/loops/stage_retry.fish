#!/usr/bin/env fish
# Runs <loops> copies of until_empty_retry.fish for one processor, then touches <done_file>. CPU processors: no GPU choice
# Usage: fish stage_retry.fish <dtab> <processor> <loops> <done_file> [<upstream_done_file>]

for i in (seq $argv[3])
    fish (status dirname)/until_empty_retry.fish $argv[1] $argv[2] $argv[5..] &
end
wait
touch $argv[4]
