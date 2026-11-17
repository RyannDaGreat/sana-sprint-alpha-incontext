#!/usr/bin/env fish
# Runs one processor again and again until it has no work. Usage: fish until_empty.fish <dtab> <processor> [<upstream_done_file>]
# With <upstream_done_file>, no work means wait for the upstream processor, until that file exists
# A failure touches $FAIL_FLAG, because fish's wait does not return a child's exit status

set aviary (realpath (status dirname)/../..)/aviary.sh
set NO_WORK 20  # bluejay run exits with 20 when its query finds no rows
set WAIT_SECONDS 10  # Between tries while the upstream processor still runs

while true
    set -l upstream_done (test (count $argv) -lt 3; or test -e $argv[3]; and echo yes)  # Read before the run, so its last commit is seen
    $aviary bluejay run $argv[1] $argv[2]
    set -l code $status
    switch $code
        case 0
            continue
        case $NO_WORK
            test -n "$upstream_done"; and exit 0
            sleep $WAIT_SECONDS
        case '*'
            echo "FAILED: $argv[2] exited with $code" >&2
            touch $FAIL_FLAG
            exit $code
    end
end
