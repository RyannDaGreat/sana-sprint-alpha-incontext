#!/usr/bin/env fish
# Runs one processor again and again until it has no work. Usage: fish until_empty.fish <dtab> <processor>
# A failure touches $FAIL_FLAG, because fish's wait does not return a child's exit status

set aviary (realpath (status dirname)/../..)/aviary.sh
set NO_WORK 20  # bluejay run exits with 20 when its query finds no rows

while true
    $aviary bluejay run $argv[1] $argv[2]
    set -l code $status
    switch $code
        case 0
            continue
        case $NO_WORK
            exit 0
        case '*'
            echo "FAILED: $argv[2] exited with $code" >&2
            touch $FAIL_FLAG
            exit $code
    end
end
