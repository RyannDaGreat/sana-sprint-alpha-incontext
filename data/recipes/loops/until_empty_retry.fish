#!/usr/bin/env fish
# Runs until_empty.fish again after a failure, for loops that must survive other agents' edits of the DTAB
# (a run that opens the project between a DTAB edit and its bluejay initialize fails). Every failure is printed
# Usage: fish until_empty_retry.fish <dtab> <processor> [<upstream_done_file>]

set RETRY_SECONDS 20  # Time for the editor to run bluejay initialize
while true
    fish (status dirname)/until_empty.fish $argv; and exit 0
    echo "RETRY: $argv[2] loop failed (error above); running it again in $RETRY_SECONDS s" >&2
    sleep $RETRY_SECONDS
end
