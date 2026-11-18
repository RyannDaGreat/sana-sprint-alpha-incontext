#!/usr/bin/env fish
# Compacts every table in the DTAB with BlueJay's compaction API, one at a time, then waits one hour, forever (as MulticamDailies does). Commits go on meanwhile

set PAUSE_S 3600  # One hour between passes
cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set tables (awk '/^tables$/ { inside = 1; next } /^[^\t]/ { inside = 0 } inside && /^\t[^\t ]+$/ { print substr($0, 2) }' $dtab)
./aviary.sh bluejay initialize $dtab; or exit  # By convention, every launch script initializes first
while true
    for table in $tables
        echo (date '+%F %T') compact $table
        ./aviary.sh bluejay database $dtab compact $table; or echo "FAILED: compact $table exited $status" >&2
    end
    echo (date '+%F %T') pass done. The next pass starts in one hour
    sleep $PAUSE_S
end
