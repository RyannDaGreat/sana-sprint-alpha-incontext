#!/usr/bin/env fish
# Runs h3_choose_clip over every cut clip every few minutes while clips are cut (a provisional train set, so training starts
# early), then once more after the last clip is cut (H3_CHOOSE_FINAL=true: exactly H3_PEXELS_TRAIN train). Manifest §22
# Usage: fish h3_choose_loop.fish <dtab> <cut_done_file>

set aviary (realpath (status dirname)/../..)/aviary.sh
set PERIOD_SECONDS 180  # The training loader reads new train rows every 50 steps
set RETRY_SECONDS 20
set unclaim "SELECT h3_choose_clip_claim AS claim_id FROM h3_pexels_clips WHERE h3_choose_clip_claim IS NOT NULL"

function choose_once --argument-names dtab
    # Command. Unclaims every row, then runs h3_choose_clip once over every cut clip. Returns its status
    $aviary bluejay unclaim $dtab $unclaim | string match -r 'Unclaimed [1-9].*'
    $aviary bluejay run $dtab h3_choose_clip
end

while not test -e $argv[2]
    choose_once $argv[1]; or echo "h3_choose_clip failed (error above); next try in $PERIOD_SECONDS s" >&2
    sleep $PERIOD_SECONDS
end
set -gx H3_CHOOSE_FINAL true
while not choose_once $argv[1]
    echo "final h3_choose_clip failed (error above); again in $RETRY_SECONDS s" >&2
    sleep $RETRY_SECONDS
end
echo "h3_choose_clip: final choice done"
