#!/usr/bin/env fish
# DiffDec (manifest §26) into Bluejay (m00998: every experiment goes through BlueJay). No GPU
# dd_find finds DD_DIR's test set and candidates (folders with genome.json); then dd_import_samples imports the test set into dd_samples,
# and dd_import_renders imports each scored candidate's renders into dd_renders
# Run it again after new candidates are scored. A test set or candidate whose files changed, or whose import failed, is imported again
# Run one copy at a time: it unclaims every row whose import is not done
# Usage: fish run_proc_diffdec.fish   (log: tee to agents/.scratchpad/logs/diffdec_import.log, as below)

set self (realpath (status filename))
cd (status dirname); or exit
mkdir -p ../agents/.scratchpad/logs
set log ../agents/.scratchpad/logs/diffdec_import.log
if not set -q DD_LOGGED
    # Re-run this script with its output also in the log: the terminal shows everything
    set -gx DD_LOGGED 1
    fish $self $argv 2>&1 | tee -a $log
    exit $pipestatus[1]
end

source uv.env; or exit
set dtab project.dtab
set aviary ./aviary.sh
set loop recipes/loops/until_empty.fish

set RENDER_LOOPS 4  # CPU: one candidate per run: 60 error maps

set -gx FAIL_FLAG (mktemp -u)

echo "== diffdec import "(date)
$aviary bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

# dd_find runs every time, so it finds new candidates and the test set: release its one job row first
$aviary bluejay unclaim $dtab "SELECT dd_find_claim AS claim_id FROM dd_jobs WHERE dd_find_claim IS NOT NULL"; or exit
$aviary bluejay run $dtab dd_find; or exit
# Import again the test set that changed after its import, or that has no import (a failed or killed run)
$aviary bluejay unclaim $dtab "SELECT dd_import_samples_claim AS claim_id FROM dd_jobs WHERE dd_import_samples_claim IS NOT NULL AND imported_samples_stamp IS DISTINCT FROM samples_stamp"; or exit
$aviary bluejay run $dtab dd_import_samples
set code $status
test $code -eq 0 -o $code -eq 20; or exit $code  # 20 = no work: the test set is already imported
# Same for each candidate: its renders changed after the import, or it has no import
$aviary bluejay unclaim $dtab "SELECT dd_import_renders_claim AS claim_id FROM dd_methods WHERE dd_import_renders_claim IS NOT NULL AND renders_sha256 IS DISTINCT FROM results_sha256"; or exit

for i in (seq $RENDER_LOOPS)
    fish $loop $dtab dd_import_renders &
end
wait

test -e $FAIL_FLAG; or exit 0
echo "A loop failed. Its error is above" >&2
exit 1
