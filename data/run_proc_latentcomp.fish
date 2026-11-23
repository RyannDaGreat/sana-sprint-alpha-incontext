#!/usr/bin/env fish
# Latent Compositor (manifest §24) into Bluejay (m00998: every experiment goes through BlueJay). No GPU
# lc_find finds LC_DIR's dataset shards and scored candidates (folders with results.json); then lc_import_shard imports each shard
# into lc_samples, and lc_import_renders imports each candidate's decoded test samples into lc_renders
# Run it again after new candidates are scored. A shard or candidate whose files changed, or whose import failed, is imported again
# Run one copy at a time: it unclaims every row whose import is not done
# Usage: fish run_proc_latentcomp.fish   (logs: tee to agents/.scratchpad/logs/)

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set aviary ./aviary.sh
set loop recipes/loops/until_empty.fish

set SHARD_LOOPS  8  # CPU: one shard per run; a test shard encodes 500 PNGs, a train shard only cuts its latents into rows
set RENDER_LOOPS 4  # CPU: one candidate per run: 96 PNG copies and 96 error maps

set -gx FAIL_FLAG (mktemp -u)

$aviary bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

# lc_find runs every time, so it finds new candidates: release its one job row first
$aviary bluejay unclaim $dtab "SELECT lc_find_claim AS claim_id FROM lc_jobs WHERE lc_find_claim IS NOT NULL"; or exit
$aviary bluejay run $dtab lc_find; or exit
# Import again each shard and candidate that changed after its import, or that has no import (a failed or killed run)
$aviary bluejay unclaim $dtab "SELECT lc_import_shard_claim AS claim_id FROM lc_shards WHERE lc_import_shard_claim IS NOT NULL AND imported_stamp IS DISTINCT FROM files_stamp"; or exit
$aviary bluejay unclaim $dtab "SELECT lc_import_renders_claim AS claim_id FROM lc_methods WHERE lc_import_renders_claim IS NOT NULL AND renders_sha256 IS DISTINCT FROM results_sha256"; or exit

for i in (seq $SHARD_LOOPS)
    fish $loop $dtab lc_import_shard &
end
for i in (seq $RENDER_LOOPS)
    fish $loop $dtab lc_import_renders &
end
wait

test -e $FAIL_FLAG; or exit 0
echo "A loop failed. Its error is above" >&2
exit 1
