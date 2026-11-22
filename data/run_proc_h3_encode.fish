#!/usr/bin/env fish
# H3 Magic latents and text embeddings (manifest §22): h3_encode (VAE) and h3_embed (text encoder) workers, committed with Bluejay's commit.
# Safe to run again: each worker does only the rows that are not done yet. Workers stop when nothing new arrives for 2 polls
# Usage: fish run_proc_h3_encode.fish GPUS_ENCODE GPUS_EMBED   e.g. fish run_proc_h3_encode.fish 0,1,2,3,4 5
#        A GPU in both lists runs both. ENCODE_PER_GPU (default 2) VAE workers per encode GPU (~24 GB each)
#        ENCODE_JOBS (default magick,frames,video): the h3_encode jobs, in priority order, e.g. video alone
#        IDLE_ENCODE_SECONDS (default 180; 0 = off): an idle embed worker encodes videos meanwhile (it then runs until killed)
#        SHARD_LO, SHARD_HI (optional): 256-shard mode, this node encodes only byte shards SHARD_LO..SHARD_HI (manifest §22 "Remote node 2")

cd (status dirname); or exit
source h3_env.fish; or exit
set encode_gpus (string split , $argv[1])
set embed_gpus (string split , $argv[2])
set -q ENCODE_PER_GPU; or set ENCODE_PER_GPU 2
set -q ENCODE_JOBS; or set ENCODE_JOBS magick,frames,video
set -q IDLE_ENCODE_SECONDS; or set IDLE_ENCODE_SECONDS 180  # An embed worker with no text to embed encodes videos this long, then looks again
set shard_args
set -q SHARD_LO; and set shard_args --shard_lo $SHARD_LO --shard_hi $SHARD_HI

./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first

set log ../agents/.scratchpad/logs
set workers (math (count $encode_gpus) x $ENCODE_PER_GPU)
set worker 0
for gpu in $encode_gpus
    for k in (seq $ENCODE_PER_GPU)
        env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV python processors/h3_encode.py $worker $workers --jobs $ENCODE_JOBS $shard_args 2>&1 | tee $log/h3_encode_$worker.log &
        set worker (math $worker + 1)
    end
end
set worker 0
for gpu in $embed_gpus
    env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV python processors/h3_embed.py $worker (count $embed_gpus) --idle_encode_seconds $IDLE_ENCODE_SECONDS $shard_args 2>&1 | tee $log/h3_embed_$worker.log &
    set worker (math $worker + 1)
end
wait
