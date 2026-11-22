#!/usr/bin/env fish
# H3 Magic latents and text embeddings (manifest §22): h3_encode (VAE) and h3_embed (text encoder) workers, committed with Bluejay's commit.
# Safe to run again: each worker does only the rows that are not done yet. Workers stop when nothing new arrives for 2 polls
# Usage: fish run_proc_h3_encode.fish GPUS_ENCODE GPUS_EMBED   e.g. fish run_proc_h3_encode.fish 0,1,2,3,4 5
#        A GPU in both lists runs both. ENCODE_PER_GPU (default 2) VAE workers per encode GPU (~24 GB each)
#        ENCODE_JOBS (default magick,frames,video): the h3_encode jobs, in priority order, e.g. video alone

cd (status dirname); or exit
source h3_env.fish; or exit
set encode_gpus (string split , $argv[1])
set embed_gpus (string split , $argv[2])
set -q ENCODE_PER_GPU; or set ENCODE_PER_GPU 2
set -q ENCODE_JOBS; or set ENCODE_JOBS magick,frames,video

./aviary.sh bluejay initialize project.dtab; or exit  # By convention, every launch script initializes first

set log ../agents/.scratchpad/logs
set workers (math (count $encode_gpus) x $ENCODE_PER_GPU)
set worker 0
for gpu in $encode_gpus
    for k in (seq $ENCODE_PER_GPU)
        env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV python processors/h3_encode.py $worker $workers --jobs $ENCODE_JOBS 2>&1 | tee $log/h3_encode_$worker.log &
        set worker (math $worker + 1)
    end
end
set worker 0
for gpu in $embed_gpus
    env CUDA_VISIBLE_DEVICES=$gpu $H3_ENV python processors/h3_embed.py $worker (count $embed_gpus) 2>&1 | tee $log/h3_embed_$worker.log &
    set worker (math $worker + 1)
end
wait
