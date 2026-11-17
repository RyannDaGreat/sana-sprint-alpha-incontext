#!/usr/bin/env fish
# Curates MAGICK into training latents: import the index, download the PNGs, split the alpha, encode the latents
# Safe to run again: each processor does only the rows that are not done yet

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set aviary ./aviary.sh
set loop recipes/loops/until_empty.fish

set DOWNLOAD_LOOPS 32  # Parallel downloads from Hugging Face. Not tuned
set SPLIT_LOOPS    32  # CPU only. Not tuned
set GPUS (nvidia-smi -L | count)  # encode_latents runs one loop per GPU

set -gx FAIL_FLAG (mktemp -u)

function wait_or_exit --argument-names processor
    wait
    test -e $FAIL_FLAG; or return 0
    echo "A $processor loop failed. Its error is above" >&2
    exit 1
end

$aviary bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

fish $loop $dtab import_index; or exit

for i in (seq $DOWNLOAD_LOOPS); fish $loop $dtab download_image &; end
wait_or_exit download_image

for i in (seq $SPLIT_LOOPS); fish $loop $dtab split_alpha &; end
wait_or_exit split_alpha

for gpu in (seq 0 (math $GPUS - 1)); CUDA_VISIBLE_DEVICES=$gpu fish $loop $dtab encode_latents &; end
wait_or_exit encode_latents
