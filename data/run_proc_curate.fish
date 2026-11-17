#!/usr/bin/env fish
# Curates MAGICK into training latents: import the index, download the PNGs, split the alpha, encode the latents
# The three image stages run at the same time: each one waits for the rows of the one before it
# Safe to run again: each processor does only the rows that are not done yet

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set aviary ./aviary.sh
set stage recipes/loops/stage.fish

set DOWNLOAD_LOOPS 8   # Hugging Face allows 5,000 downloads per 5 minutes: more loops only wait longer
set SPLIT_LOOPS    16  # CPU only
set ENCODE_LOOPS   (math 2 x (nvidia-smi -L | count))  # Two per GPU: one loads its batch while the other encodes

set -gx FAIL_FLAG (mktemp -u)
set done (mktemp -d)  # stage.fish touches done/<processor> when all its loops end

$aviary bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

fish recipes/loops/until_empty.fish $dtab import_index; or exit

fish $stage $dtab download_image $DOWNLOAD_LOOPS $done/download_image &
fish $stage $dtab split_alpha    $SPLIT_LOOPS    $done/split_alpha    $done/download_image &
fish $stage $dtab encode_latents $ENCODE_LOOPS   $done/encode_latents $done/split_alpha &
wait

test -e $FAIL_FLAG; or exit 0
echo "A loop failed. Its error is above" >&2
exit 1
