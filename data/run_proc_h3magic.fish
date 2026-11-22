#!/usr/bin/env fish
# H3 Magic data (manifest §22): MAGICK images + Pexels clips at 512 x 512, and their H3-style captions by Gemini. No GPU
# The stages run at the same time: each waits for the rows of the one before it. Captions never wait for the other source
# Pexels: import every h3_import_jobs slice, cut every clip, caption the clips certainly in the train set while cutting,
# then h3_choose_clip picks the train and val clips, then the chosen clips not yet captioned are captioned
# Loops run again after a failure (until_empty_retry.fish): other agents edit the DTAB while these run
# Safe to run again: each processor does only the rows that are not done yet
# Usage: fish run_proc_h3magic.fish   (tmux h3magic_data)

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set aviary ./aviary.sh
set retry recipes/loops/until_empty_retry.fish
set stage recipes/loops/stage_retry.fish

set IMAGE_LOOPS   8   # CPU: 256 images per run, ~20 s
set CUT_LOOPS     48  # CPU: one 4K clip per loop; decoding is most of the cost
set CAPTION_LOOPS 8   # Per source. Each run has H3_CAPTION_BATCH_SIZE Gemini calls in flight: 64 x 8 = 512 per source

set -gx FAIL_FLAG (mktemp -u)
set done (mktemp -d)

$aviary bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

fish $stage $dtab h3_magick_image   $IMAGE_LOOPS   $done/h3_magick_image &
fish $stage $dtab h3_caption_magick $CAPTION_LOOPS $done/h3_caption_magick $done/h3_magick_image &

fish $retry $dtab h3_import; or exit
fish $stage $dtab h3_cut_clip     $CUT_LOOPS     $done/h3_cut_clip &
fish $stage $dtab h3_caption_clip $CAPTION_LOOPS $done/h3_caption_clip $done/h3_cut_clip &
while not test -e $done/h3_caption_clip
    sleep 10
end
fish $retry $dtab h3_choose_clip; or exit
fish $stage $dtab h3_caption_clip $CAPTION_LOOPS $done/h3_caption_chosen &
wait
