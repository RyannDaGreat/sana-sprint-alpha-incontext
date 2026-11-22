#!/usr/bin/env fish
# H3 Magic data (manifest §22): MAGICK images + Pexels clips at 512 x 512, and their H3-style captions by Gemini. No GPU
# The stages run at the same time: each waits for the rows of the one before it. Captions never wait for the other source
# Pexels: import every h3_import_jobs slice, cut every clip; h3_choose_clip rewrites split every 3 minutes (the best share of
# the cut clips is train) and once more after the cut (exactly H3_PEXELS_TRAIN); train and val clips are captioned as chosen
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
fish recipes/loops/h3_choose_loop.fish $dtab $done/h3_cut_clip &  # provisional train set every 3 min, final after the cut
fish $stage $dtab h3_caption_clip $CAPTION_LOOPS $done/h3_caption_clip $done/h3_choose_final &
while not test -e $done/h3_cut_clip; or pgrep -f "h3_choose_loop.fish $dtab $done/h3_cut_clip" > /dev/null
    sleep 10
end
touch $done/h3_choose_final  # the caption loops end once the final train set is captioned
wait
