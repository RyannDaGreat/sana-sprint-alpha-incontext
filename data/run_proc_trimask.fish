#!/usr/bin/env fish
# Fills trimask_augmentations (manifest §13): the ten trimask_<method> processors (CPU) and encode_trimask (GPU) run at the same time
# encode_trimask waits for rows until all ten methods are done. Safe to run again: each processor does only the rows that are not done yet
# Usage: fish run_proc_trimask.fish

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set stage recipes/loops/stage.fish

set METHODS (string replace -rf '^\ttrimask_(\w+)\t.*' '$1' < $dtab)  # The ten trimask_<method> processors of the dtab
set AUGMENT_LOOPS 2  # Per method. Each run uses 8 threads (trimask_augment.WORKERS). 4 per method saturated the 96 CPUs and starved encode_trimask
set ENCODE_LOOPS  (math 2 x (nvidia-smi -L | count))  # Two per GPU: one loads its batch while the other encodes

set -gx FAIL_FLAG (mktemp -u)
set done (mktemp -d)  # stage.fish touches done/<processor> when all its loops end

./aviary.sh bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

for method in $METHODS
    fish $stage $dtab trimask_$method $AUGMENT_LOOPS $done/trimask_$method &
end
fish -c 'for file in $argv[2..]; while not test -e $file; sleep 10; end; end; touch $argv[1]' $done/augment $done/trimask_$METHODS &
fish $stage $dtab encode_trimask $ENCODE_LOOPS $done/encode_trimask $done/augment &
wait

test -e $FAIL_FLAG; or exit 0
echo "A loop failed. Its error is above" >&2
exit 1
