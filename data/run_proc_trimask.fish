#!/usr/bin/env fish
# Fills trimask_augmentations (manifest §13): every trimask_<method> processor (CPU, nice) and one encode_multires worker per GPU
# run at the same time. The workers encode rows as soon as they commit, and stop once augmentation is done and nothing is left.
# Safe to run again: each processor and worker does only the rows that are not done yet
# Usage: fish run_proc_trimask.fish

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set stage recipes/loops/stage.fish

set METHODS (string replace -rf '^\ttrimask_(\w+)\t.*' '$1' < $dtab)  # Every trimask_<method> processor of the dtab
set AUGMENT_LOOPS 1  # Per method. Each run uses 8 threads (trimask_augment.WORKERS). More starved the GPU workers of CPU
set GPUS (nvidia-smi -L | count)
set WORKER_ENV uv run --no-project --python 3.13 --index https://pypi.netflix.net/simple --default-index https://pypi.org/simple \
    --with (ls Aviary/.wheel/*.whl) --with numpy --with pillow --with opencv-python-headless --with fire==0.7.1 \
    --with torch==2.15.0 --with diffusers==0.43.0 --with safetensors==0.8.0 --with accelerate==1.10.1  # Bluejay needs Python 3.13

set -gx FAIL_FLAG (mktemp -u)
set done (mktemp -d)  # stage.fish touches done/<processor> when all its loops end

./aviary.sh bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

for method in $METHODS
    nice -n 10 fish $stage $dtab trimask_$method $AUGMENT_LOOPS $done/trimask_$method &
end
fish -c 'for file in $argv[2..]; while not test -e $file; sleep 10; end; end; touch $argv[1]' $done/augment $done/trimask_$METHODS &
for gpu in (seq 0 (math $GPUS - 1))
    env CUDA_VISIBLE_DEVICES=$gpu $WORKER_ENV python processors/encode_multires.py $gpu $GPUS $done/augment &  # A failed worker prints its error
end
wait

test -e $FAIL_FLAG; or exit 0
echo "A loop or worker failed. Its error is above" >&2
exit 1
