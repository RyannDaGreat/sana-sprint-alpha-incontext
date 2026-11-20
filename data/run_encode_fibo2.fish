#!/usr/bin/env fish
# Encodes the FIBO2 latents (manifest §14): two encode_fibo2 workers per GPU, magick_samples first, then trimask_augmentations.
# Safe to run again: each worker does only the rows that are not done yet
# Usage: fish run_encode_fibo2.fish [JOBS]   (JOBS: comma-separated encode_fibo2 jobs, default magick,trimask; split = the
#        splitter LoRA's white and black targets, manifest §17)

cd (status dirname); or exit
source uv.env; or exit
set dtab project.dtab
set GPUS (nvidia-smi -L | count)
set JOBS magick,trimask
set -q argv[1]; and set JOBS $argv[1]
set WORKER_ENV uv run --no-project --python 3.13 --index https://pypi.netflix.net/simple --default-index https://pypi.org/simple \
    --with (ls Aviary/.wheel/*.whl) --with numpy --with pillow --with opencv-python-headless --with fire==0.7.1 \
    --with torch==2.15.0 --with diffusers==0.43.0 --with safetensors==0.8.0 --with accelerate==1.10.1  # Bluejay needs Python 3.13

./aviary.sh bluejay initialize $dtab; or exit  # By convention, every launch script initializes first

set WORKERS (math 2 x $GPUS)  # Two workers per GPU: one decodes on the CPU while the other encodes
set failed 0
for worker in (seq 0 (math $WORKERS - 1))
    env CUDA_VISIBLE_DEVICES=(math $worker % $GPUS) $WORKER_ENV python processors/encode_fibo2.py $worker $WORKERS --jobs $JOBS &
end
for job in (jobs -p)
    wait $job; or set failed 1
end
test $failed = 0; and exit 0
echo "A worker failed. Its error is above" >&2
exit 1
