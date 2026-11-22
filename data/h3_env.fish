# Sourced by the H3 Magic worker launchers (manifest §22): the one Python environment of h3_encode, h3_embed and h3_generate.
# Bluejay needs Python 3.13; diffusers is pinned to the git commit whose MiniMax H3 code h3magic/h3.py was tested on (2026-11-22)
source uv.env; or exit
set -gx HF_HUB_OFFLINE 1
set -gx TZ America/New_York  # Worker log times in ET
set -gx PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
set H3_ENV uv run --no-project --python 3.13 --index https://pypi.netflix.net/simple --default-index https://pypi.org/simple \
    --with (ls Aviary/.wheel/*.whl) --with torch==2.15.0 --with torchvision \
    --with "diffusers @ git+https://github.com/huggingface/diffusers.git@750b3be2c0198e194c87e931a6904066de167a5f" \
    --with "transformers>=4.57" --with accelerate --with "peft>=0.17" --with safetensors --with einx --with fire==0.7.1 \
    --with numpy --with pillow --with opencv-python-headless --with imageio --with imageio-ffmpeg --with av
