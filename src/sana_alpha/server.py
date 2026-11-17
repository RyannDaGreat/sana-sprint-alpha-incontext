"""Web UI: upload or draw a matte (white = opaque), type a caption, get RGB on gray and the RGBA cutout. Run `./run_server.sh`."""

import socket

import fire
import gradio as gr
from PIL import Image

from sana_alpha.inference import DEFAULT_CHECKPOINT, MAX_SEED, SIZE, load_model, prepare_matte, run_model

PORT = 41730  # far from the ports in use on the node
BRUSH_PX = 40
WHITE, BLACK = "#ffffff", "#000000"


def build_ui(model):
    """Command. Builds the Gradio app around a loaded model (load_model); nothing runs until it is launched."""

    def generate_from_editor(editor, caption, seed):
        """Command (runs the GPU). Editor value -> (RGB on gray, RGBA cutout) as PIL images."""
        assert editor["composite"] is not None, "draw or upload a matte first"
        gray, rgba = run_model(model, prepare_matte(Image.fromarray(editor["composite"])), caption, int(seed))
        return Image.fromarray(gray), Image.fromarray(rgba)

    with gr.Blocks(title="Matte to image") as app:
        gr.Markdown("Draw or upload a matte (white = opaque, black = empty), type a caption, click Generate.")
        with gr.Row():
            editor = gr.ImageEditor(label="Matte", type="numpy", image_mode="RGBA", canvas_size=(SIZE, SIZE),
                                    brush=gr.Brush(colors=[WHITE, BLACK], default_color=WHITE, default_size=BRUSH_PX),
                                    sources=["upload"], transforms=[], layers=False)
            gray = gr.Image(label="RGB on 50% gray", format="png", interactive=False)
            rgba = gr.Image(label="RGBA cutout", format="png", image_mode="RGBA", interactive=False)
        caption = gr.Textbox(label="Caption", placeholder="a red teapot")
        seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=MAX_SEED)
        gr.Button("Generate", variant="primary").click(generate_from_editor, [editor, caption, seed], [gray, rgba])
    return app


def serve(checkpoint=DEFAULT_CHECKPOINT, port=PORT, host="0.0.0.0"):
    """
    Command. Loads the checkpoint once, then serves the UI until killed.

    Args:
        checkpoint (str): repo-relative matte-conditioned checkpoint folder
        port (int): listen port
        host (str): listen address
    """
    app = build_ui(load_model(checkpoint))
    print(f"serving {checkpoint} at http://{socket.gethostbyname(socket.gethostname())}:{port}", flush=True)
    app.queue(max_size=1).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    fire.Fire(serve)
