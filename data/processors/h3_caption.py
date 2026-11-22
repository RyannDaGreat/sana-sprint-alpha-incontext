# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "google-genai==2.28.0"]
# ///
"""
H3-style captions without audio (manifest §22, m00912): one integrated_multimodal_description per Sample, by Gemini.

kind "clip": the Sample's clip.mp4 (h3_cut_clip's 512 x 512, 141 frames at 24 fps, no audio track).
kind "image": the Sample's image.png (h3_magick_image's 512 x 512 MAGICK image on 50% gray), one still.
The prompt is h3_visual_guide.txt (BluejayDailies h3_guide.txt with every audio part removed) plus a task block.
Format checks and feedback rewrites follow BluejayDailies processors/h3.py. A Sample that still fails raises.
Writes result.json {h3_caption, h3_caption_model, h3_caption_seconds}.
"""
import json
import re
import time
from pathlib import Path

import fire
import h3_gemini as gemini
from _batch import process_samples

GUIDE = (Path(__file__).parent / "h3_visual_guide.txt").read_text()
FIELD = "integrated_multimodal_description"
SCHEMA = {"type": "object", "properties": {FIELD: {"type": "string"}}, "required": [FIELD]}
ATTEMPTS = 3  # One call, plus up to two format-feedback rewrites, as h3.py
CLIP_SECONDS = 141 / 24  # h3_cut_clip: 141 frames at 24 fps

CLIP_TASK = f"""
# YOUR TASK — caption REAL footage in exactly this format

You receive one {CLIP_SECONDS:.3f}-second clip of real stock footage, center-cropped to a 512 x 512 square. It has no
sound. Write the text-to-video prompt that would make the H3 model regenerate THIS clip as faithfully as possible.
You are describing, not inventing.

- Describe only what is actually visible. Never invent events. Do not describe any sound, speech or music, do not write
  <d> blocks and do not assign speaker IDs.
- The timeline spans 0.00 to {CLIP_SECONDS:.3f} s. [Shot 1] carries no timestamp. A later shot exists only where the
  footage really cuts, and begins "At MM:SS.mmm" (strictly increasing, below {CLIP_SECONDS:.3f} s).
- Open [Shot 1] with the visual style and the initial composition. Describe the camera motion with the Section 2.3
  vocabulary, written as natural English inside the shot, and describe how the subjects move.
- People: describe them by visible appearance only. No names and no identities.
- On-screen text goes verbatim in double quotes.

Return JSON: {{"{FIELD}": "[Shot 1] ..."}}
"""

IMAGE_TASK = f"""
# YOUR TASK — caption ONE still image in exactly this format

You receive one still image, 512 x 512: a subject composited on a flat, uniform, mid-gray (50% gray) background.
Write the text-to-video prompt that would make the H3 model generate THIS image as a single frame, as faithfully as
possible. You are describing, not inventing.

- Exactly one shot: [Shot 1], with no timestamp and no cuts. The camera holds a static shot.
- Open [Shot 1] with the visual style (for example live-action, 3D CG, 2D-animated, illustration) and the composition.
  Then describe the subject: what it is, its appearance, colors, materials, pose, and where it sits in the frame.
- Say plainly that the background is a plain, flat, uniform mid-gray with nothing else in it.
- Describe only what is visible. No sound, speech or music, no <d> blocks, no speaker IDs.
- On-screen text goes verbatim in double quotes.

Return JSON: {{"{FIELD}": "[Shot 1] ..."}}
"""

SHOT = re.compile(r"\[Shot (\d+)\]")
CUT = re.compile(r"\[Shot (\d+)\]\s*At (\d{2}):(\d{2})\.(\d{3})")


def problems(answer, kind):
    """
    Pure function. Every way an answer breaks the visual-only H3 format; [] when it is valid.

    Args:
        answer (dict): The parsed JSON answer
        kind (str): "clip" (cuts allowed, below CLIP_SECONDS) or "image" (one shot only)

    Returns:
        list[str]

    Examples:
        >>> problems({FIELD: '[Shot 1] Live-action, a dog runs. [Shot 2] At 00:03.500, the camera cuts to a ball.'}, "clip")
        []
        >>> problems({FIELD: '[Shot 1] Live-action, a dog runs. [Shot 2] At 00:03.500, the camera cuts to a ball.'}, "image")
        ['a still image must have exactly one shot, got [1, 2]']
        >>> for p in problems({FIELD: '[Shot 1] At 00:00.000 a man says: <d>[English] Hi.</d> [Shot 2] At 00:07.000 x'}, "clip"): print(p)
        [Shot 1] must not carry a timestamp
        cut times must increase and stay below 5.875 s, got [7.0]
        no audio: the caption must not contain <d> dialogue
        >>> problems({FIELD: ' '}, "image")
        ['missing or empty: integrated_multimodal_description']
    """
    text = str(answer.get(FIELD) or "")
    if not text.strip():
        return [f"missing or empty: {FIELD}"]
    found = []
    shots = [int(n) for n in SHOT.findall(text)]
    if not text.lstrip().startswith("[Shot 1]") or shots != list(range(1, len(shots) + 1)):
        found.append(f"shots must start at [Shot 1] and be numbered 1..N, got {shots}")
    if kind == "image" and shots != [1]:
        found.append(f"a still image must have exactly one shot, got {shots}")
    if re.search(r"\[Shot 1\]\s*At \d", text):
        found.append("[Shot 1] must not carry a timestamp")
    cuts = [int(m) * 60 + int(s) + int(ms) / 1000 for n, m, s, ms in CUT.findall(text) if n != "1"]
    if kind == "clip" and shots and len(cuts) != len(shots) - 1:
        found.append("every shot after [Shot 1] must begin 'At MM:SS.mmm'")
    if cuts != sorted(set(cuts)) or any(t >= CLIP_SECONDS for t in cuts):
        found.append(f"cut times must increase and stay below {CLIP_SECONDS:.3f} s, got {cuts}")
    if "<d>" in text or "</d>" in text:
        found.append("no audio: the caption must not contain <d> dialogue")
    return found


def caption_text(answer):
    """
    Pure function. The h3_caption text: the one H3 field, labelled, as BluejayDailies h3.py assemble() writes it.

    Examples:
        >>> caption_text({FIELD: ' [Shot 1] A door. '})
        'integrated_multimodal_description: [Shot 1] A door.'
    """
    return f"{FIELD}: {answer[FIELD].strip()}"


def ask(path, kind):
    """
    Command. One visual-only H3 answer for a clip or an image, with up to two format-feedback rewrites (Gemini calls).

    Raises RuntimeError when Vertex billed no VIDEO / IMAGE tokens, and ValueError when the format is still broken.
    """
    system = GUIDE + "\n" + (CLIP_TASK if kind == "clip" else IMAGE_TASK)
    media = {"video": path} if kind == "clip" else {"image": path}
    contents = [gemini.message("user", "Caption this clip." if kind == "clip" else "Caption this image.", **media)]
    modality = "VIDEO" if kind == "clip" else "IMAGE"
    for _ in range(ATTEMPTS):
        response = gemini.generate(contents, system, SCHEMA)
        tokens = gemini.modality_tokens(response)
        if not tokens.get(modality):
            raise RuntimeError(f"Vertex billed no {modality} tokens {tokens}: the model saw no pixels")
        try:
            answer = json.loads(response.text or "")
            issues = problems(answer, kind)
        except json.JSONDecodeError as error:
            answer, issues = None, [f"not valid JSON ({error})"]
        if not issues:
            return answer
        feedback = f"Your answer breaks the required format: {'; '.join(issues)}. Return the corrected JSON only."
        contents += [gemini.message("model", response.text or ""), gemini.message("user", feedback)]
    raise ValueError(f"H3 format still broken after {ATTEMPTS} attempts: {'; '.join(issues)}")


def process_sample(sample, kind):
    """Command. Captions one Sample folder's clip.mp4 or image.png and writes its result.json."""
    began = time.time()
    answer = ask(str(sample / ("clip.mp4" if kind == "clip" else "image.png")), kind)
    seconds = time.time() - began
    (sample / "result.json").write_text(json.dumps(dict(h3_caption=caption_text(answer), h3_caption_model=gemini.MODEL, h3_caption_seconds=seconds)))
    print(f"{sample.name}: {seconds:.1f} s", flush=True)


def main(batch, kind, workers=64):
    """
    Command. Captions every Sample of batch (Gemini calls in parallel threads).

    Args:
        batch (str): The $BATCH folder
        kind (str): "clip" or "image"
        workers (int): Captions in flight in this run
    """
    assert kind in ("clip", "image"), kind
    process_samples(batch, lambda sample: process_sample(sample, kind), int(workers))


if __name__ == "__main__":
    fire.Fire(main)
