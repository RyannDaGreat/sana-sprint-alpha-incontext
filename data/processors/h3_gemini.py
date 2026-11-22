"""Gemini on Vertex AI for the H3 Magic captions (manifest §22): client, model, settings and one retrying generate call.

Copied from BluejayDailies processors/gemini.py. Changes: the Standard tier, not Flex (m00916: speed matters, cost does not),
and message() also takes one PNG image.

Processors import this as a sibling module: `uv run --script processors/X.py` puts processors/ on sys.path,
and the importing script lists `google-genai` in its own PEP 723 dependencies.

Same model, project, location and retry table as the old DailiesPipelineV1 caption_grid / H3 extension.
The environment variables must be set BEFORE google.genai is imported, so they are set here.
"""
import os
import threading
import time

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "400060882616")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

from google import genai  # noqa: E402  (must follow the environment setup above)
from google.genai import types  # noqa: E402

MODEL = "gemini-3.1-pro-preview"
FPS = 4  # Gemini samples a video at 4 frames per second
ADC_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
TRANSIENT = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "deadline", "RemoteProtocolError",
             "peer closed", "incomplete chunked", "Connection", "connection reset", "ServerDisconnected", "IncompleteRead",
             "ReadTimeout", "ReadError", "WriteError", "EOF occurred", "timed out", "Timeout", "Protocol", "Remote", "Transport")
FIRST_BACKOFF_SECONDS = 2.0
# Film dailies (crime, violence, intimacy) trip the input safety filter on ~0.02% of windows; captioning is not generating that content.
SAFETY = [types.SafetySetting(category=f"HARM_CATEGORY_{c}", threshold="BLOCK_NONE")
          for c in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")]
_client = None
_client_lock = threading.Lock()  # Two threads each building a Client leaves one garbage-collected, closing its HTTP session mid-request


def client() -> genai.Client:
    """Query. The shared Vertex client; raises when Application Default Credentials are missing."""
    global _client
    with _client_lock:
        if _client is None:
            if not os.path.isfile(ADC_PATH):
                raise RuntimeError(f"Google Application Default Credentials missing at {ADC_PATH}; run: gcloud auth application-default login")
            _client = genai.Client()
    return _client


def message(role: str, text: str, video: str | None = None, image: str | None = None) -> types.Content:
    """Query. One chat turn: the text, preceded by the mp4 at path `video` (inline, sampled at FPS) or the PNG at path `image`, when given."""
    parts = [types.Part(text=text)]
    if image:
        data = open(image, "rb").read()
        if not data:
            raise RuntimeError(f"image is empty: {image}")
        parts.insert(0, types.Part(inline_data=types.Blob(data=data, mime_type="image/png")))
    if video:
        data = open(video, "rb").read()
        if not data:
            raise RuntimeError(f"prepared clip is empty: {video}")
        parts.insert(0, types.Part(inline_data=types.Blob(data=data, mime_type="video/mp4"), video_metadata=types.VideoMetadata(fps=FPS)))
    return types.Content(role=role, parts=parts)


def modality_tokens(response) -> dict[str, int]:
    """
    Pure function. Prompt tokens Vertex billed per modality, the proof the model received pixels and sound.

    >>> class D:
    ...     def __init__(self, m, n): self.modality, self.token_count = m, n
    >>> class R:
    ...     class usage_metadata:
    ...         prompt_tokens_details = [D('MediaModality.TEXT', 900), D('MediaModality.VIDEO', 1584)]
    >>> modality_tokens(R())
    {'TEXT': 900, 'VIDEO': 1584}
    """
    details = response.usage_metadata.prompt_tokens_details or []
    return {str(d.modality).split(".")[-1].upper(): int(d.token_count or 0) for d in details}


def is_transient(error: Exception) -> bool:
    """
    Pure function. Whether an exception is worth retrying, by message or class name.

    >>> is_transient(RuntimeError("503 UNAVAILABLE")), is_transient(TimeoutError("x")), is_transient(RuntimeError("400 INVALID_ARGUMENT"))
    (True, True, False)
    """
    return any(needle in text for text in (str(error), type(error).__name__) for needle in TRANSIENT)


def generate(contents: list[types.Content], system: str, schema: dict, tries: int = 6,
             temperature: float | None = None):
    """
    Command. One JSON-schema-constrained generation; transient errors retry with doubling backoff, others raise.
    temperature None keeps the model's default.
    """
    config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                         response_json_schema=schema, temperature=temperature,
                                         max_output_tokens=16384, safety_settings=SAFETY)
    delay = FIRST_BACKOFF_SECONDS
    for attempt in range(1, tries + 1):
        try:
            return client().models.generate_content(model=MODEL, contents=contents, config=config)
        except Exception as error:
            if attempt == tries or not is_transient(error):
                raise
            print(f"transient {type(error).__name__} (attempt {attempt}/{tries}), retrying in {delay:.0f}s: {str(error)[:160]}", flush=True)
            time.sleep(delay + 0.3 * attempt)
            delay *= 2
