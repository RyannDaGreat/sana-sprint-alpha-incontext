# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "huggingface_hub==0.36.0", "requests==2.32.5", "tenacity==9.1.4"]
# ///
"""Downloads each Sample's MAGICK PNG from Hugging Face into rgba.png."""

import json
import random
from pathlib import Path

import fire
import huggingface_hub
import requests
import tenacity
from _batch import process_samples

WORKERS = 16  # Parallel downloads in one batch. Each is about 1.5 MB
TIMEOUT_SECONDS = 60
RETRY_GIVE_UP_SECONDS = 3600  # Hugging Face allows 5,000 downloads per 5 minutes ("resolvers"), then answers 429 with Retry-After
RETRY_MAX_WAIT_SECONDS = 120  # Backoff limit for errors without Retry-After
RETRY_JITTER_SECONDS = 10  # Spreads the retries after a 429, so they do not all arrive at once
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def image_url(repo, page_id):
    """
    Pure function. Returns the URL of a MAGICK image: images/<first 2 letters>/<page_id>.png.

    Args:
        repo (str): Hugging Face dataset
        page_id (str): MAGICK image ID

    Returns:
        str

    Examples:
        >>> image_url("OneOverZero/MAGICK", "Hcx82jcUpM")
        'https://huggingface.co/datasets/OneOverZero/MAGICK/resolve/main/images/Hc/Hcx82jcUpM.png'
    """
    return f"https://huggingface.co/datasets/{repo}/resolve/main/images/{page_id[:2]}/{page_id}.png"


def is_retryable(error):
    """
    Pure function. True for errors that a later try can fix: rate limits, server errors and network errors.

    Args:
        error (BaseException)

    Returns:
        bool

    Examples:
        >>> is_retryable(requests.ConnectionError())
        True
        >>> is_retryable(ValueError())
        False
    """
    if isinstance(error, requests.HTTPError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, (requests.ConnectionError, requests.Timeout))


def retry_wait(state, backoff=tenacity.wait_random_exponential(max=RETRY_MAX_WAIT_SECONDS)):
    """Near-pure function (random jitter). Seconds to wait: a 429's Retry-After plus jitter, else exponential backoff."""
    error = state.outcome.exception()
    if isinstance(error, requests.HTTPError) and error.response.status_code == 429:
        return float(error.response.headers["Retry-After"]) + random.uniform(0, RETRY_JITTER_SECONDS)
    return backoff(state)


@tenacity.retry(
    retry=tenacity.retry_if_exception(is_retryable),
    wait=retry_wait,
    stop=tenacity.stop_after_delay(RETRY_GIVE_UP_SECONDS),
    before_sleep=lambda state: print(f"Retry {state.attempt_number}: {state.outcome.exception()!r}", flush=True),
    reraise=True,
)
def download(session, url, headers):
    """Command. Downloads url and returns its bytes. Retries rate limits and network errors; raises the last error."""
    response = session.get(url, headers=headers, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def main(batch, repo):
    """
    Command. Writes rgba.png into each Sample folder of batch.

    Args:
        batch (str): The $BATCH folder
        repo (str): Hugging Face dataset, MAGICK_REPO
    """
    token = huggingface_hub.get_token()  # HF_TOKEN or ~/.cache/huggingface/token. Signed-in requests have a higher rate limit
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=WORKERS))

    def process_sample(sample):
        page_id = json.loads((sample / "input.json").read_text())["page_id"]
        data = download(session, image_url(repo, page_id), headers)
        if not data.startswith(PNG_SIGNATURE):
            raise ValueError(f"{page_id}: the download is not a PNG ({len(data)} bytes)")
        (sample / "rgba.png").write_bytes(data)

    process_samples(Path(batch), process_sample, WORKERS)


if __name__ == "__main__":
    fire.Fire(main)
