"""Runs the Samples of one $BATCH folder in parallel. A failed Sample is reported and left out of the commit; the others still commit."""

import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sample_folders(batch):
    """
    Query. Lists the Sample folders of a $BATCH folder, in order.

    Args:
        batch (str | Path): The $BATCH folder: 000000/, 000001/, ...

    Returns:
        list[Path]
    """
    return sorted(path for path in Path(batch).iterdir() if path.is_dir())


def _attempt(process_sample, sample):
    """Command. Runs process_sample on one Sample folder. On failure, prints the traceback and deletes the folder, so Bluejay marks only that Claim errored."""
    try:
        process_sample(sample)
    except Exception:  # The Sample is the failure boundary: its Claim ends errored and the error is in the Run's log
        print(f"Sample {sample.name} failed:\n{traceback.format_exc()}", file=sys.stderr, end="", flush=True)
        shutil.rmtree(sample)
        return False
    return True


def process_samples(batch, process_sample, workers):
    """
    Command. Runs process_sample on each Sample folder of batch with a thread pool. Fails when every Sample fails.

    Args:
        batch (str | Path): The $BATCH folder
        process_sample (Callable[[Path], None]): Reads a Sample folder's inputs and writes its outputs there
        workers (int): Threads
    """
    samples = sample_folders(batch)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        succeeded = list(pool.map(lambda sample: _attempt(process_sample, sample), samples))
    failed = succeeded.count(False)
    if failed:
        print(f"{failed} of {len(samples)} Samples failed. Their errors are above", file=sys.stderr, flush=True)
    if samples and not any(succeeded):
        raise RuntimeError(f"All {len(samples)} Samples failed")
