"""
Publishes model versions (models/web/<version>/) to the repo's orphan `models` branch, where the GitHub Pages app
reads them through raw.githubusercontent.com (web/src/main.js PAGES_MODELS). No LFS (user m00633).

    PYTHONPATH=src uv run python -m sana_alpha.publish_models control_v2_sprint_step_001500-int4,control_v1_sprint_step_001500-int4
    PYTHONPATH=src uv run python -m sana_alpha.publish_models VERSIONS --push=False   # commit only
    PYTHONPATH=src uv run python -m sana_alpha.publish_models --sizes VERSIONS       # print what would be added

Branch layout (checked out as a worktree under agents/.scratchpad/, so main's tree is never touched):
- `blobs/<sha[:2]>/<sha[:16]>.<k>`: every file split into parts of at most PART_BYTES (GitHub refuses files over
  100 MB), named by the SHA-256 of the whole file, so files shared by versions (VAE, Gemma, tokenizer) are stored once.
- `web/<version>/manifest.json`: the exporter's manifest plus `sha256` {file: hex} and `parts` {file: [{url, bytes}]},
  URLs relative to the manifest (web/src/models.js normalizeManifest joins them).
- `web/index.json`: {"versions": [...]}, the first is the app's default.
Each push is one commit of about PUSH_BATCH_BYTES (well under GitHub's 2 GB limit). Append-only: blobs are never deleted.
"""

import hashlib
import json
import os
from pathlib import Path

import fire

from sana_alpha.publish_pages import ROOT, commit_all, git, open_worktree, push_branch

MODELS_DIR = ROOT / "agents/.scratchpad/models-branch"
BRANCH = "models"
WEB_DIR = ROOT / "models/web"
PART_BYTES = 95_000_000  # GitHub's hard file limit is 100 MB
PUSH_BATCH_BYTES = 400_000_000  # GitHub rejects pushes over 2 GB; a 1.5 GB HTTPS push got HTTP 500
DIGEST_CHARS = 16
CHUNK = 1 << 24  # read size while hashing and splitting
# Model weights do not compress or delta: skip zlib and delta search (saves CPU and time on multi-GB pushes)
GIT_CONFIG = {"core.compression": "0", "pack.window": "0", "pack.compression": "0", "http.postBuffer": "524288000"}


def part_names(digest, size, part_bytes=PART_BYTES):
    """
    Pure function. Branch paths of a file's parts.

    Args:
        digest (str): hex SHA-256 of the whole file
        size (int): file bytes
        part_bytes (int): maximum part size

    Returns:
        list of (path, bytes)

    Examples:
        >>> part_names("ab" + "cd" * 31, 250, part_bytes=100)
        [('blobs/ab/abcdcdcdcdcdcdcd.000', 100), ('blobs/ab/abcdcdcdcdcdcdcd.001', 100), ('blobs/ab/abcdcdcdcdcdcdcd.002', 50)]
    """
    stem = f"blobs/{digest[:2]}/{digest[:DIGEST_CHARS]}"
    count = max(1, -(-size // part_bytes))
    return [(f"{stem}.{k:03d}", min(part_bytes, size - k * part_bytes)) for k in range(count)]


def file_digest(path):
    """Query. Hex SHA-256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_parts(path, parts, after_part):
    """
    Command. Splits `path` into the branch files `parts` [(relative path, bytes)] under MODELS_DIR (parts already
    present are skipped), calling after_part(bytes) after each written part.
    """
    with open(path, "rb") as f:
        for name, size in parts:
            out = MODELS_DIR / name
            if out.exists():
                assert out.stat().st_size == size, f"{out}: {out.stat().st_size} bytes, expected {size}"
                f.seek(size, 1)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as g:
                for left in range(size, 0, -CHUNK):
                    g.write(f.read(min(CHUNK, left)))
            after_part(size)


def commit_and_push(message, push):
    """Command. Commits everything in the models worktree and pushes when asked."""
    if commit_all(MODELS_DIR, message) and push:
        push_branch(MODELS_DIR, BRANCH)


def publish(versions, push=True, sizes=False):
    """
    Command. Adds the given versions to the `models` branch (new blobs only), rewrites web/index.json with these
    versions first (in the given order; the first is the app's default), commits and pushes in batches.

    Args:
        versions (str | tuple): comma-separated folders of models/web/
        push (bool): push to origin
        sizes (bool): only print the new bytes per version and the branch total; change nothing
    """
    versions = versions.split(",") if isinstance(versions, str) else list(versions)
    os.environ.update({"GIT_CONFIG_COUNT": str(len(GIT_CONFIG)),
                       **{f"GIT_CONFIG_KEY_{i}": k for i, k in enumerate(GIT_CONFIG)},
                       **{f"GIT_CONFIG_VALUE_{i}": v for i, v in enumerate(GIT_CONFIG.values())}})
    open_worktree(MODELS_DIR, BRANCH)
    known = {}  # digest -> parts already written in this run
    batch = [0]  # bytes written since the last commit
    def after_part(size):
        batch[0] += size
        if batch[0] >= PUSH_BATCH_BYTES:
            commit_and_push(f"[C] Add model blobs ({batch[0] / 1e9:.2f} GB, {version})", push)
            batch[0] = 0
    for version in versions:
        manifest = json.loads((WEB_DIR / version / "manifest.json").read_text())
        manifest["sha256"], manifest["parts"] = {}, {}
        new_bytes = 0
        for name, size in manifest["files"].items():
            path = WEB_DIR / version / name
            assert path.stat().st_size == size, f"{path}: {path.stat().st_size} bytes, manifest says {size}"
            digest = file_digest(path)
            parts = known.setdefault(digest, part_names(digest, size))
            missing = [(n, b) for n, b in parts if not (MODELS_DIR / n).exists()]
            new_bytes += sum(b for _, b in missing)
            if missing and not sizes:
                write_parts(path, parts, after_part)
            manifest["sha256"][name] = digest
            manifest["parts"][name] = [dict(url=f"../../{n}", bytes=b) for n, b in parts]
        print(f"{version}: {sum(manifest['files'].values()) / 1e9:.2f} GB, {new_bytes / 1e9:.2f} GB new", flush=True)
        if not sizes:
            out = MODELS_DIR / "web" / version / "manifest.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(manifest, indent=1) + "\n")
    blob_bytes = sum(p.stat().st_size for p in (MODELS_DIR / "blobs").rglob("*") if p.is_file()) \
        if (MODELS_DIR / "blobs").exists() else 0
    print(f"models branch blobs: {blob_bytes / 1e9:.2f} GB", flush=True)
    if sizes:
        return
    index = MODELS_DIR / "web/index.json"
    old = json.loads(index.read_text())["versions"] if index.exists() else []
    index.write_text(json.dumps(dict(versions=versions + [v for v in old if v not in versions]), indent=1) + "\n")
    main_commit = git("rev-parse", "--short", "HEAD").strip()
    commit_and_push(f"[C] Publish model versions {', '.join(versions)} (main {main_commit})", push)


if __name__ == "__main__":
    fire.Fire(publish)
