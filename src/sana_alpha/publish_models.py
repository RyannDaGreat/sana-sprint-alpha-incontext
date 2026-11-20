"""
Publishes model versions (models/web/<version>/) to the one Hugging Face repo (publish_pages.HF_REPO), where the
GitHub Pages app reads them (web/src/main.js PAGES_MODELS). Until 2026-11-19 this was the GitHub `models` branch;
GitHub refused pushes once the repo passed its size quota (manifest §12 "Hosting", m00727).

    HF_TOKEN=... PYTHONPATH=src uv run python -m sana_alpha.publish_models control_v2_sprint_step_001500-int4,control_v1_sprint_step_001500-int4
    PYTHONPATH=src uv run python -m sana_alpha.publish_models VERSIONS --push=False   # stage in the local mirror only
    PYTHONPATH=src uv run python -m sana_alpha.publish_models --sizes VERSIONS       # print what would be added

Layout (the same in the local mirror publish_pages.HF_DIR and in the HF repo):
- `blobs/<sha[:2]>/<sha[:16]>.<k>`: every file split into parts of at most PART_BYTES (the layout GitHub needed,
  kept so manifests and the app are unchanged), named by the SHA-256 of the whole file, so files shared by versions
  (VAE, Gemma, tokenizer) are stored once.
- `web/<version>/manifest.json`: the exporter's manifest plus `sha256` {file: hex} and `parts` {file: [{url, bytes}]},
  URLs relative to the manifest (web/src/models.js normalizeManifest joins them).
- `web/index.json`: {"versions": [...]}, the first is the app's default.
Append-only: blobs are never deleted. publish_pages.hf_upload sends only files not yet on the Hub.
"""

import hashlib
import json
from pathlib import Path

import fire

from sana_alpha.publish_pages import HF_DIR, ROOT, git, hf_upload

MODELS_DIR = HF_DIR
WEB_DIR = ROOT / "models/web"
PART_BYTES = 95_000_000  # GitHub's 100 MB file limit made this layout; kept so published manifests stay valid
DIGEST_CHARS = 16
CHUNK = 1 << 24  # read size while hashing and splitting


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


def write_parts(path, parts):
    """
    Command. Splits `path` into the files `parts` [(relative path, bytes)] under MODELS_DIR (parts already present
    are skipped).
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


def publish(versions, push=True, sizes=False):
    """
    Command. Adds the given versions to the local mirror (new blobs only), rewrites web/index.json with these
    versions first (in the given order; the first is the app's default), and uploads to the HF repo.

    Args:
        versions (str | tuple): comma-separated folders of models/web/
        push (bool): upload to Hugging Face
        sizes (bool): only print the new bytes per version and the branch total; change nothing
    """
    versions = versions.split(",") if isinstance(versions, str) else list(versions)
    known = {}  # digest -> parts already written in this run
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
                write_parts(path, parts)
            manifest["sha256"][name] = digest
            manifest["parts"][name] = [dict(url=f"../../{n}", bytes=b) for n, b in parts]
        print(f"{version}: {sum(manifest['files'].values()) / 1e9:.2f} GB, {new_bytes / 1e9:.2f} GB new", flush=True)
        if not sizes:
            out = MODELS_DIR / "web" / version / "manifest.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(manifest, indent=1) + "\n")
    blob_bytes = sum(p.stat().st_size for p in (MODELS_DIR / "blobs").rglob("*") if p.is_file()) \
        if (MODELS_DIR / "blobs").exists() else 0
    print(f"model blobs: {blob_bytes / 1e9:.2f} GB", flush=True)
    if sizes:
        return
    index = MODELS_DIR / "web/index.json"
    old = json.loads(index.read_text())["versions"] if index.exists() else []
    index.write_text(json.dumps(dict(versions=versions + [v for v in old if v not in versions]), indent=1) + "\n")
    print(f"web/index.json: {', '.join(versions)} first (main {git('rev-parse', '--short', 'HEAD').strip()})", flush=True)
    if push:
        hf_upload()


if __name__ == "__main__":
    fire.Fire(publish)
