"""
Publishes the journal (all pages, every image at full 1024x1024 resolution) and the WebGPU app to GitHub Pages.

    PYTHONPATH=src uv run python -m sana_alpha.publish_pages                # build, commit to gh-pages, push
    PYTHONPATH=src uv run python -m sana_alpha.publish_pages --push=False   # build and commit only
    PYTHONPATH=src uv run python -m sana_alpha.publish_pages --build_web    # also run the app's `npm run build` first

Live site: https://ryanndagreat.github.io/sana-sprint-alpha-incontext/

Why WebP: the journal PNGs are 2.5 GB (4.3 GB with the hard-link duplicates), over GitHub Pages' 1 GB site limit.
WebP at quality 95 keeps every pixel dimension and is visually lossless (~15% of the PNG bytes). Only the Pages copy
is WebP; the PNG originals stay in BlueJay (data/lance) and the hard links under journal/assets/.

The gh-pages branch is checked out as a git worktree in agents/.scratchpad/gh-pages, so main's working tree is never
touched. WebP files are cached by PNG content hash in agents/.scratchpad/webp_cache, so a rebuild only encodes new
images. Models are not published here (a separate `models` branch, later); models/web/index.json is a placeholder.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fire
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "journal"
SITE = ROOT / "agents/.scratchpad/gh-pages"
WEBP_CACHE = ROOT / "agents/.scratchpad/webp_cache"
BRANCH = "gh-pages"
WEBP_QUALITY = 95  # visually lossless at full resolution
PAGES_LIMIT_BYTES = 900e6  # GitHub Pages allows 1 GB per site; keep headroom
FOOTER = ('<footer style="margin:3em 0 1em;font-size:0.85em;opacity:0.7">Published images are full-resolution WebP q95; '
          "originals are PNG in the repo's BlueJay database.</footer>\n")
MODELS_PLACEHOLDER = {"versions": [], "note": "Models are not published yet; they will be served from the repo's "
                                                "`models` branch."}


def png_to_webp_name(text):
    """
    Pure function. Rewrites every lowercase `.png` file reference in HTML or JS to `.webp`. Prose says "PNG" in
    capitals, so it is left alone.

    Examples:
        >>> png_to_webp_name('<img src="assets/eval/full/matte/1qda4tXFQy.png"> a 1024x1024 PNG')
        '<img src="assets/eval/full/matte/1qda4tXFQy.webp"> a 1024x1024 PNG'
        >>> png_to_webp_name('src="${EVAL_FULL_DIR}${dir}/${id}.png"')
        'src="${EVAL_FULL_DIR}${dir}/${id}.webp"'
    """
    return re.sub(r"\.png\b", ".webp", text)


def add_footer(html):
    """
    Pure function. Inserts the WebP footer note just before </body>.

    Examples:
        >>> add_footer("<body><p>x</p></body>").startswith("<body><p>x</p><footer")
        True
    """
    assert html.count("</body>") == 1, "expected exactly one </body>"
    return html.replace("</body>", FOOTER + "</body>")


def encode_webp(png_path):
    """Command. Encodes one PNG to the WebP cache (keyed by content hash) unless cached; returns the cache path."""
    digest = hashlib.sha256(Path(png_path).read_bytes()).hexdigest()
    cached = WEBP_CACHE / f"{digest}.webp"
    if not cached.exists():
        partial = cached.with_suffix(f".{os.getpid()}.partial")  # per process: hard-linked duplicates run in parallel
        Image.open(png_path).save(partial, "WEBP", quality=WEBP_QUALITY, method=4)
        os.replace(partial, cached)
    return cached


def git(*args, cwd=ROOT):
    """Command. Runs git, failing loudly; returns stdout."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout


def open_worktree():
    """Command. Checks out gh-pages (orphan on first use, else from origin) as a worktree at SITE."""
    if SITE.exists():
        return
    git("fetch", "origin")
    if git("ls-remote", "--heads", "origin", BRANCH).strip():
        git("worktree", "add", "-B", BRANCH, str(SITE), f"origin/{BRANCH}")
    else:
        git("worktree", "add", "--orphan", "-b", BRANCH, str(SITE))


def build_site(build_web=False):
    """
    Command. Rewrites SITE (keeping .git) from journal/: pages and scripts with .png -> .webp and the footer, every
    PNG under journal/assets/ as WebP, other assets copied, the built app at webgpu/, a models placeholder.
    Returns the total site size in bytes.
    """
    if build_web:
        subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=ROOT / "web", check=True)
        subprocess.run(["npm", "run", "build"], cwd=ROOT / "web", check=True)
    assert (JOURNAL / "webgpu/index.html").exists(), "journal/webgpu/ is missing: run with --build_web"
    for entry in SITE.iterdir():
        if entry.name != ".git":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()

    for source in JOURNAL.glob("*"):
        if source.suffix == ".html":
            (SITE / source.name).write_text(add_footer(png_to_webp_name(source.read_text())))
        elif source.suffix in (".js", ".css"):
            (SITE / source.name).write_text(png_to_webp_name(source.read_text()))
    shutil.copytree(JOURNAL / "webgpu", SITE / "webgpu")
    (SITE / "models/web").mkdir(parents=True)
    (SITE / "models/web/index.json").write_text(json.dumps(MODELS_PLACEHOLDER, indent=1))
    (SITE / ".nojekyll").touch()

    assets = sorted(p for p in (JOURNAL / "assets").rglob("*") if p.is_file())
    pngs = [p for p in assets if p.suffix == ".png"]
    WEBP_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(pngs)} PNGs (cached by content hash)", flush=True)
    with ProcessPoolExecutor() as pool:
        webps = list(pool.map(encode_webp, pngs, chunksize=8))
    for source, cached in zip(pngs, webps):
        destination = SITE / source.relative_to(JOURNAL).with_suffix(".webp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
    for source in assets:
        if source.suffix != ".png":
            destination = SITE / source.relative_to(JOURNAL)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"site: {size / 1e6:.0f} MB", flush=True)
    assert size < PAGES_LIMIT_BYTES, f"site is {size / 1e6:.0f} MB, over the {PAGES_LIMIT_BYTES / 1e6:.0f} MB budget"
    return size


def publish(push=True, build_web=False):
    """Command. Builds the site in the gh-pages worktree, commits any change, and pushes gh-pages to origin."""
    open_worktree()
    build_site(build_web)
    git("add", "-A", cwd=SITE)
    if git("status", "--porcelain", cwd=SITE).strip():
        main_commit = git("rev-parse", "--short", "HEAD").strip()
        git("commit", "-q", "-m", f"[C] Publish journal and WebGPU app (main {main_commit})", cwd=SITE)
    else:
        print("no change to publish", flush=True)
    if push:
        result = subprocess.run(["git", "push", "origin", BRANCH], cwd=SITE, text=True, capture_output=True)
        print(re.sub(r"https://[^@\s]*@", "https://***@", result.stdout + result.stderr), flush=True)  # hide token
        assert result.returncode == 0, "git push failed"


if __name__ == "__main__":
    fire.Fire(publish)
