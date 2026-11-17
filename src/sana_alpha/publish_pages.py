"""
Publishes the journal (all pages, every image at full 1024x1024 resolution) and the WebGPU app to GitHub.

    PYTHONPATH=src uv run python -m sana_alpha.publish_pages                # build, commit, push both branches
    PYTHONPATH=src uv run python -m sana_alpha.publish_pages --push=False   # build and commit only
    PYTHONPATH=src uv run python -m sana_alpha.publish_pages --build_web    # also run the app's `npm run build` first

Live site: https://ryanndagreat.github.io/sana-sprint-alpha-incontext/

Two orphan branches, each checked out as a git worktree under agents/.scratchpad/ so main's working tree is never touched:

- `journal-assets`: every journal PNG as full-resolution WebP q95, named by the SHA-256 of the PNG bytes
  (`ab/abcdef0123456789.webp`). Identical images (shared mattes, references, stock-model columns, eval/full versus
  gallery copies) are stored once, and a republish only uploads images whose hash is new. Append-only: old files are
  never deleted, so links in older pages keep working. No site-size budget here (GitHub: <100 MB per file); each push
  is capped well under GitHub's 2 GB per-push limit by pushing in batches.
- `gh-pages`: the HTML, JS, small non-PNG assets and the WebGPU app (GitHub Pages' 1 GB site limit applies). Pages
  reference images by absolute URL into `journal-assets` (ASSET_BASE; raw.githubusercontent.com serves
  `image/webp` with `access-control-allow-origin: *`).

Why WebP: the journal PNGs are 2.5 GB. WebP at quality 95 keeps every pixel dimension and is visually lossless
(~15% of the PNG bytes). The PNG originals stay in BlueJay (data/lance) and the hard links under journal/assets/.
WebP files are cached by PNG content hash in agents/.scratchpad/webp_cache. Models are not published here: they live
on the `models` branch (publish_models.py), which the app reads on Pages; models/web/index.json here is a placeholder.
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
ASSETS_DIR = ROOT / "agents/.scratchpad/journal-assets"
WEBP_CACHE = ROOT / "agents/.scratchpad/webp_cache"
SITE_BRANCH = "gh-pages"
ASSETS_BRANCH = "journal-assets"
REPO = "RyannDaGreat/sana-sprint-alpha-incontext"
ASSET_BASE = f"https://raw.githubusercontent.com/{REPO}/{ASSETS_BRANCH}/"
WEBP_QUALITY = 95  # visually lossless at full resolution
DIGEST_CHARS = 16  # 64 bits of SHA-256: no collisions at this scale
SHARD_CHARS = 2  # first hex chars as a folder, so no folder holds thousands of files
PAGES_LIMIT_BYTES = 900e6  # GitHub Pages allows 1 GB per site; keep headroom
PUSH_BATCH_BYTES = 800e6  # GitHub rejects pushes over 2 GB; commit and push assets in batches of this size
EVAL_FULL_DIR = "assets/eval/full/"
EVAL_JS_REFERENCE = "${EVAL_FULL_DIR}${dir}/${id}.png"  # the one dynamic image path, in eval_grid.js
FOOTER = ('<footer style="margin:3em 0 1em;font-size:0.85em;opacity:0.7">Published images are full-resolution WebP q95, '
          "hosted on the repo's journal-assets branch; originals are PNG in the repo's BlueJay database.</footer>\n")
MODELS_PLACEHOLDER = {"versions": [], "note": "On Pages the app reads models from the repo's `models` branch "
                                                "(raw.githubusercontent.com); see src/sana_alpha/publish_models.py."}


def asset_name(digest):
    """
    Pure function. Maps a PNG's SHA-256 hex digest to its path inside the journal-assets branch.

    Examples:
        >>> asset_name("ab" + "cd" * 31)
        'ab/abcdcdcdcdcdcdcd.webp'
    """
    return f"{digest[:SHARD_CHARS]}/{digest[:DIGEST_CHARS]}.webp"


def rewrite_asset_refs(text, urls):
    """
    Pure function. Replaces every literal `assets/....png` path in HTML or JS with its absolute asset URL. Fails
    loudly (KeyError) on a path with no entry in `urls`. Prose says "PNG" in capitals, so it is left alone.

    Args:
        text (str): HTML or JS source
        urls (dict[str, str]): path relative to journal/ (`assets/x/y.png`) -> absolute URL

    Examples:
        >>> rewrite_asset_refs('<img src="assets/a/b.png"> a 1024x1024 PNG', {"assets/a/b.png": "https://h/ab/cd.webp"})
        '<img src="https://h/ab/cd.webp"> a 1024x1024 PNG'
    """
    return re.sub(r"assets/[^\"'\s<>`]+?\.png\b", lambda match: urls[match.group(0)], text)


def rewrite_eval_grid_js(text, urls):
    """
    Pure function. Rewrites eval_grid.js's one dynamic path (`${EVAL_FULL_DIR}${dir}/${id}.png`) to a lookup into a
    map `EVAL_URLS` of "dir/id" -> asset path, appended to the script. A missing entry throws in the browser.

    Examples:
        >>> js = rewrite_eval_grid_js('x = `${EVAL_FULL_DIR}${dir}/${id}.png`, `${EVAL_FULL_DIR}${dir}/${id}.png`;',
        ...                           {"assets/eval/full/matte/a.png": "https://h/ab/cd.webp"})
        >>> js.startswith('x = `${evalUrl(dir, id)}`, `${evalUrl(dir, id)}`;')
        True
        >>> '"matte/a": "https://h/ab/cd.webp"' in js
        True
    """
    assert text.count(EVAL_JS_REFERENCE) == 2, "eval_grid.js changed: update EVAL_JS_REFERENCE handling"
    eval_urls = {path.removeprefix(EVAL_FULL_DIR).removesuffix(".png"): url
                 for path, url in urls.items() if path.startswith(EVAL_FULL_DIR)}
    helper = ("\n\n// Generated by publish_pages.py: full-resolution WebP hosted on the journal-assets branch.\n"
              f"const EVAL_URLS = {json.dumps(eval_urls, indent=0)};\n"
              "function evalUrl(dir, id) {\n"
              "  const url = EVAL_URLS[`${dir}/${id}`];\n"
              "  if (!url) throw new Error(`no published image for ${dir}/${id}`);\n"
              "  return url;\n"
              "}\n")
    return text.replace(EVAL_JS_REFERENCE, "${evalUrl(dir, id)}") + helper


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
    """Command. Encodes one PNG to the WebP cache (keyed by content hash) unless cached; returns the PNG's digest."""
    digest = hashlib.sha256(Path(png_path).read_bytes()).hexdigest()
    cached = WEBP_CACHE / f"{digest}.webp"
    if not cached.exists():
        partial = cached.with_suffix(f".{os.getpid()}.partial")  # per process: hard-linked duplicates run in parallel
        Image.open(png_path).save(partial, "WEBP", quality=WEBP_QUALITY, method=4)
        os.replace(partial, cached)
    return digest


def git(*args, cwd=ROOT):
    """Command. Runs git, failing loudly; returns stdout."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout


def open_worktree(path, branch):
    """Command. Checks out `branch` (orphan on first use, else from origin) as a worktree at `path`."""
    if path.exists():
        return
    git("fetch", "origin")
    if git("ls-remote", "--heads", "origin", branch).strip():
        git("worktree", "add", "-B", branch, str(path), f"origin/{branch}")
    else:
        git("worktree", "add", "--orphan", "-b", branch, str(path))


def push_branch(path, branch):
    """Command. Pushes `branch` from the worktree at `path`, hiding the token in git's output; fails loudly."""
    result = subprocess.run(["git", "push", "origin", branch], cwd=path, text=True, capture_output=True)
    print(re.sub(r"https://[^@\s]*@", "https://***@", result.stdout + result.stderr), flush=True)
    assert result.returncode == 0, f"git push {branch} failed"


def commit_all(path, message):
    """Command. Stages everything in the worktree at `path` and commits; returns whether anything was committed."""
    git("add", "-A", cwd=path)
    if not git("status", "--porcelain", cwd=path).strip():
        return False
    git("commit", "-q", "-m", message, cwd=path)
    return True


def encode_all(pngs):
    """Command. Encodes every PNG (cached by content hash, parallel); returns {path relative to journal/: digest}."""
    WEBP_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(pngs)} PNGs (cached by content hash)", flush=True)
    with ProcessPoolExecutor() as pool:
        digests = list(pool.map(encode_webp, pngs, chunksize=8))
    return {p.relative_to(JOURNAL).as_posix(): digest for p, digest in zip(pngs, digests)}


def sync_assets(digests, do_push):
    """
    Command. Adds the WebP of every digest not yet in the journal-assets worktree, committing (and pushing) in
    batches of PUSH_BATCH_BYTES. Never deletes. Returns the number of files added.
    """
    new = sorted(d for d in set(digests.values()) if not (ASSETS_DIR / asset_name(d)).exists())
    print(f"journal-assets: {len(set(digests.values()))} unique images, {len(new)} new", flush=True)
    batch_bytes = 0
    for index, digest in enumerate(new):
        destination = ASSETS_DIR / asset_name(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WEBP_CACHE / f"{digest}.webp", destination)
        batch_bytes += destination.stat().st_size
        if batch_bytes >= PUSH_BATCH_BYTES or index == len(new) - 1:
            commit_all(ASSETS_DIR, f"[C] Add {batch_bytes / 1e6:.0f} MB of journal images (up to #{index + 1}/{len(new)})")
            if do_push:
                push_branch(ASSETS_DIR, ASSETS_BRANCH)
            batch_bytes = 0
    return len(new)


def build_site(urls, build_web=False):
    """
    Command. Rewrites SITE (keeping .git) from journal/: pages (image paths -> absolute asset URLs, plus footer),
    scripts, other small assets, the built app at webgpu/, a models placeholder. Returns the site size in bytes.
    """
    if build_web:
        subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=ROOT / "web", check=True)
        subprocess.run(["npm", "run", "build"], cwd=ROOT / "web", check=True)
    assert (JOURNAL / "webgpu/index.html").exists(), "journal/webgpu/ is missing: run with --build_web"
    for entry in SITE.iterdir():
        if entry.name != ".git":
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()

    for source in JOURNAL.glob("*"):
        text = source.read_text() if source.suffix in (".html", ".js", ".css") else None
        if source.suffix == ".html":
            (SITE / source.name).write_text(add_footer(rewrite_asset_refs(text, urls)))
        elif source.name == "eval_grid.js":
            (SITE / source.name).write_text(rewrite_eval_grid_js(rewrite_asset_refs(text, urls), urls))
        elif text is not None:
            (SITE / source.name).write_text(rewrite_asset_refs(text, urls))
    shutil.copytree(JOURNAL / "webgpu", SITE / "webgpu")
    (SITE / "models/web").mkdir(parents=True)
    (SITE / "models/web/index.json").write_text(json.dumps(MODELS_PLACEHOLDER, indent=1))
    (SITE / ".nojekyll").touch()
    for source in (p for p in (JOURNAL / "assets").rglob("*") if p.is_file() and p.suffix != ".png"):
        destination = SITE / source.relative_to(JOURNAL)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"site: {size / 1e6:.1f} MB", flush=True)
    assert size < PAGES_LIMIT_BYTES, f"site is {size / 1e6:.0f} MB, over the {PAGES_LIMIT_BYTES / 1e6:.0f} MB budget"
    return size


def publish(push=True, build_web=False):
    """Command. Pushes new images to journal-assets, then builds and pushes gh-pages (assets first: no broken links)."""
    open_worktree(ASSETS_DIR, ASSETS_BRANCH)
    open_worktree(SITE, SITE_BRANCH)
    pngs = sorted(p for p in (JOURNAL / "assets").rglob("*.png") if p.is_file())
    digests = encode_all(pngs)
    sync_assets(digests, push)
    if push:
        push_branch(ASSETS_DIR, ASSETS_BRANCH)  # also sends commits left unpushed by an earlier --push=False or failed run
    urls = {path: ASSET_BASE + asset_name(digest) for path, digest in digests.items()}
    build_site(urls, build_web)
    main_commit = git("rev-parse", "--short", "HEAD").strip()
    if not commit_all(SITE, f"[C] Publish journal and WebGPU app (main {main_commit})"):
        print("no change to gh-pages", flush=True)
    if push:
        push_branch(SITE, SITE_BRANCH)


if __name__ == "__main__":
    fire.Fire(publish)
