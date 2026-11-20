"""
Publishes the journal (all pages, every image at full 1024x1024 resolution) and the WebGPU app to GitHub.

    PYTHONPATH=src uv run python -m sana_alpha.publish_pages                # build, commit, push both branches
    PYTHONPATH=src uv run python -m sana_alpha.publish_pages --push=False   # build and commit only
    PYTHONPATH=src uv run python -m sana_alpha.publish_pages --build_web    # also run the app's `npm run build` first

Live site: https://ryanndagreat.github.io/sana-sprint-alpha-incontext/

Two orphan branches, each checked out as a git worktree under agents/.scratchpad/ so main's working tree is never touched:

- `journal-assets`: every journal PNG, byte for byte (original resolution, lossless), named by its SHA-256
  (`ab/abcdef0123456789.png`). Identical images (shared mattes, references, stock-model columns, eval/full versus
  gallery copies) are stored once, and a republish only uploads images whose hash is new. Append-only: old files are
  never deleted, so links in older pages keep working. No site-size budget here (GitHub: <100 MB per file); each push
  is capped well under GitHub's 2 GB per-push limit by pushing in batches.
- `gh-pages`: the HTML, JS, small non-PNG assets and the WebGPU app (GitHub Pages' 1 GB site limit applies). Pages
  reference images by absolute URL into `journal-assets` (ASSET_BASE; raw.githubusercontent.com serves
  `image/png` with `access-control-allow-origin: *`).

Why the original PNGs: the user wants every image at original resolution and lossless (m00705). Before 2026-11-19
this published WebP q95; those files stay on the branch (append-only) but nothing references them. Models are not
published here: publish_models.py uploads them to the one Hugging Face repo HF_REPO (hf_upload), which the app reads
on Pages; models/web/index.json here is a placeholder. Why Hugging Face for models (manifest §12 "Hosting", m00727):
GitHub refused every push once the repo passed its size quota with the 88.65 GB `models` branch; the journal images
stay here because the Hub allows only 20,000 files per repo and the journal has ~46k PNGs.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

import fire

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "journal"
SITE = ROOT / "agents/.scratchpad/gh-pages"
ASSETS_DIR = ROOT / "agents/.scratchpad/journal-assets"
SITE_BRANCH = "gh-pages"
ASSETS_BRANCH = "journal-assets"
REPO = "RyannDaGreat/sana-sprint-alpha-incontext"
HF_REPO = "OneOverZero/sana-sprint-alpha-incontext"  # the one Hugging Face repo, for the models (m00727)
HF_DIR = ROOT / "agents/.scratchpad/hf-repo"  # local mirror of HF_REPO: blobs/, web/ (publish_models)
ASSET_BASE = f"https://raw.githubusercontent.com/{REPO}/{ASSETS_BRANCH}/"
DIGEST_CHARS = 16  # 64 bits of SHA-256: no collisions at this scale
SHARD_CHARS = 2  # first hex chars as a folder, so no folder holds thousands of files
HASH_PROCESSES = 8  # the training jobs need the CPUs
PAGES_LIMIT_BYTES = 900e6  # GitHub Pages allows 1 GB per site; keep headroom
PUSH_BATCH_BYTES = 400e6  # GitHub rejects pushes over 2 GB, and a 2.4 GB push got HTTP 500 (publish_models): push in batches of this size
NEW_COUNT = 3  # index entries marked "new" (m00719: the user could not tell which journals are the latest)
INDEX_ENTRY = re.compile(r'^<li><a href="([^"]+)">')
STAMP_STYLE = "margin:0.5em 1em;font:13px system-ui,sans-serif;opacity:0.75"
SUBDIRS = ("compare", "lib")  # journal/ folders of text files (HTML, JS, CSS, JSON), copied with their image paths rewritten (manifest §15)
EVAL_FULL_DIR = "assets/eval/full/"
EVAL_JS_REFERENCE = "${EVAL_FULL_DIR}${dir}/${id}.png"  # the one dynamic image path, in eval_grid.js
FOOTER = ('<footer style="margin:3em 0 1em;font-size:0.85em;opacity:0.7">Published images are the original PNG files '
          "(original resolution, lossless), hosted on the repo's journal-assets branch.</footer>\n")
MODELS_PLACEHOLDER = {"versions": [], "note": f"On Pages the app reads models from the Hugging Face repo {HF_REPO} "
                                                "(web/); see src/sana_alpha/publish_models.py."}


def asset_name(digest):
    """
    Pure function. Maps a PNG's SHA-256 hex digest to its path inside the journal-assets branch.

    Examples:
        >>> asset_name("ab" + "cd" * 31)
        'ab/abcdcdcdcdcdcdcd.png'
    """
    return f"{digest[:SHARD_CHARS]}/{digest[:DIGEST_CHARS]}.png"


def rewrite_asset_refs(text, urls):
    """
    Pure function. Replaces every literal `assets/....png` path in HTML or JS with its absolute asset URL. Fails
    loudly (KeyError) on a path with no entry in `urls`. Prose says "PNG" in capitals, so it is left alone.

    Args:
        text (str): HTML or JS source
        urls (dict[str, str]): path relative to journal/ (`assets/x/y.png`) -> absolute URL

    Examples:
        >>> rewrite_asset_refs('<img src="assets/a/b.png"> a 1024x1024 PNG', {"assets/a/b.png": "https://h/ab/cd.png"})
        '<img src="https://h/ab/cd.png"> a 1024x1024 PNG'
    """
    return re.sub(r"assets/[^\"'\s<>`]+?\.png\b", lambda match: urls[match.group(0)], text)


def rewrite_eval_grid_js(text, urls):
    """
    Pure function. Rewrites eval_grid.js's one dynamic path (`${EVAL_FULL_DIR}${dir}/${id}.png`) to a lookup into a
    map `EVAL_URLS` of "dir/id" -> asset path, appended to the script. A missing entry throws in the browser.

    Examples:
        >>> js = rewrite_eval_grid_js('x = `${EVAL_FULL_DIR}${dir}/${id}.png`, `${EVAL_FULL_DIR}${dir}/${id}.png`;',
        ...                           {"assets/eval/full/matte/a.png": "https://h/ab/cd.png"})
        >>> js.startswith('x = `${evalUrl(dir, id)}`, `${evalUrl(dir, id)}`;')
        True
        >>> '"matte/a": "https://h/ab/cd.png"' in js
        True
    """
    assert text.count(EVAL_JS_REFERENCE) == 2, "eval_grid.js changed: update EVAL_JS_REFERENCE handling"
    eval_urls = {path.removeprefix(EVAL_FULL_DIR).removesuffix(".png"): url
                 for path, url in urls.items() if path.startswith(EVAL_FULL_DIR)}
    helper = ("\n\n// Generated by publish_pages.py: the original PNGs, hosted on the journal-assets branch.\n"
              f"const EVAL_URLS = {json.dumps(eval_urls, indent=0)};\n"
              "function evalUrl(dir, id) {\n"
              "  const url = EVAL_URLS[`${dir}/${id}`];\n"
              "  if (!url) throw new Error(`no published image for ${dir}/${id}`);\n"
              "  return url;\n"
              "}\n")
    return text.replace(EVAL_JS_REFERENCE, "${evalUrl(dir, id)}") + helper


def add_footer(html):
    """
    Pure function. Inserts the footer note (where the images live) just before </body>.

    Examples:
        >>> add_footer("<body><p>x</p></body>").startswith("<body><p>x</p><footer")
        True
    """
    assert html.count("</body>") == 1, "expected exactly one </body>"
    return html.replace("</body>", FOOTER + "</body>")


def page_sources(href):
    """
    Pure function. The repo paths whose git history dates a published page: the page file itself; for the compare
    reports their shells, data and shared scripts; for the app its source.

    Examples:
        >>> page_sources("gallery_v2.html"), page_sources("compare/scrub.html"), page_sources("webgpu/")
        (['journal/gallery_v2.html'], ['journal/compare', 'journal/lib'], ['web/src', 'web/index.html'])
    """
    if href.startswith("compare/"):
        return ["journal/compare", "journal/lib"]
    if href.startswith("webgpu"):
        return ["web/src", "web/index.html"]
    return [f"journal/{href}"]


def last_commit_time(paths):
    """Query (git). Unix time of the newest commit touching any of `paths`; fails loudly when none has history."""
    stamp = git("log", "-1", "--format=%ct", "--", *paths).strip()
    assert stamp, f"no git history for {paths}"
    return int(stamp)


def eastern(timestamp):
    """
    Pure function. A Unix time as US Eastern date and 12-hour time.

    Examples:
        >>> eastern(1763542800)
        'Nov 19, 2025, 4:00 AM ET'
    """
    when = datetime.fromtimestamp(timestamp, ZoneInfo("America/New_York"))
    return f"{when:%b} {when.day}, {when.year}, {when.hour % 12 or 12}:{when:%M} {'AM' if when.hour < 12 else 'PM'} ET"


def add_updated(html, timestamp):
    """
    Pure function. Inserts a "Last updated" line right after the <body ...> tag.

    Examples:
        >>> add_updated('<body data-view="x"><main>', 1763542800)
        '<body data-view="x"><p class="updated-stamp" style="margin:0.5em 1em;font:13px system-ui,sans-serif;opacity:0.75">Last updated Nov 19, 2025, 4:00 AM ET</p><main>'
    """
    match = re.search(r"<body[^>]*>", html)
    assert match, "expected a <body> tag"
    stamp = f'<p class="updated-stamp" style="{STAMP_STYLE}">Last updated {eastern(timestamp)}</p>'
    return html[:match.end()] + stamp + html[match.end():]


def order_index(html, times, new_count=NEW_COUNT):
    r"""
    Pure function. Sorts the index's entries (one `<li><a href="...">` per line) newest first by `times[href]`,
    putting each entry's date and time, and a "new" mark on the newest `new_count`, right after its title link. A
    listed href without a time fails loudly (KeyError).

    Examples:
        >>> html = '<ul>\n<li><a href="a.html">A</a></li>\n<li><a href="b.html">B</a></li>\n</ul>'
        >>> print(order_index(html, {"a.html": 1763542800, "b.html": 1763546400}, new_count=1))
        <ul>
        <li><a href="b.html">B</a> <span class="updated">Nov 19, 2025, 5:00 AM ET</span> <b class="new">new</b></li>
        <li><a href="a.html">A</a> <span class="updated">Nov 19, 2025, 4:00 AM ET</span></li>
        </ul>
    """
    lines = html.split("\n")
    rows = [k for k, line in enumerate(lines) if INDEX_ENTRY.match(line)]
    assert rows == list(range(rows[0], rows[-1] + 1)), "index entries must be consecutive lines"
    entries = sorted((lines[k] for k in rows), key=lambda line: -times[INDEX_ENTRY.match(line).group(1)])
    mark = lambda line, rank: (f' <span class="updated">{eastern(times[INDEX_ENTRY.match(line).group(1)])}</span>'
                               + (' <b class="new">new</b>' if rank < new_count else ""))
    stamped = [line.replace("</a>", "</a>" + mark(line, rank), 1) for rank, line in enumerate(entries)]
    return "\n".join(lines[:rows[0]] + stamped + lines[rows[-1] + 1:])


def png_digest(png_path):
    """Query. SHA-256 hex digest of a PNG file's bytes."""
    return hashlib.sha256(Path(png_path).read_bytes()).hexdigest()


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


def hf_upload():
    """
    Command (network). Uploads the local model mirror HF_DIR to HF_REPO with huggingface_hub (2.x) upload_folder: with
    hf_xet it streams, commits large folders in several batches, resumes when rerun, and skips files already
    committed, so a republish sends only new content-hashed parts. The token comes only from the HF_TOKEN environment
    variable (m00727: never stored in the dump). HF_DIR/.gitattributes routes blobs/** to large-file (Xet) storage:
    without it the Hub refuses the small parts as plain-git binaries.
    """
    from huggingface_hub import HfApi
    assert os.environ.get("HF_TOKEN"), "set HF_TOKEN to upload to Hugging Face"
    HfApi().upload_folder(repo_id=HF_REPO, repo_type="model", folder_path=HF_DIR,
                          commit_message=f"Publish model versions (main {git('rev-parse', '--short', 'HEAD').strip()})")


def commit_all(path, message):
    """Command. Stages everything in the worktree at `path` and commits; returns whether anything was committed."""
    git("add", "-A", cwd=path)
    if not git("status", "--porcelain", cwd=path).strip():
        return False
    git("commit", "-q", "-m", message, cwd=path)
    return True


def hash_all(pngs):
    """Query. SHA-256 of every PNG (parallel, HASH_PROCESSES); returns {path relative to journal/: digest}."""
    print(f"hashing {len(pngs)} PNGs", flush=True)
    with ProcessPoolExecutor(HASH_PROCESSES) as pool:
        digests = list(pool.map(png_digest, pngs, chunksize=32))
    return {p.relative_to(JOURNAL).as_posix(): digest for p, digest in zip(pngs, digests)}


def sync_assets(digests, sources, do_push):
    """
    Command. Copies the PNG of every digest not yet in the journal-assets worktree, committing (and pushing) in
    batches of PUSH_BATCH_BYTES. Never deletes. Returns the number of files added.
    """
    new = sorted(d for d in set(digests.values()) if not (ASSETS_DIR / asset_name(d)).exists())
    print(f"journal-assets: {len(set(digests.values()))} unique images, {len(new)} new", flush=True)
    batch_bytes = 0
    for index, digest in enumerate(new):
        destination = ASSETS_DIR / asset_name(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[digest], destination)
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

    times = {href: last_commit_time(page_sources(href)) for href in
             [p.name for p in JOURNAL.glob("*.html")] + [f"compare/{p.name}" for p in (JOURNAL / "compare").glob("*.html")] + ["webgpu/"]}
    for source in JOURNAL.glob("*"):
        text = source.read_text() if source.suffix in (".html", ".js", ".css") else None
        if source.name == "index.html":
            (SITE / source.name).write_text(add_footer(order_index(rewrite_asset_refs(text, urls), times)))
        elif source.suffix == ".html":
            (SITE / source.name).write_text(add_footer(add_updated(rewrite_asset_refs(text, urls), times[source.name])))
        elif source.name == "eval_grid.js":
            (SITE / source.name).write_text(rewrite_eval_grid_js(rewrite_asset_refs(text, urls), urls))
        elif text is not None:
            (SITE / source.name).write_text(rewrite_asset_refs(text, urls))
    for source in (p for d in SUBDIRS for p in (JOURNAL / d).rglob("*") if p.is_file()):  # the compare reports
        destination = SITE / source.relative_to(JOURNAL)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = rewrite_asset_refs(source.read_text(), urls)
        page = source.relative_to(JOURNAL).as_posix()
        destination.write_text(add_footer(add_updated(text, times[page])) if source.suffix == ".html" else text)
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
    digests = hash_all(pngs)
    sources = {digest: JOURNAL / path for path, digest in digests.items()}
    sync_assets(digests, sources, push)
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
