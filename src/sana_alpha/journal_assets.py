"""
Journal images at full 1024x1024 resolution: hard-links the files BlueJay stored (no copies, no disk cost).

    uv run python -m sana_alpha.journal_assets          # recreates journal/assets/eval/full/ (git ignores it)

journal/assets/eval/full/<model_tag>/<page_id>.png   every eval_generations image (blob files in data/lance)
journal/assets/eval/full/matte/<page_id>.png         resources/eval/mattes
journal/assets/eval/full/reference/<page_id>.png     resources/eval/references (the MAGICK image on gray)
journal/assets/eval/full/samples/<run>/step_N.png    models/adapter/<run>/samples (training sample sheets, 2048 x 1024)
"""

import os
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "journal/assets/eval/full"
LANCE = ROOT / "data/lance/eval_generations.lance"


def link(source, destination):
    """Command. Hard-links source to destination (replacing a stale file there), creating the folder."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    os.link(source, destination)


def link_all(*tags, out=FULL):
    """
    Command. Hard-links the images of the given eval_models tags (default: all) into OUT/<tag>/<page_id>.png, plus
    the mattes, MAGICK references and adapter training sample sheets. Raises if a blob file is missing.

    Args:
        *tags (str): eval_models tags, e.g. adapter_v1_step002000. Default: every tag with generated images.
        out (Path): destination folder
    """
    import lance

    where = "image IS NOT NULL" + (" AND model_tag IN (%s)" % ", ".join("'%s'" % tag for tag in tags) if tags else "")
    rows = lance.dataset(str(LANCE)).to_table(columns=["model_tag", "page_id", "image"], filter=where).to_pylist()
    for row in rows:
        link(LANCE / row["image"]["blob_uri"], Path(out) / row["model_tag"] / (row["page_id"] + ".png"))
    for kind, folder in dict(matte="mattes", reference="references").items():
        for path in (ROOT / "resources/eval" / folder).glob("*.png"):
            link(path, Path(out) / kind / path.name)
    for path in (ROOT / "models/adapter").glob("*/samples/*.png"):
        link(path, Path(out) / "samples" / path.parent.parent.name / path.name)
    print("linked %d generations into %s" % (len(rows), out))


if __name__ == "__main__":
    fire.Fire(link_all)
