# sana-sprint-alpha-incontext

Experiment: train SANA-Sprint to turn an alpha matte into the RGB foreground on a 50% gray background, in-context.

**Journal (live, all images at full 1024×1024): https://ryanndagreat.github.io/sana-sprint-alpha-incontext/**

- The site is the `gh-pages` branch, built by `PYTHONPATH=src uv run python -m sana_alpha.publish_pages` (images are WebP q95 there; the PNG originals are in the BlueJay database, `data/project.dtab`).
- Agent instructions: `AGENTS.md` and `agents/`. Non-commercial research use (trained on MAGICK, CC BY-NC 4.0).
