# sana-sprint-alpha-incontext

Planning repo for an experiment: train SANA-Sprint (via the AnimateDiff-style adapter trick) to turn an alpha matte into the RGB foreground on a 50% gray background, in-context.

**Before planning or changing files, read everything in `agents/`:** `agents/manifest.md` (the project), `agents/concerns.md` (history and lessons), `agents/todo.md` (open work), `agents/transcripts.md` (what the user actually said).

- AI scratch work goes in `agents/.scratchpad/`; logs go in `agents/.scratchpad/logs/`.
- The human-facing journal goes in `journal/`. Live copy: https://ryanndagreat.github.io/sana-sprint-alpha-incontext/ (republish with `PYTHONPATH=src uv run python -m sana_alpha.publish_pages`).
- All paths are relative to the repo root.

## DTAB formatting — preserve this exactly

<!-- SEMANTIC BINDING: dtab-rules -->
Copied from MulticamDailies. `data/project.dtab` is the BlueJay project. Edit it by hand: it is the source, not a generated file.

- Render one tab as four spaces in every editor.
- After you edit a DTAB, run `python3 agents/dtab_format.py check data/project.dtab`. `fix` realigns the TYPE / DESCRIPTION / EXAMPLE tables. This is the agent's job, not the user's.
- `python3 agents/dtab_fold.py data/project.dtab FIELD...` wraps long descriptions onto continuation lines.
- Formatting is part of the human-facing interface. Before you edit, look at the nearby blocks and copy their tabs, padding, column widths and comment places exactly. Never normalize a block the user aligned by hand.
- `@` is followed immediately by the literal tab that starts the inline comment. Padding goes after that tab.
- Each header (TYPE, DESCRIPTION, EXAMPLE, FIELD, SUBFIELD) hangs one character left of its data column. `[Key]  str`: the `]` sits one space before TYPE.
- A key with no cells (`input.json`, `result.json`) carries the header on its own line, and its rows go one tab deeper.
- In a run of one-line queries, the SQL starts at one shared column. Keep that alignment: `dtab_format.py` does not check it.
- When you change a comment, change only the text after its tab.
- The rest of the rules are in the `RULES FOR MAINTAINING THIS DTAB` block at the top of `data/project.dtab`.

## Processors

- Each processor in `data/processors/` is a PEP 723 script, self-contained through its inline dependencies, run with `uv run --script`.
- Every function docstring starts with `Pure function.`, `Query.` or `Command.`
