#!/usr/bin/env python3
"""
Fold the descriptions of chosen fields in .dtab TYPE / DESCRIPTION / EXAMPLE tables, and unfold every other one.

Usage:
    python3 agents/dtab_fold.py FILE [FIELD...]   fold each named FIELD's description, unfold the rest, then realign

A table that holds a folded description or a named FIELD gets its EXAMPLE header moved to two columns past its longest
unfolded description, so no gap is wider than it needs to be; every other table keeps its hand-set header. A named
FIELD then folds to fit: continuation lines follow agents/dtab_format.py, which realigns the file and must pass.
"""
import re
import sys
import textwrap
from pathlib import Path

from dtab_format import TAB, columns, format_dtab, row_cells


def tables(lines: list[str]):
    """
    Pure. Yield (header line index, [row line indexes]) for every table; a row's continuation lines are not rows.

    >>> list(tables(["x\\t TYPE  DESCRIPTION  EXAMPLE", "\\ta @\\t str   Text  1", "\\t\\t     more", "", "\\tb"]))
    [(0, [1])]
    """
    start = depth = None
    rows = []
    for index, line in enumerate(lines + [""]):
        tabs = len(line) - len(line.lstrip("\t"))
        if start is not None and (columns(line) or not line.strip() or (depth is not None and tabs < depth)):
            yield start, rows
            start = None
        if columns(line):
            start, depth, rows = index, None, []
        elif start is not None:
            depth = tabs if depth is None else depth
            if tabs == depth:
                rows.append(index)


def cells(line: str, header: dict) -> dict:
    """
    Pure. Return a row's cell texts by header name.

    >>> cells("\\ta @\\t str   Text  1", columns("x\\t TYPE  DESCRIPTION  EXAMPLE"))
    {'TYPE': 'str', 'DESCRIPTION': 'Text', 'EXAMPLE': '1'}
    """
    return {name: text for name, _, text in row_cells(line, header)[2] if name != "FIELD"}


def unfold(lines: list[str]) -> tuple[list[str], set[int]]:
    """
    Pure. Join every continuation line onto its row; return the lines and the header indexes of tables that changed.

    >>> unfold(["x\\t TYPE  DESCRIPTION  EXAMPLE", "\\ta @\\t str   Long", "\\t\\t       text  1"])
    (['x\\t TYPE  DESCRIPTION  EXAMPLE', '\\ta @\\t str   Long text  1'], {0})
    """
    out, changed, start, depth = [], set(), None, None
    for line in lines:
        tabs = len(line) - len(line.lstrip("\t"))
        if columns(line):
            start, depth = len(out), None
        elif start is None or not line.strip() or (depth is not None and tabs < depth):
            start = None
        elif depth is None or tabs == depth:
            depth = tabs
        else:
            texts = re.split(" {2,}", line.strip(" \t"))
            out[-1] += " " + texts[0] + "".join("  " + text for text in texts[1:])
            changed.add(start)
            continue
        out.append(line)
    return out, changed


def fold(line: str, header: dict) -> list[str]:
    """
    Pure. Split a row so its description fits before EXAMPLE, the example on the last line; dtab_format aligns them.

    >>> fold("\\ta @\\t str   One two three four  1", columns("x\\t TYPE  DESCRIPTION       EXAMPLE"))
    ['\\ta @\\t str   One two three', '\\t\\t four  1']
    """
    texts = cells(line, header)
    first, *rest = textwrap.wrap(texts["DESCRIPTION"], header["EXAMPLE"] - header["DESCRIPTION"] - 2)
    rest = textwrap.wrap(" ".join(rest), header["EXAMPLE"] - header["DESCRIPTION"] - 4)
    tabs = len(line) - len(line.lstrip("\t"))
    folded = [line[: line.index(texts["DESCRIPTION"])] + first] + ["\t" * (tabs + 1) + " " + text for text in rest]
    folded[-1] += "  " + texts["EXAMPLE"] if "EXAMPLE" in texts else ""
    return folded


def narrow(header_line: str, column: int) -> str:
    """
    Pure. Move a header line's EXAMPLE to a rendered column.

    >>> narrow("x\\t TYPE  DESCRIPTION       EXAMPLE", 25)
    'x\\t TYPE  DESCRIPTION   EXAMPLE'
    """
    kept = header_line[: header_line.rindex("EXAMPLE")].rstrip(" ")
    return kept + " " * (column - len(kept.expandtabs(TAB))) + "EXAMPLE"


def fold_dtab(text: str, fields: set[str]) -> tuple[str, list]:
    """
    Pure. Return the text with only the named fields' descriptions folded, and the (line, problem) dtab_format could not fix.

    >>> fold_dtab("x\\t TYPE  DESCRIPTION  EXAMPLE\\n\\ta @\\t str   Short       1\\n\\tb @\\t str   One two three four  2\\n", {"b"})[0]
    'x\\t TYPE  DESCRIPTION  EXAMPLE\\n\\ta @\\t str   Short       1\\n\\tb @\\t str   One two\\n\\t\\t     three\\n\\t\\t     four        2\\n'
    """
    lines, changed = unfold(text.split("\n"))
    folds = []
    for start, rows in tables(lines):
        header = columns(lines[start])
        named = [index for index in rows if lines[index].split()[0] in fields]
        if start not in changed and not named:
            continue
        ends = [header["DESCRIPTION"] + 1 + len(cells(lines[index], header)["DESCRIPTION"]) for index in rows if index not in named]
        lines[start] = narrow(lines[start], max(*ends, header["DESCRIPTION"] + len("DESCRIPTION") + 1) + 1)
        folds += [(index, start) for index in named]
    for index, start in sorted(folds, reverse=True):
        lines[index : index + 1] = fold(lines[index], columns(lines[start]))
    result, _, unfixed = format_dtab("\n".join(lines))
    return result, unfixed


def main(file: str, *fields: str) -> None:
    """Command. Fold the named fields in FILE, unfold the rest, realign; exit 1 when a row cannot be aligned."""
    path = Path(file)
    result, unfixed = fold_dtab(path.read_text(), set(fields))
    for number, problem in unfixed:
        print(f"{file}:{number}: {problem}")
    path.write_text(result)
    sys.exit(1 if unfixed else 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
