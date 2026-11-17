#!/usr/bin/env python3
"""
Check and fix the hand-aligned TYPE / DESCRIPTION / EXAMPLE tables in .dtab files. Copied from MulticamDailies.
SEMANTIC BINDING: dtab-rules

Usage:
    python3 agents/dtab_format.py check FILE...   print FILE:LINE: problem for every misaligned table row; exit 1 if any
    python3 agents/dtab_format.py fix FILE...     realign table rows in place; print rows it cannot fix and exit 1 if any

Rules, from AGENTS.md "DTAB formatting":
- One tab renders as four spaces.
- In a row, `@` is followed immediately by the literal tab that starts the inline comment; padding goes after that tab.
- Each header hangs one character left of its data column: TYPE, DESCRIPTION and EXAMPLE, and FIELD (or SUBFIELD, for the keys inside a JSON column) for the
  field names of comment rows; a key inside a key sits one tab deeper.
- A description too long for one line continues on the next lines, each one tab deeper than the row, then spaces so its text starts two
  columns right of DESCRIPTION's data column. EXAMPLE goes on the last line, one right of its header; the earlier lines have none.
  No line ends in a space, so the break replaces the space between words.
- `[Key]  str`: the closing `]` sits one space before the TYPE header begins and two spaces before the type value.
- `~str` marks a materialized field copied unchanged to the commit output, with no other use, including diagnostic logging.
  The `~` replaces the space before the type, under TYPE; `[Key] ~str` preserves column positions. A space must remain after the comment tab.
- A key with no cells (`input.json` then its tab) nests the rows one tab deeper under it; they use the same header.
A table starts at a line holding TYPE, DESCRIPTION and EXAMPLE, and ends at a blank line, the next table, or a
shallower line. Headers never move. A row whose text cannot reach its column, or whose DESCRIPTION and EXAMPLE
cannot be told apart, is reported and left as it is.
"""
import re
import sys
from pathlib import Path

TAB = 4
HEADER = re.compile(r"(?:(?P<FIELD>(?:SUB)?FIELD) +)?(?P<TYPE>TYPE) +(?P<DESCRIPTION>DESCRIPTION) +(?P<EXAMPLE>EXAMPLE) *$")
CELLS = re.compile(r" *(?:(?P<key>\[Key\]) +)?(?P<type>\S+) +(?P<rest>\S.*)")


def columns(line: str) -> dict | None:
    """
    Pure. Return the rendered column of each header in a table header line, or None for any other line.

    >>> columns("\\t\\t\\t\\tfaces.parquet\\t             TYPE  DESCRIPTION                    EXAMPLE")
    {'TYPE': 45, 'DESCRIPTION': 51, 'EXAMPLE': 82}
    """
    match = HEADER.search(line.expandtabs(TAB))
    return {name: match.start(name) for name, text in match.groupdict().items() if text} if match else None


def row_cells(line: str, header: dict) -> tuple[str, int, list, list[str]]:
    """
    Pure. Split a table row into the text kept as it is, where its cells start, its cells and the problems in the kept text.

    Each cell is (header name, target column, text). Raises ValueError when the row cannot be split without guessing.
    """
    indent = line[: len(line) - len(line.lstrip("\t"))]
    body, rendered, problems = line[len(indent) :], line.expandtabs(TAB), []
    if body.startswith(" "):  # A comment row: its field name sits under FIELD.
        if "FIELD" not in header:
            raise ValueError("comment row in a table without a FIELD header")
        name = body.split()[0]
        prefix, consumed = indent, indent + body[: body.index(name) + len(name)]
        nested = max(rendered.index(name) - header["FIELD"] - 1, 0) // TAB * TAB  # a key inside a key sits one tab deeper per level
        cells, cursor = [("FIELD", header["FIELD"] + 1 + nested, name)], len(indent.expandtabs(TAB))
    else:
        if "\t" in body:
            key = body.split("\t", 1)[0]
            consumed = indent + key + "\t"
        elif " @ " in body:
            key = body[: body.index(" @ ") + 2]
            consumed = indent + key
            problems.append("`@` must be followed immediately by the literal tab that starts the inline comment")
        else:
            raise ValueError("cannot find the tab that ends the key")
        if key != key.rstrip(" ") and key.rstrip(" ").endswith("@"):
            key = key.rstrip(" ")
            problems.append("spaces between `@` and its tab become part of the source-field name")
        prefix, cells = indent + key + "\t", []
        cursor = len(consumed.expandtabs(TAB))
    rest = rendered[len(consumed.expandtabs(TAB)) :]
    if rest != rest.rstrip(" "):
        raise ValueError("trailing spaces")
    match = CELLS.fullmatch(rest)
    if not match:
        raise ValueError("cannot find TYPE and DESCRIPTION")
    texts = re.split(" {2,}", match["rest"])
    if len(texts) > 2:
        raise ValueError("cannot tell DESCRIPTION from EXAMPLE: more than one gap of two or more spaces")
    if len(texts) == 1 and len(rendered) > header.get("EXAMPLE", len(rendered)):
        raise ValueError("DESCRIPTION runs under EXAMPLE: separate EXAMPLE by two or more spaces")
    if match["key"]:
        cells.append(("[Key]", header["TYPE"] - 6, "[Key]"))
    cells.append(("TYPE", header["TYPE"] + (0 if match["type"].startswith("~") else 1), match["type"]))
    cells += [(name, header[name] + 1, text) for name, text in zip(("DESCRIPTION", "EXAMPLE"), texts)]
    return prefix, cursor, cells, problems


def align_continuation(line: str, header: dict) -> tuple[str, list[str]]:
    """
    Pure. Return a continuation line with its text two right of DESCRIPTION's data column and its EXAMPLE, if any, aligned.

    >>> header = columns("\\t\\t\\t\\tfaces.parquet\\t             TYPE  DESCRIPTION                    EXAMPLE")
    >>> align_continuation("\\t\\t\\t\\t\\t\\t                                    more text  0.77", header)[0]
    '\\t\\t\\t\\t\\t\\t                             more text                     0.77'
    """
    indent = line[: len(line) - len(line.lstrip("\t"))]
    body = line[len(indent) :]
    if body != body.rstrip(" "):
        raise ValueError("trailing spaces")
    texts = re.split(" {2,}", body.strip(" "))
    if len(texts) > 1 + ("EXAMPLE" in header):
        raise ValueError("cannot tell DESCRIPTION from EXAMPLE: more than one gap of two or more spaces")
    out = indent + " " * (header["DESCRIPTION"] + 2 - len(indent) * TAB) + texts[0]
    if len(texts) == 2:
        gap = header["EXAMPLE"] + 1 - len(out.expandtabs(TAB))
        if gap < 1:
            raise ValueError(f"EXAMPLE cannot reach column {header['EXAMPLE'] + 2}; widen the table by hand")
        out += " " * gap + texts[1]
    problems = [] if out == line else ["continuation text must start two right of DESCRIPTION's data column, EXAMPLE one right of its header"]
    return out, problems


def align_row(line: str, header: dict) -> tuple[str, list[str]]:
    """
    Pure function. Align cells one right of their headers; a type's optional ~ sits under TYPE.

    Args:
        line: One DTAB table row.
        header: Rendered header columns from columns().
    Returns:
        tuple: Aligned row and a list of reported problems.

    >>> header = columns("\\t\\t\\t\\tfaces.parquet\\t             TYPE  DESCRIPTION                    EXAMPLE")
    >>> align_row("\\t\\t\\t\\t\\tbox_x0 @\\t              float Normalized Proxy box left        0.77", header)
    ('\\t\\t\\t\\t\\tbox_x0 @\\t              float Normalized Proxy box left      0.77', ['EXAMPLE starts at column 86; it must start at 84, one right of its header'])
    >>> align_row("\\t\\t\\t\\t\\tface_index @       [Key]  int   Face index", header)[1]
    ['`@` must be followed immediately by the literal tab that starts the inline comment']
    >>> align_row("\\t\\t\\t\\t\\tface_index @\\t   [Key]  int   Face index", header)
    ('\\t\\t\\t\\t\\tface_index @\\t   [Key]  int   Face index', [])
    >>> align_row("\\t\\t\\t\\t\\tface_index @\\t   [Key] ~int   Face index", header)
    ('\\t\\t\\t\\t\\tface_index @\\t   [Key] ~int   Face index', [])
    >>> align_row("\\t\\t\\t\\t\\tclip_id @\\t             ~str   Copied for commit", header)
    ('\\t\\t\\t\\t\\tclip_id @\\t             ~str   Copied for commit', [])
    >>> align_row("\\t\\t\\t\\t\\tface_index @\\t   [Key]  ~int   Face index", header)[0]
    '\\t\\t\\t\\t\\tface_index @\\t   [Key] ~int   Face index'
    >>> align_row("\\tb @\\t~str   Copied for commit", {"TYPE": 8, "DESCRIPTION": 14, "EXAMPLE": 60})
    Traceback (most recent call last):
        ...
    ValueError: TYPE cannot reach column 9; widen the table by hand
    """
    prefix, cursor, cells, problems = row_cells(line, header)
    rendered, out, width = line.expandtabs(TAB), prefix, len(prefix.expandtabs(TAB))
    for index, (name, target, text) in enumerate(cells):
        start = rendered.find(text, cursor)
        cursor = start + len(text)
        if start != target:
            rule = "so `]` sits one space before TYPE" if name == "[Key]" else "one right of its header"
            if name == "TYPE" and text.startswith("~"):
                rule = "so `~` sits under TYPE and the type stays one right of its header"
            problems.append(f"{name} starts at column {start + 1}; it must start at {target + 1}, {rule}")
        gap = target - width
        if gap < (0 if index == 0 and prefix.endswith("\t") and not text.startswith("~") else 1):
            raise ValueError(f"{name} cannot reach column {target + 1}; widen the table by hand")
        out, width = out + " " * gap + text, target + len(text)
    if out != line and not problems:
        problems.append("padding after the inline tab must be spaces")
    return out, problems


def format_dtab(text: str) -> tuple[str, list, list]:
    """
    Pure. Return the text with its table rows aligned, the (line, problem) it fixed and the (line, problem) it could not.

    >>> format_dtab("x\\t\\t TYPE  DESCRIPTION  EXAMPLE\\n\\tb @\\t  str   Way too long a description  1\\n")[2]
    [(2, 'EXAMPLE cannot reach column 30; widen the table by hand')]
    >>> format_dtab("x\\t\\t\\tTYPE  DESCRIPTION  EXAMPLE\\n\\tj\\t   \\n\\t\\tb @\\t str   B            1\\n")[1:]
    ([], [])
    """
    lines, fixed, unfixed = text.split("\n"), [], []
    header = depth = None
    for number, line in enumerate(lines, 1):
        if columns(line):
            header, depth = columns(line), None
            continue
        tabs = len(line) - len(line.lstrip("\t"))
        if header is None or not line.strip() or (depth is not None and tabs < depth):
            header = None
            continue
        depth = tabs if depth is None else depth
        body = line[tabs:]
        if re.fullmatch(r"[^\t ]+\t *", body):
            continue  # A key with no cells: the rows under it are nested rows
        try:
            align = align_continuation if tabs > depth and body.startswith(" ") else align_row
            lines[number - 1], problems = align(line, header)
        except ValueError as error:
            unfixed.append((number, str(error)))
            continue
        fixed += [(number, problem) for problem in problems]
    return "\n".join(lines), fixed, unfixed


def main(mode: str, *files: str) -> None:
    """Command. Check or fix each file; exit 1 when a row is misaligned (check) or cannot be fixed (fix)."""
    if mode not in ("check", "fix") or not files:
        raise SystemExit(__doc__)
    failed = False
    for file in files:
        text = Path(file).read_text()
        result, fixed, unfixed = format_dtab(text)
        for number, problem in (fixed if mode == "check" else []) + unfixed:
            print(f"{file}:{number}: {problem}")
        failed = failed or bool(unfixed) or (mode == "check" and bool(fixed))
        if mode == "fix" and result != text:
            Path(file).write_text(result)
            print(f"{file}: fixed {len({number for number, _ in fixed})} rows")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(*sys.argv[1:])
