# /// script
# requires-python = ">=3.11"
# dependencies = ["fire==0.7.1", "pyarrow==25.0.1"]
# ///
"""
Writes rows.parquet: the h3_pexels_clips rows of one h3_import_jobs job, rows [row_offset, row_offset + row_count) of the
selection (fork PexelsPick appends rows, so each job imports one slice exactly once).
"""
import json
from pathlib import Path

import fire
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("pexels_id", pa.string()),
    ("source_path", pa.string()),
    ("split", pa.string()),
    ("birdseye_caption", pa.string()),
    ("duration", pa.float64()),
    ("fps", pa.float64()),
    ("flow_mean_magnitude", pa.float64()),
    ("width", pa.int64()),
    ("height", pa.int64()),
])


def job_rows(table, row_offset, row_count):
    """
    Pure function. The slice of the selection table that one import job owns, cast to SCHEMA.

    Args:
        table (pa.Table): The selection, with at least the SCHEMA columns (width and height may be float)
        row_offset (int): First row
        row_count (int): Rows

    Returns:
        pa.Table

    Examples:
        >>> t = pa.table({"pexels_id": ["1", "2", "3"], "source_path": ["a", "b", "c"], "split": ["train", "val", "spare"],
        ...               "birdseye_caption": ["x", "y", "z"], "duration": [7.0, 8.0, 9.0], "fps": [24.0, 25.0, 30.0],
        ...               "flow_mean_magnitude": [1.0, 2.0, 3.0], "width": [3840.0, 1920.0, 1280.0], "height": [2160.0, 1080.0, 720.0]})
        >>> job_rows(t, 1, 2).to_pydict()["pexels_id"], job_rows(t, 1, 2).schema.field("width").type
        (['2', '3'], DataType(int64))
    """
    assert row_offset + row_count <= table.num_rows, (row_offset, row_count, table.num_rows)
    part = table.slice(row_offset, row_count).select(SCHEMA.names)
    assert pc.all(pc.equal(part["width"], pc.round(part["width"]))).as_py(), "width is not whole"
    return part.cast(SCHEMA)


def main(sample, selection):
    """
    Command. Writes SAMPLE/rows.parquet from the job's slice of the selection parquet.

    Args:
        sample (str): The $SAMPLE folder; input.json has job_id, row_offset, row_count
        selection (str): H3_PEXELS_SELECTION
    """
    sample = Path(sample)
    job = json.loads((sample / "input.json").read_text())
    rows = job_rows(pq.read_table(selection), int(job["row_offset"]), int(job["row_count"]))
    pq.write_table(rows, sample / "rows.parquet")
    print(f"job {job['job_id']}: {rows.num_rows} clips, splits {pc.value_counts(rows['split']).to_pylist()}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
