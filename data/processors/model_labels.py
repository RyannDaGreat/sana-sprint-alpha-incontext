"""
The model_labels Project SQL function: the journal's weight equation of eval_models rows (journal_assets.label_of).
Runs in the repo's uv project, like generate_eval.py. Each call takes lists, so that one query starts one process:
DataFusion calls a function once per record batch, and eval_models has many small batches (~1.5 s per process).
"""

import sys

import pyarrow as pa
import pyarrow.parquet as pq

from sana_alpha.journal_assets import label_of


def main(sample):
    """Command. Reads sample/models.parquet (list columns pipeline, checkpoint, steps, prompt_variant), writes sample/labels.parquet."""
    calls = pq.read_table(f"{sample}/models.parquet").to_pylist()
    labels = [[label_of(*row) for row in zip(c["pipeline"], c["checkpoint"], c["steps"], c["prompt_variant"])] for c in calls]
    pq.write_table(pa.table(dict(label=labels)), f"{sample}/labels.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
