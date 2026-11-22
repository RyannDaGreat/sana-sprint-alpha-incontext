"""
H3 text embeddings of the H3 Magic project (manifest §22): long-lived text-encoder workers (Qwen3-VL-32B, ~67 GB),
one per GPU, committed with the database's commit. ~0.1 s per caption on an A100.

Each blob is safetensors {"prompt_embeds": (1, N, 5120) bfloat16, "text_token_tags": (N,) int64}: hidden state 50 of the
t2va text encoder, exactly what H3's denoiser takes (h3magic/h3.py embed). Column h3_text_embed of:
    h3_pexels_clips   (h3_caption)     h3_magick_images  (h3_caption)     h3_eval_prompts  (prompt)

    h3_embed.py WORKER WORKERS [--poll_seconds 120]   (from data/; GPU = $CUDA_VISIBLE_DEVICES)
"""

import os
import sys

import fire
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from h3_blobs import round_robin, shard_sql, tensors_bytes
from h3magic import h3

TABLES = {  # table: (key, text column, rows per query)
    "h3_eval_prompts": ("prompt_id", "prompt", 64),
    "h3_pexels_clips": ("pexels_id", "h3_caption", 256),
    "h3_magick_images": ("page_id", "h3_caption", 256),
}


def table_sql(table, worker, workers):
    """
    Pure function. The query of one table's texts that have no embedding yet, of this worker.

    Examples:
        >>> table_sql("h3_magick_images", 0, 1).startswith("SELECT page_id, h3_caption AS text FROM h3_magick_images WHERE h3_text_embed IS NULL")
        True
    """
    key, text, limit = TABLES[table]
    return (f"SELECT {key}, {text} AS text FROM {table} WHERE h3_text_embed IS NULL AND {text} IS NOT NULL "
            f"AND {shard_sql(key, worker, workers)} LIMIT {limit}")


def main(worker, workers, poll_seconds=120):
    """
    Command. Embeds this worker's texts of every table, one query per table per round, until a round finds none twice
    `poll_seconds` apart (captions still arrive).
    """
    from bluejay import open_database

    db = open_database("project.dtab")
    encoder = h3.load_text_encoder()

    def handler(table):
        key = TABLES[table][0]

        def handle(rows):
            db.commit(table, ["h3_text_embed"], [{key: row[key], "h3_text_embed": tensors_bytes(h3.embed(encoder, row["text"]), layer=50)}
                                                 for row in rows])
        return handle

    with torch.inference_mode():
        round_robin(db, [(table_sql(t, worker, workers), handler(t), f"embed worker {worker} {t}") for t in TABLES], poll_seconds)
    print(f"embed worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
