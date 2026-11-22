"""
H3 text embeddings of the H3 Magic project (manifest §22): long-lived text-encoder workers (Qwen3-VL-32B, ~67 GB),
one per GPU, committed with the database's commit. ~0.1 s per caption on an A100.

Each blob is safetensors {"prompt_embeds": (1, N, 5120) bfloat16, "text_token_tags": (N,) int64}: hidden state 50 of the
t2va text encoder, exactly what H3's denoiser takes (h3magic/h3.py embed). h3_text_sha256 = sha256 of the text it embeds:
a reader must use h3_text_embed only when h3_text_sha256 = encode(sha256(text), 'hex') (m00939: recaptions). Tables:
    h3_pexels_clips   (h3_caption)     h3_magick_images  (h3_caption)     h3_eval_prompts  (prompt)

    h3_embed.py WORKER WORKERS [--poll_seconds 120]   (from data/; GPU = $CUDA_VISIBLE_DEVICES)
"""

import hashlib
import os
import sys

import fire
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from h3_blobs import round_robin, shard_sql, tensors_bytes
from h3_encode import train_first
from h3magic import h3

TABLES = {  # table: (key, text column, rows per query)
    "h3_eval_prompts": ("prompt_id", "prompt", 64),
    "h3_pexels_clips": ("pexels_id", "h3_caption", 256),
    "h3_magick_images": ("page_id", "h3_caption", 256),
}


def text_sha256(text):
    """
    Pure function. Hex sha256 of the UTF-8 text: the key that ties an embedding to the exact text it embeds (the SQL
    side computes the same value with encode(sha256(text), 'hex')).

    Examples:
        >>> text_sha256("abc")[:16]
        'ba7816bf8f01cfea'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table_sql(table, worker, workers):
    """
    Pure function. The query of one table's texts whose embedding is missing or stale (h3_text_sha256 is not the
    sha256 of the current text, e.g. after a recaption, m00939), of this worker.

    Examples:
        >>> sql = table_sql("h3_magick_images", 0, 1)
        >>> sql.startswith("SELECT page_id, h3_caption AS text FROM h3_magick_images") and "h3_text_sha256 <> encode(sha256(h3_caption), 'hex')" in sql
        True
    """
    key, text, limit = TABLES[table]
    return (f"SELECT {key}, {text} AS text FROM {table} WHERE {text} IS NOT NULL "
            f"AND (h3_text_embed IS NULL OR h3_text_sha256 IS NULL OR h3_text_sha256 <> encode(sha256({text}), 'hex')) "
            f"AND {shard_sql(key, worker, workers)} {'' if table == 'h3_eval_prompts' else train_first(table)} LIMIT {limit}")


def main(worker, workers, poll_seconds=120, idle_encode_seconds=0, shard_lo=None, shard_hi=None):
    """
    Command. Embeds this worker's texts of every table, one query per table per round, until a round finds none twice
    `poll_seconds` apart (captions still arrive). With idle_encode_seconds > 0, a round with no text moves the text
    encoder to the CPU and runs h3_encode's video job on this GPU for that long instead of waiting (no GPU idles while
    captions trickle in); the worker then runs until killed. shard_lo, shard_hi: the idle video encode takes only byte
    shards shard_lo..shard_hi (h3_encode 256-shard mode), so it never overlaps another node's encode workers.
    """
    import time
    from bluejay import open_database

    db = open_database("project.dtab")
    encoder = h3.load_text_encoder()
    vae, pool = None, None

    def handler(table):
        key = TABLES[table][0]

        def handle(rows):
            db.commit(table, ["h3_text_embed", "h3_text_sha256"],
                      [{key: row[key], "h3_text_embed": tensors_bytes(h3.embed(encoder, row["text"]), layer=50, text_sha256=text_sha256(row["text"])),
                        "h3_text_sha256": text_sha256(row["text"])} for row in rows])
        return handle

    tasks = [(table_sql(t, worker, workers), handler(t), f"embed worker {worker} {t}") for t in TABLES]
    with torch.inference_mode():
        if idle_encode_seconds <= 0:
            round_robin(db, tasks, poll_seconds)
        while idle_encode_seconds > 0:
            round_robin(db, tasks, 0)  # Embeds until a round finds no text
            import h3_encode
            from concurrent.futures import ThreadPoolExecutor
            encoder.to("cpu")
            torch.cuda.empty_cache()
            if vae is None:
                vae, pool = h3.load_vae(), ThreadPoolExecutor(h3_encode.IO_WORKERS)
            vae.to("cuda")
            end, sql = time.time() + idle_encode_seconds, h3_encode.job_sql("video", 0, 1, None if shard_lo is None else list(range(shard_lo, shard_hi + 1)))
            while time.time() < end:
                rows = db.query(sql).to_pylist()
                if not rows:
                    time.sleep(10)
                    continue
                blobs = h3_encode.encode_rows(vae, pool, "video", rows)
                db.commit("h3_pexels_clips", ["h3_latent_video"], [{"pexels_id": r["pexels_id"], "h3_latent_video": b} for r, b in zip(rows, blobs)])
                print(f"{time.strftime('%I:%M:%S %p')} embed worker {worker} idle: +{len(rows)} video latents", flush=True)
            vae.to("cpu")
            torch.cuda.empty_cache()
            encoder.to("cuda")
    print(f"embed worker {worker}: done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
