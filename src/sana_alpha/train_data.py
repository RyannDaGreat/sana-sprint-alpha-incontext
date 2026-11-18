"""
Training records for the adapter: raw (unscaled) DC-AE latents of MAGICK gray composites + captions.

Every source is called as source(rank, world_size, seed) and returns THIS RANK'S SHARD (each rank loads only its
own latents and encodes only its own captions):
    dict(latents=(N, 32, 32, 32) raw DC-AE latents at 1024² (multiply by vae scaling_factor 0.41407),
         captions=list[str] of N MAGICK `subject` strings, page_ids=list[str])
Sources: `lance` (data/lance/magick_samples.lance, written by the BlueJay curation) and `toy`
(agents/.scratchpad/toy_train: latents.safetensors + rows.tsv, for tests).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import pandas as pd
import safetensors.torch
import torch
from safetensors.torch import load_file

BLOB_READ_THREADS = 16


def select_train_rows(rows):
    """
    Pure function. Keeps training rows: split == 'train' and not nsfw. Keeps the original order and index.

    Args:
        rows (pd.DataFrame): columns include split, nsfw

    Returns:
        pd.DataFrame

    Examples:
        >>> df = pd.DataFrame(dict(page_id=["a", "b", "c"], split=["train", "eval", "train"], nsfw=[0, 0, 1]))
        >>> select_train_rows(df).page_id.tolist()
        ['a']
    """
    return rows[(rows.split == "train") & (rows.nsfw == 0)]


def load_toy_records(folder, rank, world_size, seed):
    """
    Query. Reads this rank's shard of the toy set written by agents/.scratchpad/scratchpad_agent_3.py.

    Args:
        folder (str | Path): holds latents.safetensors (latent_rgb_1024: (N, 32, 32, 32)) and rows.tsv
        rank (int): this process's rank
        world_size (int): number of processes
        seed (int): shard permutation seed

    Returns:
        dict: latents (M, 32, 32, 32) float32, captions list[str], page_ids list[str]
    """
    folder = Path(folder)
    rows = pd.read_csv(folder / "rows.tsv", sep="\t", quoting=3)
    latents = load_file(folder / "latents.safetensors")["latent_rgb_1024"]
    assert len(rows) == len(latents), f"{len(rows)} rows but {len(latents)} latents"
    keep = select_train_rows(rows)
    keep = keep.iloc[rank_shard(len(keep), rank, world_size, seed).numpy()]
    return dict(latents=latents[torch.tensor(keep.index.to_numpy())], captions=keep.subject.astype(str).tolist(),
                page_ids=keep.page_id.tolist())


def load_lance_records(path, rank, world_size, seed, column="latent_rgb_1024"):
    """
    Query. Reads this rank's shard of training rows (split 'train', not nsfw, latent present) from magick_samples.

    Rows are sorted by page_id before sharding, so the shard depends only on which rows exist, not on table order.
    Row ids can change after compaction, so they are read fresh on every call.

    Args:
        path (str | Path): magick_samples.lance folder
        rank (int): this process's rank
        world_size (int): number of processes
        seed (int): shard permutation seed
        column (str): latent column: latent_rgb_1024 (on gray) or latent_rgb_colorbg_1024 (on a random color)

    Returns:
        dict: latents (M, 32, 32, 32) bfloat16, captions list[str], page_ids list[str]
    """
    ds = lance.dataset(str(path))
    table = ds.to_table(columns=["page_id", "subject"], with_row_id=True,
                        filter=f"{column} IS NOT NULL AND split = 'train' AND nsfw = false").to_pandas()
    table = table.sort_values("page_id").reset_index(drop=True)
    table = table.iloc[rank_shard(len(table), rank, world_size, seed).numpy()]
    return dict(latents=read_latent_blobs(ds, column, table["_rowid"].tolist()), captions=table.subject.astype(str).tolist(),
                page_ids=table.page_id.tolist())


def read_latent_blobs(ds, column, ids):
    """
    Query. Reads safetensors blobs {"latent": (32, 32, 32)} of a lance blob column, in the order of ids.

    Args:
        ds (lance.LanceDataset): e.g. magick_samples or trimask_augmentations
        column (str): blob column, e.g. latent_rgb_1024
        ids (list[int]): row ids (_rowid)

    Returns:
        torch.Tensor: (len(ids), 32, 32, 32) as stored (raw, unscaled bf16)
    """
    with ThreadPoolExecutor(BLOB_READ_THREADS) as pool:
        return torch.stack(list(pool.map(lambda b: safetensors.torch.load(b.read())["latent"], ds.take_blobs(column, ids=ids))))


def rank_shard(count, rank, world_size, seed):
    """
    Pure function. A fixed random split of record indices across ranks (the same every epoch).

    Each rank only needs caption embeddings for its own shard, so it encodes ~1/world of the captions.

    Args:
        count (int): N records
        rank (int): this process's rank
        world_size (int): number of processes
        seed (int): permutation seed

    Returns:
        torch.Tensor: (≈N / world_size,) int64 indices

    Examples:
        >>> a, b = rank_shard(10, 0, 2, 0), rank_shard(10, 1, 2, 0)
        >>> len(a), len(b), sorted(a.tolist() + b.tolist()) == list(range(10))
        (5, 5, True)
    """
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))[rank::world_size]


def epoch_batches(count, batch_size, epoch, seed):
    """
    Pure function. Shuffled batches of positions 0..count-1 for one epoch; the ragged last batch is dropped.

    Args:
        count (int): records in this rank's shard
        batch_size (int): per-rank batch size
        epoch (int): epoch number (changes the order)
        seed (int): base seed

    Returns:
        list[torch.Tensor]: count // batch_size tensors of shape (batch_size,)

    Examples:
        >>> b = epoch_batches(10, 4, 0, 0)
        >>> len(b), tuple(b[0].shape), len(set(torch.cat(b).tolist()))
        (2, (4,), 8)
    """
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed * 100003 + epoch))
    return list(order[: count // batch_size * batch_size].split(batch_size))
