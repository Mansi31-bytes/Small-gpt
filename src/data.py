"""
Dataloader for tokenized TinyStories binary files.

The dataloader is intentionally tiny — it's just a function that draws
random windows of consecutive tokens from a memmap'd binary file and
returns them as (input, target) tensors for next-token prediction.

Why so simple: with pre-tokenized data on disk, there's no preprocessing
to parallelize, no transforms to compose, and no need for PyTorch's
Dataset/DataLoader machinery. A function call beats a class hierarchy here.
"""
from pathlib import Path
import numpy as np
import torch

DATA_DIR = Path("data/tokenized")


def load_tokens(split: str) -> np.memmap:
    """
    Open a tokenized .bin file as a memory-mapped numpy array.

    The 'r' mode is read-only; the OS pages chunks of the file into RAM on
    demand as we slice into them, so we never have to hold the full 1 GB
    train file in memory.
    """
    assert split in {"train", "val"}, f"unknown split: {split!r}"
    return np.memmap(DATA_DIR / f"{split}.bin", dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    batch_size: int,
    context_len: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `batch_size` random windows of length `context_len + 1` from
    `data`, then split each window into:

        x = window[:-1]   shape (context_len,)   # input tokens
        y = window[1:]    shape (context_len,)   # target tokens

    For each position i in x, the model is trained to predict y[i] from
    x[:i+1]. So every single token position contributes a training signal,
    which is why language model training is so token-efficient.

    Returns:
        x: (batch_size, context_len) int64 tensor of input token IDs
        y: (batch_size, context_len) int64 tensor of target token IDs
    """
    # Valid start positions: we need context_len + 1 tokens after start.
    max_start = len(data) - context_len - 1
    starts = torch.randint(low=0, high=max_start, size=(batch_size,))

    # Slice batch_size windows from the memmap. The .astype(np.int64)
    # cast is needed because nn.Embedding requires int64 indices; we
    # stored uint16 just to halve the disk footprint.
    x = torch.stack([
        torch.from_numpy(data[s : s + context_len].astype(np.int64))
        for s in starts
    ])
    y = torch.stack([
        torch.from_numpy(data[s + 1 : s + 1 + context_len].astype(np.int64))
        for s in starts
    ])

    # Device transfer. pin_memory + non_blocking lets the H2D copy overlap
    # with whatever the GPU is currently doing — meaningful speedup on
    # GPU, no-op on CPU.
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y