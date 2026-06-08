"""Verify the tokenized binaries are readable and round-trip correctly."""
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# memmap reads directly from disk without loading the whole file into RAM.
# This is exactly how the dataloader will access these files during training,
# so verifying it works now is a useful proxy for training-time correctness.
val_tokens = np.memmap("data/tokenized/val.bin", dtype=np.uint16, mode="r")
train_tokens = np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")

print(f"Val tokens:   {len(val_tokens):,}")
print(f"Train tokens: {len(train_tokens):,}")

# First 50 raw token IDs
print(f"\nFirst 50 val token IDs:\n  {val_tokens[:50].tolist()}")

# Decode them — should look like the start of a story
print("\nDecoded first 50 val tokens:")
print(repr(enc.decode(val_tokens[:50].tolist())))

# Find story boundaries (EOT = 50256) in first 1000 tokens
eots = np.where(val_tokens[:1000] == 50256)[0]
print(f"\nEOT (50256) positions in first 1000 val tokens:\n  {eots.tolist()}")
print(f"  → {len(eots)} story boundaries in that window")
print(f"  → mean story length in tokens (rough): "
      f"{1000 / max(len(eots), 1):.0f}")