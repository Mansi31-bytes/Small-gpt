"""Smoke-test the dataloader by drawing one batch and inspecting it."""
import sys
sys.path.insert(0, ".")  # so 'from src.data import ...' works

from src.data import load_tokens, get_batch
import tiktoken

enc = tiktoken.get_encoding("gpt2")

train_data = load_tokens("train")
val_data = load_tokens("val")
print(f"Train: {len(train_data):,} tokens")
print(f"Val:   {len(val_data):,} tokens")

# Draw one batch
x, y = get_batch(train_data, batch_size=4, context_len=128, device="cpu")
print(f"\nBatch shapes:")
print(f"  x: {tuple(x.shape)}, dtype {x.dtype}")
print(f"  y: {tuple(y.shape)}, dtype {y.dtype}")

# Sanity check: y must equal x shifted by 1 (since they come from
# overlapping slices of the same underlying token stream)
assert (x[:, 1:] == y[:, :-1]).all(), "y is not x shifted by 1 — bug!"
print("\n✓ y == x shifted by 1 (next-token target verified)")

# Decode one sample to confirm it looks like text
print("\nFirst sample of x decoded:")
print(repr(enc.decode(x[0].tolist())))