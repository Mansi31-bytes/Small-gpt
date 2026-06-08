"""Smoke-test the embedding layer on a real batch."""
import sys
sys.path.insert(0, ".")

import torch
from src.data import load_tokens, get_batch
from src.model import GPTConfig, Embeddings

config = GPTConfig()
print(f"Config: {config}")

emb = Embeddings(config)
n_params = sum(p.numel() for p in emb.parameters())
tok_params = config.vocab_size * config.n_embd
pos_params = config.context_len * config.n_embd
print(f"\nEmbedding params total: {n_params:,}")
print(f"  token emb: {tok_params:,}  ({100*tok_params/n_params:.1f}%)")
print(f"  pos   emb: {pos_params:,}  ({100*pos_params/n_params:.1f}%)")

# Embed a real batch
train_data = load_tokens("train")
x, _ = get_batch(train_data, batch_size=4, context_len=128, device="cpu")
print(f"\nInput x: shape {tuple(x.shape)}, dtype {x.dtype}")

out = emb(x)
print(f"Output:  shape {tuple(out.shape)}, dtype {out.dtype}")
print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}")