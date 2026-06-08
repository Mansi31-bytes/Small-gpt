"""
Full model smoke test: construction, forward, loss, backward.

Key sanity check: at random init, cross-entropy loss should be near
log(vocab_size), because the model is producing essentially uniform
predictions over the vocabulary. Any significant deviation suggests
either an init bug or a math bug.
"""
import sys
sys.path.insert(0, ".")

import math
import torch
from src.model import GPTConfig, GPT
from src.data import load_tokens, get_batch

torch.manual_seed(0)
config = GPTConfig()
model = GPT(config)

# Param accounting
total = sum(p.numel() for p in model.parameters())
tok_emb_params = config.vocab_size * config.n_embd
pos_emb_params = config.context_len * config.n_embd
non_emb = total - tok_emb_params - pos_emb_params

print(f"Total parameters:        {total:,}")
print(f"  token embedding (tied with output head): {tok_emb_params:,}")
print(f"  position embedding:                      {pos_emb_params:,}")
print(f"  transformer blocks + final ln:           {non_emb:,}")
print(f"  non-embedding params: {non_emb / 1e6:.1f}M  (the 'effective model size')")

# Forward + loss
train_data = load_tokens("train")
x, y = get_batch(train_data, batch_size=4, context_len=128, device="cpu")
print(f"\nBatch: x {tuple(x.shape)}, y {tuple(y.shape)}")

logits, loss = model(x, targets=y)
print(f"Logits: {tuple(logits.shape)}")
print(f"Loss:   {loss.item():.4f}")

# At random init, loss should be near log(vocab_size)
expected = math.log(config.vocab_size)
print(f"\nLoss sanity check:")
print(f"  Expected (uniform random over vocab): {expected:.4f}")
print(f"  Difference from observed:             {abs(loss.item() - expected):.4f}")
assert abs(loss.item() - expected) < 1.0, "Loss is too far from log(vocab_size) — check init"
print(f"  ✓ Loss is within 1.0 of log(vocab_size)")

# Backward pass — verify every parameter receives a gradient
loss.backward()
total_p = sum(1 for _ in model.parameters())
with_grad = sum(1 for p in model.parameters() if p.grad is not None)
print(f"\nBackward pass: {with_grad}/{total_p} parameters received gradients")
assert with_grad == total_p, "Some parameters got no gradient — disconnected from loss"
print(f"  ✓ All parameters connected to loss")