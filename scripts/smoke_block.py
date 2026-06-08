"""
Smoke-test the full transformer block.

Three checks:
  1. Output shape matches input shape (residuals preserve shape).
  2. Causal property holds through the whole block, not just attention.
  3. Parameter count matches our calculation.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.model import GPTConfig, Embeddings, Block

torch.manual_seed(0)

config = GPTConfig()
emb = Embeddings(config)
block = Block(config)

# Param accounting
n_block_params = sum(p.numel() for p in block.parameters())
attn_params = 4 * config.n_embd ** 2                  # qkv (3C²) + out proj (C²)
mlp_params  = 2 * config.n_embd * 4 * config.n_embd   # fc (C × 4C) + proj (4C × C)
ln_params   = 2 * config.n_embd                       # 2 LayerNorms × n_embd gain
expected = attn_params + mlp_params + ln_params
print(f"Block params: {n_block_params:,}")
print(f"  expected:    {expected:,}")
print(f"    attn:      {attn_params:,}")
print(f"    mlp:       {mlp_params:,}")
print(f"    layernorm: {ln_params:,}")

# Forward shape check
x = torch.randint(0, config.vocab_size, (2, 64))
h = emb(x)
print(f"\nBlock input:  {tuple(h.shape)}")
out = block(h)
print(f"Block output: {tuple(out.shape)}")
print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}")

# End-to-end causal check through the full block
print("\n--- Causal verification through full block ---")
emb.eval(); block.eval()
with torch.no_grad():
    x_a = torch.randint(0, config.vocab_size, (1, 8))
    x_b = x_a.clone()
    x_b[0, -1] = (x_a[0, -1] + 1) % config.vocab_size
    out_a = block(emb(x_a))
    out_b = block(emb(x_b))
    diff = (out_a - out_b).abs().max(dim=-1).values.squeeze()
    for i, d in enumerate(diff):
        flag = "PASS" if (d < 1e-6 if i < 7 else d > 1e-3) else "FAIL"
        marker = " ← changed" if i == 7 else ""
        print(f"  pos {i}: {d.item():.2e}  [{flag}]{marker}")