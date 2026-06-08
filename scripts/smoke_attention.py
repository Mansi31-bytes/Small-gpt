"""
Smoke-test attention.

Two things to verify:
  1. Output shape matches input shape — trivial but catches reshape bugs.
  2. Causal mask actually prevents future leakage — THE most important
     property of causal attention. If this is broken, training will
     mysteriously work but the model will be useless at generation.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.model import GPTConfig, Embeddings, CausalSelfAttention

torch.manual_seed(0)

config = GPTConfig()
emb = Embeddings(config)
attn = CausalSelfAttention(config)

# Param accounting
n_attn_params = sum(p.numel() for p in attn.parameters())
qkv_params = 3 * config.n_embd * config.n_embd
proj_params = config.n_embd * config.n_embd
print(f"Attention params: {n_attn_params:,}")
print(f"  qkv proj: {qkv_params:,}")
print(f"  out proj: {proj_params:,}")

# Forward shape check
x = torch.randint(0, config.vocab_size, (2, 64))
h = emb(x)
print(f"\nInput to attention:  {tuple(h.shape)}")
out = attn(h)
print(f"Output of attention: {tuple(out.shape)}")
print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}")

# Causal mask verification: change ONLY the last token, then check that
# outputs at earlier positions are unchanged.
print("\n--- Causal mask verification ---")
emb.eval(); attn.eval()   # disable dropout so outputs are deterministic
with torch.no_grad():
    x_a = torch.randint(0, config.vocab_size, (1, 8))
    x_b = x_a.clone()
    # Change the last token to a different one
    x_b[0, -1] = (x_a[0, -1] + 1) % config.vocab_size

    h_a, h_b = emb(x_a), emb(x_b)
    out_a, out_b = attn(h_a), attn(h_b)

    # Per-position max difference across the embedding dimension
    diff = (out_a - out_b).abs().max(dim=-1).values.squeeze()
    print("Per-position max difference after changing only token[7]:")
    for i, d in enumerate(diff):
        marker = " ← changed token" if i == 7 else ""
        flag = "PASS" if (d < 1e-6 if i < 7 else d > 1e-3) else "FAIL"
        print(f"  pos {i}: {d.item():.2e}  [{flag}]{marker}")