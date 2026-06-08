# Results

## Overview

Two decoder-only transformer models were trained from scratch on TinyStories V2,
differing only in their positional encoding scheme. Everything else — architecture,
optimizer, data, hardware, number of steps — was held constant.

**Key findings:**
- RoPE achieves **5.2% lower perplexity** at the training context length
- Beyond the training context, RoPE degrades **2.5× less** than absolute PE
- Both findings are visible in a single controlled experiment with no confounders

---

## Training Setup

| | |
|---|---|
| **Architecture** | Decoder-only transformer (GPT-style) |
| **Parameters** | 30M total · 10.6M non-embedding |
| **Depth / Width** | 6 layers · 6 heads · d_model = 384 |
| **Context length** | 256 tokens |
| **Dataset** | TinyStories V2 GPT-4 · 2.7M stories · 541M tokens |
| **Tokenizer** | GPT-2 BPE · 50,257 vocab |
| **Optimizer** | AdamW · lr = 3e-4 · β = (0.9, 0.95) · weight decay = 0.1 |
| **LR schedule** | Linear warmup 500 steps → cosine decay to 10% of peak |
| **Batch** | 32 sequences × 256 tokens = 8,192 tokens/step |
| **Steps** | 10,000 |
| **Hardware** | NVIDIA T4 16 GB · fp16 mixed precision |
| **Training time** | ~44 minutes per run |

---

## Finding 1 — RoPE converges faster and achieves lower val loss

| | Absolute PE | RoPE | Difference |
|---|---|---|---|
| Best val loss | 1.6792 | **1.6257** | −0.054 |
| Best perplexity | 5.36 | **5.08** | −5.2% |
| Parameters | 30,018,816 | 29,920,512 | −98,304 |
| Best checkpoint | step 10,000 | step 9,000 | — |
| Training time | 43.7 min | 43.1 min | — |

RoPE achieves lower perplexity with *fewer* parameters, since it has no position
embedding table. It also converges faster: at step 500 the val loss gap was already
0.30 (3.07 vs 3.37), narrowing to ~0.05 by step 5,000 and stabilizing there.

One honest caveat: the absolute PE model was still improving at step 10,000 and
had not fully converged. With a longer training run it would likely narrow the gap.
The extrapolation experiment below is less sensitive to this.

---

## Finding 2 — RoPE degrades far less beyond the training context

Both models were trained at context length 256. At evaluation time, we extended
the context window up to 512 tokens to test out-of-distribution (OOD) generalization.

**What happens at OOD positions:**
- Absolute PE has a learned embedding for each of positions 0–255. Positions
  256–511 received randomly-initialized embeddings during inference — the model
  was never trained to use them, so the positional signal is noise.
- RoPE applies the same sin/cos frequency functions at every position. Position
  300 uses the same rotation formula as position 44; the model hasn't been tuned
  for those specific values, but the pattern is systematic rather than random.

| Context | Abs PE loss | RoPE loss | Gap (Abs − RoPE) |
|---|---|---|---|
| 64 | 1.942 | 1.885 | +0.056 |
| 128 | 1.795 | 1.738 | +0.057 |
| 192 | 1.724 | 1.667 | +0.057 |
| **256 — training context** | **1.671** | **1.594** | **+0.077** |
| 320 ← OOD | 1.916 | 1.629 | +0.287 |
| 384 ← OOD | 2.160 | 1.756 | +0.404 |
| 448 ← OOD | 2.425 | 1.905 | +0.521 |
| 512 ← OOD | 2.636 | 2.096 | +0.540 |

Within the training distribution (ctx ≤ 256), the gap is stable at ~0.057 —
consistent with the training comparison. The moment context exceeds 256, the gap
grows sharply and monotonically.

At **2× the training context (512 tokens)**:

| | At ctx 256 | At ctx 512 | Increase |
|---|---|---|---|
| Absolute PE | perplexity 5.31 | perplexity 13.94 | **+163%** |
| RoPE | perplexity 4.90 | perplexity 8.10 | **+65%** |

Absolute PE perplexity nearly **triples** when moving to 2× context.
RoPE increases by **less than double**. RoPE degrades roughly 2.5× more gracefully.

---

## Plots

![Training curves, val loss comparison, and length extrapolation](results/comparison_plots.png)

*Left: training curves for both models over 10,000 steps. Centre: best val loss
comparison. Right: val loss as a function of context length — the dashed line marks
the training context (256); everything to its right is out-of-distribution.*

---

## Generated samples (temperature 0.8, top-k 50)

**Prompt:** *"Once upon a time, there was a little girl named Lily."*

**Absolute PE (val loss 1.68):**
> Once upon a time, there was a little girl named Lily. She had a big, modern box
> with many toys. One day, Lily was playing in her garden when she saw something
> strange. It was a big, green sock. [...] From that day on, Lily and Tim played
> together every day. They learned that sharing and being kind made them happy.

**RoPE (val loss 1.63):**
> Once upon a time, there was a little girl named Lily. She had a big, modern box
> with her favorite things inside. One day, Lily went to play outside with her
> friend, Tom. As they played, Lily accidentally broke her box. [...] Lily and Tom
> were happy. They learned that helping others made everyone happy.

Both models produce coherent story structure, dialogue, characters, and a moral
conclusion. The 0.05 val loss difference does not dramatically change qualitative
generation quality at this scale — the quantitative improvement is real but subtle
in text output.