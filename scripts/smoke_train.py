"""
CPU training smoke test — verify the training loop reduces loss.

Goal: watch loss drop from log(vocab_size) ≈ 10.83 to something noticeably
lower (~7-8 range) over a few dozen steps. We don't need to train to
convergence here, just confirm the loop functions end-to-end.

This is the bare-minimum training loop. The production version (next
session, on Colab GPU) adds LR scheduling, gradient clipping, mixed
precision, and checkpointing.
"""
import sys
sys.path.insert(0, ".")

import math
import time
import torch
from src.model import GPTConfig, GPT
from src.data import load_tokens, get_batch

torch.manual_seed(0)

# Hyperparameters — small for CPU
BATCH_SIZE   = 4
CONTEXT_LEN  = 128
NUM_STEPS    = 50
EVAL_EVERY   = 10
EVAL_ITERS   = 5
LEARNING_RATE = 3e-4

device = "cpu"
print(f"Device: {device}")

config = GPTConfig()
model = GPT(config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# AdamW: Adam + decoupled weight decay. Standard for transformer training.
# Betas (0.9, 0.95) from GPT-3 — the second beta is lower than Adam's
# default (0.999) because LM loss landscapes need faster adaptation
# of the variance estimate.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

train_data = load_tokens("train")
val_data = load_tokens("val")


@torch.no_grad()
def estimate_val_loss(n_iters: int) -> float:
    """Compute mean loss over n_iters random val batches."""
    model.eval()
    losses = []
    for _ in range(n_iters):
        x, y = get_batch(val_data, BATCH_SIZE, CONTEXT_LEN, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# Initial loss should be near log(vocab_size)
initial_val = estimate_val_loss(EVAL_ITERS)
print(f"\nInitial val loss: {initial_val:.4f}")
print(f"  expected at random init: {math.log(config.vocab_size):.4f}\n")

print(f"Training for {NUM_STEPS} steps "
      f"(rough estimate: 10-15 min on CPU; interrupt with Ctrl+C if too slow)\n")
t_start = time.time()

for step in range(NUM_STEPS):
    x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LEN, device)

    # The four lines that are 95% of all PyTorch training:
    _, loss = model(x, targets=y)              # forward + loss
    optimizer.zero_grad(set_to_none=True)       # clear old gradients
    loss.backward()                             # backward pass
    optimizer.step()                            # apply gradient update

    if (step + 1) % EVAL_EVERY == 0:
        val = estimate_val_loss(EVAL_ITERS)
        elapsed = time.time() - t_start
        sec_per_step = elapsed / (step + 1)
        print(f"step {step+1:3d}  |  train {loss.item():.4f}  |  "
              f"val {val:.4f}  |  {sec_per_step:.1f}s/step  |  "
              f"{elapsed:.0f}s total")

total_time = time.time() - t_start
final_val = estimate_val_loss(EVAL_ITERS)
print(f"\nTotal training time: {total_time:.1f}s")
print(f"Final val loss: {final_val:.4f}  "
      f"(dropped {initial_val - final_val:.2f} from initial)")