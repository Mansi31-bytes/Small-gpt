"""
Production training script.

Differences from smoke_train.py:
  - LR schedule (linear warmup + cosine decay)
  - Param-grouped optimizer (weight decay only on 2D weights)
  - Gradient clipping
  - Checkpoint save (best val + final)
  - JSON log of per-step metrics

For real GPU training we'll add mixed precision in a separate pass.
"""
import sys
sys.path.insert(0, ".")

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from src.model import GPTConfig, GPT
from src.data import load_tokens, get_batch


@dataclass
class TrainConfig:
    """All hyperparameters for one training run."""
    # Run identity
    run_name: str = "smoke_absolute"
    out_dir: str = "results"

    # Training duration
    num_steps: int = 50
    eval_every: int = 10
    eval_iters: int = 5          # batches to average for val loss

    # Batch
    batch_size: int = 4

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule
    warmup_steps: int = 10        # tiny for smoke test; for real runs use ~1000
    min_lr_ratio: float = 0.1     # cosine decays to 10% of peak

    # Misc
    seed: int = 0
    device: str = "cpu"


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup → cosine decay to min_lr_ratio × peak."""
    # Warmup phase
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps

    # After total steps, stay at the minimum
    if step >= cfg.num_steps:
        return cfg.learning_rate * cfg.min_lr_ratio

    # Cosine decay
    progress = (step - cfg.warmup_steps) / (cfg.num_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + (cfg.learning_rate - min_lr) * cosine


def configure_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """
    AdamW with two parameter groups:
        - 2D+ tensors (Linear/Embedding weights): weight_decay applied
        - 1D tensors (biases, LayerNorm gains): no weight_decay
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    print(f"Optimizer groups: "
          f"{sum(p.numel() for p in decay):,} decayed, "
          f"{sum(p.numel() for p in no_decay):,} not decayed")

    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
    )


@torch.no_grad()
def estimate_val_loss(model, val_data, cfg: TrainConfig) -> float:
    """Mean loss over cfg.eval_iters random val batches."""
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(val_data, cfg.batch_size,
                          model.config.context_len, cfg.device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, step, val_loss, cfg, path):
    """Save complete training state to a single .pt file."""
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "train_config": asdict(cfg),
        "val_loss": val_loss,
    }, path)


def train(cfg: TrainConfig, model_config: GPTConfig | None = None):
    if model_config is None:
        model_config = GPTConfig()
    torch.manual_seed(cfg.seed)

    # Output directory: results/<run_name>/
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Persist configs alongside the run for reproducibility
    (out_dir / "model_config.json").write_text(json.dumps(asdict(model_config), indent=2))
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    model = GPT(model_config).to(cfg.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params on {cfg.device}")

    optimizer = configure_optimizer(model, cfg)
    train_data = load_tokens("train")
    val_data   = load_tokens("val")

    log = []
    best_val = float("inf")
    print(f"\nTraining for {cfg.num_steps} steps\n")
    t_start = time.time()

    for step in range(cfg.num_steps):
        # 1. Set LR according to schedule (every step — needed for warmup + decay)
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 2. Train step
        x, y = get_batch(train_data, cfg.batch_size,
                          model_config.context_len, cfg.device)
        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )
        optimizer.step()

        # 3. Log every step (lightweight)
        entry = {
            "step": step + 1,
            "train_loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm.item(),
        }

        # 4. Periodic eval + checkpoint
        if (step + 1) % cfg.eval_every == 0:
            val_loss = estimate_val_loss(model, val_data, cfg)
            entry["val_loss"] = val_loss
            elapsed = time.time() - t_start
            print(f"step {step+1:5d}  |  train {loss.item():.4f}  "
                  f"|  val {val_loss:.4f}  |  lr {lr:.2e}  "
                  f"|  gn {grad_norm:.2f}  |  {elapsed:.0f}s")

            # Track best-val checkpoint
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step + 1, val_loss, cfg,
                                out_dir / "best.pt")

        log.append(entry)

    # Final checkpoint + full log
    save_checkpoint(model, optimizer, cfg.num_steps, best_val, cfg,
                    out_dir / "final.pt")
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))

    print(f"\nDone in {time.time() - t_start:.0f}s")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Artifacts in:  {out_dir}")


if __name__ == "__main__":
    train(TrainConfig())