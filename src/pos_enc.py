"""
Positional encoding utilities.

Absolute: a learned [context_len, n_embd] embedding added to tokens at input.
RoPE:     rotates Q and K vectors inside attention by position-dependent angles.
          No parameters — purely a function of position and head_dim.
"""
import torch
from torch import Tensor


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute cos/sin tables for RoPE.

    Returns:
        cos, sin: each (max_seq_len, head_dim)

    The tables are built for head_dim/2 frequency pairs and then repeated
    along the last dim — this supports the rotate_half trick, which processes
    dimension pairs (i, i + head_dim/2) rather than adjacent pairs (2i, 2i+1).
    Mathematically equivalent; this layout is computationally convenient.
    """
    assert head_dim % 2 == 0
    # Frequencies: θᵢ = 1 / 10000^(2i / head_dim) for i in [0, head_dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Outer product with position indices → (max_seq_len, head_dim/2)
    positions = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(positions, freqs)
    # Repeat to full head_dim
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (max_seq_len, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """
    Rotate the last dimension by swapping and negating halves.

    For x = [a, b] (split at midpoint):
        rotate_half(x) = [-b, a]

    Combined with x * cos + rotate_half(x) * sin, this implements
    the 2D rotation: a' = a·cos - b·sin, b' = b·cos + a·sin.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: Tensor, k: Tensor,
               cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """
    Rotate query and key tensors by their positions.

    Args:
        q, k:   (B, n_head, T, head_dim)
        cos, sin: (max_seq_len, head_dim) — we slice to [:T]

    Returns:
        rotated q and k, same shape
    """
    T = q.shape[2]
    # Slice to actual sequence length and add batch/head dims for broadcasting
    c = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s