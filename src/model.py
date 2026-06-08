"""
Decoder-only transformer (GPT-style) implemented from scratch.

This file grows as we add components. Today: config + embeddings.
Coming next: attention, MLP, transformer block, full GPT, generation.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    """Configuration for the GPT model.

    Defaults give a ~10M non-embedding-param model (~30M total) suitable
    for training on TinyStories in a few hours on a single T4 GPU.
    """
    vocab_size: int = 50257           # GPT-2 BPE vocab (fixed by tiktoken)
    context_len: int = 256            # max sequence length
    n_layer: int = 6                  # number of transformer blocks
    n_head: int = 6                   # attention heads per block
    n_embd: int = 384                 # residual stream / embedding dim
    dropout: float = 0.1
    bias: bool = False                # bias in Linear and LayerNorm
    pos_encoding: str = "absolute"    # "absolute" or "rope" (added later)


class Embeddings(nn.Module):
    """
    Token embeddings + optional positional embeddings.

    Absolute PE: adds a learned position vector to each token embedding at input.
    RoPE:        no positional signal here — position is encoded inside attention
                 by rotating Q and K, so the input is just the token embedding.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb      = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_encoding = config.pos_encoding
        self.context_len  = config.context_len

        if config.pos_encoding == "absolute":
            self.pos_emb = nn.Embedding(config.context_len, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.context_len

        tok = self.tok_emb(idx)                          # (B, T, n_embd)

        if self.pos_encoding == "absolute":
            pos = torch.arange(T, device=idx.device)
            tok = tok + self.pos_emb(pos)               # add position signal

        # RoPE: nothing to add here — position is handled inside attention

        return self.dropout(tok)
    

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Supports both absolute PE (no changes here) and RoPE (rotates Q/K before
    computing attention scores). Everything else is identical between the two.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv          = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj         = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.context_len, config.context_len))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.context_len, config.context_len),
        )

        # RoPE: precompute cos/sin tables once; move to GPU with the model.
        # Not needed for absolute PE — the positional signal comes from the
        # input embeddings instead.
        if config.pos_encoding == "rope":
            from src.pos_enc import precompute_rope
            cos, sin = precompute_rope(self.head_dim, config.context_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE: rotate Q and K by their positions before computing scores.
        # This is the only line that differs between absolute-PE and RoPE attention.
        if hasattr(self, 'rope_cos'):
            from src.pos_enc import apply_rope
            q, k = apply_rope(q, k, self.rope_cos, self.rope_sin)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))
    
class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently at each position — no mixing across positions.
    Standard 4x expansion: n_embd → 4*n_embd → n_embd, with GELU between.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    One transformer block: pre-norm residual attention + pre-norm residual MLP.

    The "residual stream" is the tensor `x` flowing through the block. Each
    sublayer reads from a normalized copy and adds its output back to the
    unnormalized stream:

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Pre-norm (LayerNorm before the sublayer) keeps the residual path
    unnormalized, which is why pre-norm transformers train stably without
    careful warmup. Modern LMs (GPT-2 onwards, Llama, etc.) all use pre-norm.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Two LayerNorms — one before attention, one before MLP.
        # bias=False is a small convention from modern LMs; LayerNorm with
        # only the gain parameter (no bias) seems to work as well in practice.
        self.ln1  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    """
    Full decoder-only transformer.

    Architecture:
        idx → Embeddings → [Block × n_layer] → LayerNorm → output_head → logits

    The output head shares weights with the token embedding ("weight tying"),
    halving the parameter count of the largest matrix in the model.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: head.weight and tok_emb.weight are now the SAME tensor.
        # Updates to one propagate to the other (it's literally one parameter
        # that we're using in two places).
        self.head.weight = self.embeddings.tok_emb.weight

        # GPT-2 init scheme
        self.apply(self._init_weights)
        # Special init for residual-stream-writing projections: scale down so
        # the residual stream variance doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                std = 0.02 / (2 * config.n_layer) ** 0.5
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style init: narrow Gaussian on Linears and Embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) int64 token IDs
            targets: (B, T) int64 next-token targets (optional)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar tensor if targets given, else None
        """
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        # Cross-entropy over the vocabulary. Flatten (B, T, V) → (B*T, V)
        # and (B, T) → (B*T,) so F.cross_entropy treats every position as
        # an independent classification problem.
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample new tokens.

        Args:
            idx:            (B, T) starting tokens
            max_new_tokens: how many tokens to append
            temperature:    1.0 = sample from raw distribution
                            <1 = sharper (more deterministic)
                            >1 = flatter (more random)
            top_k:          if set, sample only from top-k most likely tokens
                            (truncates the long tail of low-prob tokens)

        Returns:
            (B, T + max_new_tokens) extended sequence
        """
        self.eval()
        for _ in range(max_new_tokens):
            # If context is longer than what model was trained on, crop
            ctx = self.config.context_len
            idx_cond = idx if idx.size(1) <= ctx else idx[:, -ctx:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)

        return idx