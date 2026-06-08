"""
Tokenize TinyStories raw text into uint16 binary files for fast training.

Reads:
    data/raw/TinyStoriesV2-GPT4-{train,valid}.txt

Writes:
    data/tokenized/train.bin
    data/tokenized/val.bin

Each story is tokenized with GPT-2 BPE, then followed by the <|endoftext|>
token (ID 50256) so the model learns story boundaries during training.
Tokens are stored as uint16 since GPT-2 vocab (50,257) fits in 16 bits.
"""
from pathlib import Path
import numpy as np
import tiktoken
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/tokenized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GPT-2 tokenizer — same one OpenAI used for GPT-2 / GPT-3
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token   # 50256 — the <|endoftext|> special token

CHUNK_SIZE = 1000     # stories per batched encode call


def tokenize_file(in_path: Path, out_path: Path) -> None:
    print(f"\n=== {in_path.name} ===")
    print("Reading file into memory...")
    text = in_path.read_text(encoding="utf-8")

    # The raw file uses the literal string '<|endoftext|>' as a separator
    # between stories. We split on it to get individual stories, then later
    # we'll re-insert the EOT *token* (id 50256) between them.
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    print(f"  {len(stories):,} stories, {len(text)/1e6:.1f} MB raw text")

    # Free the giant text string before we start tokenizing
    del text

    # Tokenize in batches, write to disk incrementally
    print("  Tokenizing + writing to disk...")
    total_tokens = 0
    with open(out_path, "wb") as f:
        for i in tqdm(range(0, len(stories), CHUNK_SIZE)):
            chunk = stories[i : i + CHUNK_SIZE]

            # encode_ordinary_batch is faster than a Python loop because
            # tiktoken releases the GIL and uses multiple threads.
            # "ordinary" means: treat all input as literal text, don't try
            # to interpret special tokens like <|endoftext|>.
            encoded = enc.encode_ordinary_batch(chunk)

            # Flatten: [story_1_tokens, EOT, story_2_tokens, EOT, ...]
            ids = []
            for story_ids in encoded:
                ids.extend(story_ids)
                ids.append(EOT)

            # Save as uint16 — see module docstring for why
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += len(arr)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {total_tokens:,} tokens to {out_path}")
    print(f"  File size: {size_mb:.1f} MB  ({total_tokens * 2 / 1e6:.1f} MB expected at 2 bytes/token)")


if __name__ == "__main__":
    # Validation first — small, ~1 minute. If this works, training will too.
    tokenize_file(RAW_DIR / "TinyStoriesV2-GPT4-valid.txt", OUT_DIR / "val.bin")
    # Training file is 1.6 GB → ~5-10 minutes on CPU
    tokenize_file(RAW_DIR / "TinyStoriesV2-GPT4-train.txt", OUT_DIR / "train.bin")