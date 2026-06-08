"""
Explore TinyStories from local files (skips the datasets library).

Each story in the file is separated by a line containing '<|endoftext|>'.
"""
from pathlib import Path
import statistics

DATA_DIR = Path("data/raw")
TRAIN_FILE = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VAL_FILE = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

def load_stories(path: Path, limit: int | None = None) -> list[str]:
    """Read file and split on <|endoftext|> into a list of stories."""
    print(f"Reading {path}...")
    text = path.read_text(encoding="utf-8")
    stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
    if limit is not None:
        stories = stories[:limit]
    return stories

# Validation file is small — load it all
val_stories = load_stories(VAL_FILE)
print(f"\nValidation stories: {len(val_stories):,}")

# Training file is huge — only load first 10k for inspection
train_sample = load_stories(TRAIN_FILE, limit=10_000)
print(f"Train sample loaded: {len(train_sample):,} stories (out of ~2M total)")

print("\n" + "=" * 60)
print("SAMPLE STORIES")
print("=" * 60)
for i in range(3):
    story = train_sample[i]
    print(f"\n--- Story {i+1}  ({len(story)} chars) ---")
    print(story[:600] + ("..." if len(story) > 600 else ""))

print("\n" + "=" * 60)
print("LENGTH STATISTICS (10k sample)")
print("=" * 60)
lengths = [len(s) for s in train_sample]
print(f"Mean:   {statistics.mean(lengths):.0f} chars")
print(f"Median: {statistics.median(lengths):.0f} chars")
print(f"Min:    {min(lengths)} chars")
print(f"Max:    {max(lengths)} chars")
print(f"P95:    {sorted(lengths)[int(0.95 * len(lengths))]} chars")