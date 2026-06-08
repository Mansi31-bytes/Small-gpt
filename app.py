import streamlit as st
import torch
import tiktoken
import sys, math
from pathlib import Path

st.set_page_config(page_title="TinyStories GPT", page_icon="📖", layout="wide")

@st.cache_resource(show_spinner="Loading model weights...")
def load_model(pe_type: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from src.model import GPTConfig, GPT
    from huggingface_hub import hf_hub_download
    fname     = "abs_pe.pt" if pe_type == "Absolute PE" else "rope_pe.pt"
    ckpt_path = hf_hub_download(repo_id="Mansi3110/tinystories-gpt", filename=fname)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg   = GPTConfig(**ckpt["model_cfg"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt["val_loss"], ckpt["step"]

@st.cache_resource
def load_tokenizer():
    return tiktoken.get_encoding("gpt2")

with st.sidebar:
    st.title("📖 TinyStories GPT")
    st.caption("Decoder-only transformer built from scratch in PyTorch. "
               "Trained on TinyStories V2 (2.7M stories, 541M tokens).")
    st.divider()
    pe_type     = st.radio("Positional encoding", ["Absolute PE", "RoPE"])
    st.divider()
    temperature = st.slider("Temperature", 0.5, 1.5, 0.8, 0.05)
    max_tokens  = st.slider("Max new tokens", 50, 400, 200, 10)
    top_k       = st.slider("Top-k", 10, 100, 50, 5)

model, cfg, val_loss, step = load_model(pe_type)
enc = load_tokenizer()
EOT = enc.eot_token

with st.sidebar:
    st.divider()
    st.subheader("Model stats")
    c1, c2 = st.columns(2)
    c1.metric("Params", f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    c2.metric("Val loss", f"{val_loss:.4f}")
    c1.metric("Perplexity", f"{math.exp(val_loss):.2f}")
    c2.metric("Trained steps", f"{step:,}")

st.header(f"Generate with {pe_type}")

EXAMPLES = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom was a curious dog who loved to explore",
    "One sunny morning, a tiny rabbit",
    "The old wizard looked at the children and said,",
    "In a small house near the forest, a young boy named Max",
]

prompt = st.text_area("Prompt", value=EXAMPLES[0], height=80)

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"#{i+1}", help=ex):
        prompt = ex
        st.rerun()

if st.button("✨ Generate", type="primary", use_container_width=True):
    with st.spinner("Generating..."):
        ids = enc.encode(prompt)
        x   = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=max_tokens,
                                 temperature=temperature, top_k=top_k)
        tokens = out[0].tolist()
        if EOT in tokens[len(ids):]:
            tokens = tokens[:tokens.index(EOT, len(ids))]
        text = enc.decode(tokens)
    st.markdown("---")
    st.markdown(text)
    st.caption(f"{len(tokens)-len(ids)} new tokens · {pe_type} · "
               f"temp {temperature} · top-k {top_k}")
else:
    st.markdown("*Your generated story will appear here...*")

with st.expander("📊 Experimental results"):
    st.markdown("""
| | Absolute PE | RoPE |
|---|---|---|
| Val loss | 1.6792 | **1.6257** |
| Perplexity | 5.36 | **5.08** |

At 2× training context (512 tokens): absolute PE perplexity +163%, RoPE +65%.
    """)