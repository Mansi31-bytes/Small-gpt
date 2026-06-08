import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"VRAM: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"GPU matmul OK, output shape {y.shape}, dtype {y.dtype}")
else:
    print("CUDA not available — installation issue.")