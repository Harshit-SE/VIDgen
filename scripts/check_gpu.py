"""Check GPU availability and memory."""

import sys
from pathlib import Path

# Add src to path for direct invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from videosdk.models.memory import get_device, get_gpu_memory

def main() -> None:
    print("=== VideoSDK Hardware & Memory Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Device: {device_name}")
        mem = get_gpu_memory()
        print(f"Total VRAM:     {mem['total_mb']} MB ({mem['total_mb']/1024:.2f} GB)")
        print(f"Allocated VRAM: {mem['allocated_mb']} MB")
        print(f"Free VRAM:      {mem['free_mb']} MB ({mem['free_mb']/1024:.2f} GB)")
    else:
        print("No CUDA GPU detected by PyTorch (running in CPU fallback mode).")

    device = get_device("auto")
    print(f"Default resolved device: {device}")

if __name__ == "__main__":
    main()
