import torch
import sys
import os

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

print("\nEnvironment Variables:")
for k, v in os.environ.items():
    if "CUDA" in k or "NVIDIA" in k:
        print(f"{k}={v}")
