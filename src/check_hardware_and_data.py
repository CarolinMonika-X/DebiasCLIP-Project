"""
src/check_hardware_and_data.py
Check hardware acceleration (CUDA/GPU, VRAM) and inspect raw FairVLMed columns.
"""
import torch
import os
import glob
import pandas as pd
from datasets import load_dataset

print("=" * 80)
print("1. HARDWARE ACCELERATION AUDIT")
print("=" * 80)
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available: {cuda_avail}")
if cuda_avail:
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Device:     {device_name}")
    print(f"VRAM Capacity:  {vram_gb:.2f} GB")
else:
    cpu_count = os.cpu_count()
    print(f"Running on CPU ({cpu_count} logical cores)")

print("\n" + "=" * 80)
print("2. DATASET COLUMNS & CLINICAL METADATA AUDIT")
print("=" * 80)
ds = load_dataset("harvardairobotics/FairVLMed", "default", split="train")
print(f"Total instances: {len(ds)}")
print(f"Available columns: {ds.column_names}")

# Print first 2 non-image records
print("\nSample record metadata:")
sample = ds[0]
for k in ds.column_names:
    if k != "image":
        print(f"  {k}: {str(sample[k])[:100]}")

print("\n" + "=" * 80)