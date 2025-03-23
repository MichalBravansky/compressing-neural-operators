import os
import torch

# Path to your model
model_path = "models/model-fno-darcy-16-resolution-2025-02-05-19-55.pt"

# 1. Get File Size on Disk
file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
print(f"Model file size on disk: {file_size:.2f} MB")
