#!/usr/bin/env python3

import torch

# Set a fixed random seed for deterministic behavior
torch.manual_seed(42)

# If using CUDA, ensure deterministic behavior on CUDA devices as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    # Ensure deterministic operations on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

x = torch.rand(5, 3).to("cuda")
y = torch.rand(3, 5).to("cuda")

z = x @ y
