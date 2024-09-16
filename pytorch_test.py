#!/usr/bin/env python3

import torch

x = torch.rand(5, 3).to("cuda")
y = torch.rand(3, 5).to("cuda")

z = x @ y
