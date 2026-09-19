#!/usr/bin/env python3

from pathlib import Path
from sys import argv

from safetensors.torch import save_file
from torch import nn, randn

if len(argv) == 1:
    raise ValueError("Script expects a path argument for saving simple model.")

save_path = Path(argv[1])

if (not save_path.parent.exists()) or save_path.is_dir():
    raise ValueError("Invalid path supplied")

# defaults to torch.float32


class ZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(randn(()))


model = ZeroModel()
(f"Empty attention model:\n{model}")
save_file(model.state_dict(), save_path)
print(f"Model saved to {save_path}.")
