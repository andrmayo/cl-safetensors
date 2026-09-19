#!/usr/bin/env python3

from pathlib import Path
from sys import argv

from safetensors.torch import save_file
from torch import einsum, nn, randn

if len(argv) == 1:
    raise ValueError("Script expects a path argument for saving simple model.")

save_path = Path(argv[1])

if (not save_path.parent.exists()) or save_path.is_dir():
    raise ValueError("Invalid path supplied")

# defaults to torch.float32


class SmallAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # 3-D attention weight:
        self.q_weight = nn.Parameter(randn(4, 8, 16))
        self.k_weight = nn.Parameter(randn(4, 8, 16))

    def forward(self, x):
        return einsum("bshd, hde->bshe", x, self.q_weight)


model = SmallAttention()

print(f"Attention model:\n{model}")
save_file(model.state_dict(), save_path)
print(f"Model saved to {save_path}.")
