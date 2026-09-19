#!/usr/bin/env python3

from pathlib import Path
from sys import argv, exit

from safetensors.torch import load_file, save_file
import torch

if len(argv) < 1:
    raise ValueError(
        "You must supply path to .safetensors file or directory with precisely 1 .safetensors file"
    )

path = Path(argv[1])

if path.is_dir():
    safetensors = list(path.glob("*.safetensors"))
    if len(safetensors) > 1:
        raise ValueError("Directory contains multiple .safetensors files")
    elif len(safetensors) == 0:
        raise ValueError("Directory does not contain a .safetensors file")
    path = safetensors[0]
else:
    if path.suffix != ".safetensors":
        raise ValueError("File is not a .safetensors file")

state_dict = load_file(path)

count = 0

for k, v in state_dict.items():
    if v.dtype == torch.float32:
        print(f"Tensor {k} is already in fl32 format")
    else:
        print(f"Converting Tensor {k} from {v.dtype} to fl32")
        state_dict[k] = v.to(torch.float32)
        count += 1

print(f"{count} out of {len(state_dict)} tensors converted to fl32 format")

if count > 0:
    answer = input("Overwrite file with converted tensors? [y/N]")
    dest_path = path
    if answer.lower() not in ("y", "yes"):
        answer = input("Provide new file path to save converted tensors? [Y/n]")
        if answer.lower in ("n", "no"):
            exit(1)

        dest_path = answer

    print(f"Saving converted tensors to {dest_path}...")
    save_file(state_dict, dest_path)
    print("...converted tensors saved")
