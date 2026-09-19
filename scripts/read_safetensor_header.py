#!/usr/bin/env python3

import json
import struct
from pathlib import Path
from sys import argv, version, version_info

if version_info < (3, 7):
    raise RuntimeError(
        f"This script requires Python 3.7 or higher. You are running Python {version}"
    )


if len(argv) == 1:
    raise ValueError("No path to safetensor file supplied")

file_path = Path(argv[1])

if not file_path.parent.exists():
    raise ValueError(f"{file_path} is not in a valid directory.")


with open(file_path, "rb") as f:
    n_bytes = struct.unpack("<Q", f.read(8))[0]
    header_bytes = f.read(n_bytes)
    header = json.loads(header_bytes)
    assert isinstance(header, dict), (
        "Error parsing safetensor header json as dictionary"
    )
print(f"safetensor header consists of {n_bytes} bytes")

header_dict = {"header_size": n_bytes}

header_dict |= header
json_path = file_path.with_suffix(".json")

with open(json_path, "w") as f:
    json.dump(header_dict, f)
print(f"header data written to {json_path}")
