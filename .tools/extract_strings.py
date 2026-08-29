"""Extract printable ASCII strings from a binary, with optional substring filter."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--contains", default="")
parser.add_argument("--min-length", type=int, default=4)
args = parser.parse_args()
data = Path(args.path).read_bytes()
pattern = re.compile(rb"[ -~]{%d,}" % args.min_length)
needle = args.contains.lower()
for match in pattern.finditer(data):
    text = match.group().decode("ascii", "replace")
    if not needle or needle in text.lower():
        print(f"0x{match.start():X}\t{text}")
