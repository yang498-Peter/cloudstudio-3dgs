"""List PE exports, optionally limited to an RVA window."""

from __future__ import annotations

import argparse

import pefile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pe")
    parser.add_argument("--start", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--end", type=lambda value: int(value, 0), default=2**64 - 1)
    args = parser.parse_args()
    pe = pefile.PE(args.pe, fast_load=False)
    for symbol in getattr(pe, "DIRECTORY_ENTRY_EXPORT", ()).symbols:
        if args.start <= symbol.address < args.end:
            name = symbol.name.decode("utf-8", "replace") if symbol.name else f"ord{symbol.ordinal}"
            print(f"0x{symbol.address:X}\t{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
