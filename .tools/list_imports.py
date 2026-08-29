"""List normal and delay-load PE imports."""

from __future__ import annotations

import argparse

import pefile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pe")
    parser.add_argument("--contains", default="")
    args = parser.parse_args()
    pe = pefile.PE(args.pe, fast_load=False)
    needle = args.contains.lower()
    for kind, entries in (
        ("import", getattr(pe, "DIRECTORY_ENTRY_IMPORT", ())),
        ("delay", getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", ())),
    ):
        for descriptor in entries:
            library = descriptor.dll.decode("utf-8", "replace")
            for symbol in descriptor.imports:
                name = symbol.name.decode("utf-8", "replace") if symbol.name else f"ord{symbol.ordinal}"
                line = f"{kind}\t{library}\t0x{symbol.address - pe.OPTIONAL_HEADER.ImageBase:X}\t{name}"
                if not needle or needle in line.lower():
                    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
