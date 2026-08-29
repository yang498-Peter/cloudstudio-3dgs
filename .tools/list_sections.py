"""List PE section RVA/raw spans."""

from __future__ import annotations

import argparse
import pefile


parser = argparse.ArgumentParser()
parser.add_argument("pe")
args = parser.parse_args()
pe = pefile.PE(args.pe, fast_load=False)
for section in pe.sections:
    print(
        f"{section.Name.rstrip(bytes([0])).decode('ascii', 'replace')}\t"
        f"RVA 0x{section.VirtualAddress:X}\tVS 0x{section.Misc_VirtualSize:X}\t"
        f"RAW 0x{section.PointerToRawData:X}+0x{section.SizeOfRawData:X}"
    )
