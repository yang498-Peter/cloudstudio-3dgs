"""Find x64 instructions using selected memory displacements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pefile

sys.path.insert(0, str(Path(__file__).with_name("python")))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86_const import X86_OP_MEM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pe")
    parser.add_argument("displacements", nargs="+", type=lambda value: int(value, 0))
    args = parser.parse_args()
    wanted = set(args.displacements)
    pe = pefile.PE(args.pe, fast_load=False)
    base = pe.OPTIONAL_HEADER.ImageBase
    text_section = next(section for section in pe.sections if section.Name.rstrip(b"\0") == b".text")
    disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
    disassembler.detail = True
    disassembler.skipdata = True
    for instruction in disassembler.disasm(text_section.get_data(), base + text_section.VirtualAddress):
        if instruction.id == 0:
            continue
        if any(op.type == X86_OP_MEM and op.mem.disp in wanted for op in instruction.operands):
            print(f"0x{instruction.address - base:X}: {instruction.mnemonic} {instruction.op_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
