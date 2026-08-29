"""Find direct x64 call sites targeting one PE RVA (read-only analysis)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pefile

sys.path.insert(0, str(Path(__file__).with_name("python")))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86_const import X86_OP_IMM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pe")
    parser.add_argument("target_rva", type=lambda value: int(value, 0))
    args = parser.parse_args()
    pe = pefile.PE(args.pe, fast_load=False)
    base = pe.OPTIONAL_HEADER.ImageBase
    target = base + args.target_rva
    text_section = next(section for section in pe.sections if section.Name.rstrip(b"\0") == b".text")
    code = text_section.get_data()
    start = base + text_section.VirtualAddress
    disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
    disassembler.detail = True
    disassembler.skipdata = True
    for instruction in disassembler.disasm(code, start):
        if instruction.mnemonic != "call" or not instruction.operands:
            continue
        operand = instruction.operands[0]
        if operand.type == X86_OP_IMM and operand.imm == target:
            print(f"0x{instruction.address - base:X}: {instruction.mnemonic} {instruction.op_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
