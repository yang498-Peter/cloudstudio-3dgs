"""Find x64 RIP-relative references to one PE RVA (for IAT/data xrefs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pefile

sys.path.insert(0, str(Path(__file__).with_name("python")))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86_const import X86_OP_MEM, X86_REG_RIP


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pe")
    parser.add_argument("target_rva", type=lambda value: int(value, 0))
    args = parser.parse_args()
    pe = pefile.PE(args.pe, fast_load=False)
    base = pe.OPTIONAL_HEADER.ImageBase
    text_section = next(section for section in pe.sections if section.Name.rstrip(b"\0") == b".text")
    disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
    disassembler.detail = True
    disassembler.skipdata = True
    for instruction in disassembler.disasm(text_section.get_data(), base + text_section.VirtualAddress):
        if instruction.id == 0:
            continue
        for operand in instruction.operands:
            if operand.type != X86_OP_MEM or operand.mem.base != X86_REG_RIP:
                continue
            target_rva = instruction.address + instruction.size + operand.mem.disp - base
            if target_rva == args.target_rva:
                print(f"0x{instruction.address - base:X}: {instruction.mnemonic} {instruction.op_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
