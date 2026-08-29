"""Find RIP-relative x64 references to one exact UTF-8/ASCII PE string."""

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
    parser.add_argument("text")
    args = parser.parse_args()

    pe = pefile.PE(args.pe, fast_load=False)
    base = pe.OPTIONAL_HEADER.ImageBase
    needle = args.text.encode("utf-8") + b"\0"
    image = pe.get_memory_mapped_image()
    targets: set[int] = set()
    offset = 0
    while True:
        found = image.find(needle, offset)
        if found < 0:
            break
        targets.add(found)
        print(f"string RVA 0x{found:X}")
        offset = found + 1

    text_section = next(section for section in pe.sections if section.Name.rstrip(b"\0") == b".text")
    code = text_section.get_data()
    start = base + text_section.VirtualAddress
    disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
    disassembler.detail = True
    disassembler.skipdata = True
    for instruction in disassembler.disasm(code, start):
        if instruction.id == 0:
            continue
        for operand in instruction.operands:
            if operand.type != X86_OP_MEM or operand.mem.base != X86_REG_RIP:
                continue
            target_rva = instruction.address + instruction.size + operand.mem.disp - base
            if target_rva in targets:
                print(f"xref RVA 0x{instruction.address - base:X}: {instruction.mnemonic} {instruction.op_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
