"""Small read-only PE x64 disassembler used for the MipMap binary audit."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import pefile

sys.path.insert(0, str(Path(__file__).with_name("python")))
from capstone import CS_ARCH_X86, CS_MODE_64, Cs
from capstone.x86_const import X86_OP_MEM, X86_REG_RIP


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pe")
    parser.add_argument("start_rva", type=lambda value: int(value, 0))
    parser.add_argument("end_rva", type=lambda value: int(value, 0))
    parser.add_argument("--calls-only", action="store_true")
    args = parser.parse_args()

    pe = pefile.PE(args.pe, fast_load=False)
    image_base = pe.OPTIONAL_HEADER.ImageBase
    import_map: dict[int, str] = {}
    for descriptor in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        library = descriptor.dll.decode("ascii", "replace")
        for imported in descriptor.imports:
            name = imported.name.decode("utf-8", "replace") if imported.name else f"ord{imported.ordinal}"
            import_map[imported.address - image_base] = f"{library}!{name}"
    size = args.end_rva - args.start_rva
    code = pe.get_data(args.start_rva, size)

    disassembler = Cs(CS_ARCH_X86, CS_MODE_64)
    disassembler.detail = True
    disassembler.skipdata = True
    for instruction in disassembler.disasm(code, image_base + args.start_rva):
        rva = instruction.address - image_base
        if instruction.id == 0:
            if not args.calls_only:
                print(f"0x{rva:08X}: .byte    {instruction.op_str}")
            continue
        if args.calls_only and instruction.mnemonic != "call":
            continue
        annotations: list[str] = []
        for operand in instruction.operands:
            if operand.type != X86_OP_MEM or operand.mem.base != X86_REG_RIP:
                continue
            target_va = instruction.address + instruction.size + operand.mem.disp
            target_rva = target_va - image_base
            annotations.append(f"rip->RVA 0x{target_rva:X}")
            if target_rva in import_map:
                annotations.append(import_map[target_rva])
            try:
                raw = pe.get_data(target_rva, 8)
            except pefile.PEFormatError:
                raw = b""
            if len(raw) == 8:
                annotations.append(
                    "u64=0x%016X f32=%g f64=%g"
                    % (struct.unpack("<Q", raw)[0], struct.unpack("<f", raw[:4])[0], struct.unpack("<d", raw)[0])
                )
        suffix = f" ; {'; '.join(annotations)}" if annotations else ""
        print(f"0x{rva:08X}: {instruction.mnemonic:8s} {instruction.op_str}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
