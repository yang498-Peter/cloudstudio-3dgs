#!/usr/bin/env python3
"""Import a standard 3DGS viewer PLY as a CloudStudio checkpoint.

The inverse of export_gaussian_ply.py, written so a competitor's model can be
measured by exactly the tools that judge our own runs - same renderer, same
validation views, same energy/agreement/holes. Without this, "their result is
better" stays an impression; with it, it is three numbers and a target.

Conventions match the INRIA layout the exporter documents: scales arrive as
log-scales and opacity as logits, so both pass through UNTRANSFORMED into the
checkpoint, which stores exactly those domains. f_rest arrives channel-major
([3, K-1] per point) and is transposed back to the [N, K-1, 3] the trainer
holds. A degree-0 file simply produces an empty shN.

    python tools/import_gaussian_ply.py --ply USAgs.ply --output usa_gs.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def import_ply(ply_path: Path, output_path: Path) -> dict:
    import numpy as np
    import torch

    from gaussian_health import read_ply_records

    records = read_ply_records(ply_path)
    names = set(records.dtype.names)
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"not a 3DGS PLY, missing: {missing}")

    count = len(records)

    def stack(*fields):
        return np.stack(
            [np.asarray(records[f], dtype=np.float32) for f in fields], axis=1
        )

    means = stack("x", "y", "z")
    scales = stack("scale_0", "scale_1", "scale_2")  # log domain, kept as-is
    quats = stack("rot_0", "rot_1", "rot_2", "rot_3")
    opacities = np.asarray(records["opacity"], dtype=np.float32)  # logit, as-is
    sh0 = stack("f_dc_0", "f_dc_1", "f_dc_2").reshape(count, 1, 3)

    rest_fields = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda n: int(n.split("_")[-1]),
    )
    if len(rest_fields) % 3 != 0:
        raise ValueError(f"f_rest count {len(rest_fields)} is not divisible by 3")
    rest_coeffs = len(rest_fields) // 3
    if rest_coeffs:
        flat = stack(*rest_fields)  # [N, 3*(K-1)] channel-major
        shn = np.transpose(
            flat.reshape(count, 3, rest_coeffs), (0, 2, 1)
        ).copy()  # -> [N, K-1, 3]
    else:
        shn = np.zeros((count, 0, 3), dtype=np.float32)

    # Un-normalised quaternions are common in exported files; the rasterizer
    # normalises internally but keeping the checkpoint clean costs nothing.
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("zero-length quaternion in PLY")
    quats = quats / norms

    params = {
        "means": torch.from_numpy(means),
        "scales": torch.from_numpy(scales),
        "quats": torch.from_numpy(quats),
        "opacities": torch.from_numpy(opacities),
        "sh0": torch.from_numpy(sh0),
        "shN": torch.from_numpy(shn),
    }
    payload = {
        "params": params,
        "source_ply": str(ply_path),
        "imported": True,
        "sh_rest_coeffs": rest_coeffs,
    }
    torch.save(payload, output_path)
    return {
        "count": count,
        "sh_rest_coeffs": rest_coeffs,
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.ply.exists():
        print(f"not found: {args.ply}", file=sys.stderr)
        return 2
    report = import_ply(args.ply, args.output)
    print(f"imported {report['count']:,} gaussians "
          f"(SH rest coeffs {report['sh_rest_coeffs']}) -> {report['output']} "
          f"({report['size_bytes'] / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
