#!/usr/bin/env python3
"""Export a trained CloudStudio checkpoint as a standard 3DGS viewer PLY.

Writes the INRIA gaussian-splatting binary-little-endian layout that
SuperSplat, gsplat viewers, and most web viewers accept:
x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3, with
opacity as logits, scales as log-scales, and SH rest coefficients in
channel-major order. Colors from the rgb_sigmoid model are converted to
an SH DC band so viewers shade them identically.

Coordinates stay in the trainer's local metric frame; consult the run's
coordinate_transform_manifest.json for georeferencing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SH_C0 = 0.28209479177387814


def _sky_layer_from_report(path: Path, payload: dict, gaussian_count: int) -> dict:
    from cloudstudio_3dgs.data.manifest import canonical_json_bytes

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(report.get("sky_layer_report_sha256", ""))
    if not expected:
        raise ValueError("sky layer report has no sky_layer_report_sha256")
    unsigned = dict(report)
    unsigned.pop("sky_layer_report_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError("sky layer report SHA256 mismatch")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("checkpoint has no identity for sky report validation")
    nested = identity.get("source_identity", {})
    dataset_sha = identity.get("dataset_manifest_sha256")
    if dataset_sha is None and isinstance(nested, dict):
        dataset_sha = nested.get("dataset_manifest_sha256")
    if dataset_sha != report.get("dataset_manifest_sha256"):
        raise ValueError("sky layer report is bound to a different dataset")
    if int(report.get("total_gaussian_count", -1)) != gaussian_count:
        raise ValueError("sky layer report total does not match checkpoint params")
    return report


def export_checkpoint_ply(
    checkpoint_path: Path,
    output_path: Path,
    *,
    min_opacity: float = 0.0,
    layer: str = "all",
    sky_layer_report: Path | None = None,
) -> dict:
    import numpy as np
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = payload.get("params") or payload.get("splats")
    if params is None:
        raise ValueError("checkpoint has no params/splats dictionary")

    means = params["means"].detach().float().numpy()
    scales_log = params["scales"].detach().float().numpy()
    quats = params["quats"].detach().float().numpy()
    opacity_logits = params["opacities"].detach().float().numpy().reshape(-1)
    if layer not in {"all", "surface", "sky"}:
        raise ValueError("layer must be one of: all, surface, sky")

    if "sh0" in params:
        f_dc = params["sh0"].detach().float().numpy().reshape(len(means), 3)
        sh_rest = params["shN"].detach().float().numpy()  # [N, K-1, 3]
    else:
        rgb = torch.sigmoid(params["colors"].detach().float()).numpy()
        f_dc = (rgb - 0.5) / SH_C0
        sh_rest = np.zeros((len(means), 0, 3), dtype=np.float32)

    keep = np.ones(len(means), dtype=bool)
    if layer != "all":
        sky_layer = payload.get("sky_layer")
        if not isinstance(sky_layer, dict) and sky_layer_report is not None:
            sky_layer = _sky_layer_from_report(
                sky_layer_report, payload, len(means)
            )
        if not isinstance(sky_layer, dict):
            raise ValueError(
                "surface/sky export requires checkpoint sky_layer metadata "
                "or a signed sky layer report"
            )
        sky_start = int(sky_layer.get("sky_gaussian_start", -1))
        sky_count = int(sky_layer.get("sky_gaussian_count", -1))
        if sky_start < 0 or sky_count <= 0 or sky_start + sky_count != len(means):
            raise ValueError("checkpoint sky_layer boundary is inconsistent with params")
        if layer == "surface":
            keep[sky_start:] = False
        else:
            keep[:sky_start] = False
    if min_opacity > 0.0:
        keep &= 1.0 / (1.0 + np.exp(-opacity_logits)) >= min_opacity
    count = int(keep.sum())
    if count == 0:
        raise ValueError("opacity filter removed every gaussian")

    rest_coeffs = sh_rest.shape[1]
    # INRIA layout: [N, K-1, 3] -> [N, 3, K-1] flattened channel-major.
    f_rest = np.transpose(sh_rest, (0, 2, 1)).reshape(len(means), 3 * rest_coeffs)

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(3 * rest_coeffs)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    columns = np.concatenate(
        [
            means,
            np.zeros((len(means), 3), dtype=np.float32),
            f_dc,
            f_rest,
            opacity_logits[:, None],
            scales_log,
            quats,
        ],
        axis=1,
    ).astype("<f4")[keep]

    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
        + [f"property float {name}" for name in fields]
        + ["end_header", ""]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(columns.tobytes())

    return {
        "gaussians_written": count,
        "gaussians_total": len(means),
        "layer": layer,
        "sh_rest_coefficients": rest_coeffs,
        "bytes": output_path.stat().st_size,
        "fields": len(fields),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.0,
        help="drop gaussians below this sigmoid opacity (0 keeps all)",
    )
    parser.add_argument(
        "--layer",
        choices=("all", "surface", "sky"),
        default="all",
        help="export the full checkpoint or one signed sky-layer partition",
    )
    parser.add_argument(
        "--sky-layer-report",
        type=Path,
        help="signed augmentation report when a trained warm-start omitted layer metadata",
    )
    args = parser.parse_args()
    report = export_checkpoint_ply(
        args.checkpoint,
        args.output,
        min_opacity=args.min_opacity,
        layer=args.layer,
        sky_layer_report=args.sky_layer_report,
    )
    print(
        f"exported {report['gaussians_written']}/{report['gaussians_total']} gaussians, "
        f"{report['sh_rest_coefficients']} SH rest coeffs, "
        f"{report['bytes']/1e6:.1f} MB -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
