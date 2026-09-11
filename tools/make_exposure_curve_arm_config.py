#!/usr/bin/env python3
"""Derive the X1 (frozen camera-curve exposure) arm from an existing X0 arm.

X0 is the per-image-gain control exactly as it ran (``diag_<region>_40_R1_c134``);
X1 is the same config with ``exposure_compensation`` switched to
``mode: camera_curve`` reading the scene-wide frozen curve written by
``tools/fit_exposure_curve.py``.  Nothing else changes: same views, schedule
contract, cap, seed, initialization and evaluation inputs, so the pair differs
in one thing only.

The derived config is named ``<base stem with the label replaced>`` (run_id,
output_dir leaf and file), carries the base config's sha256 and the frozen
curve's sha256 in its ``diag`` block, and can be validated through
``TrainerConfig.from_dict(...).validate()`` (``--validate``; needs the training
venv and hashes the Tile PLY / geometry like the trainer would).

Example::

    python tools/make_exposure_curve_arm_config.py \
        --base C:/Peter/3dgs-runs/house0305_sop/diag_indoor_door_leaf_Tile_1_40_R1_c134.json \
        --base C:/Peter/3dgs-runs/house0305_sop/diag_outdoor_gravel_Tile_0_40_R1_c134.json \
        --frozen-curve C:/Peter/3dgs-runs/house0305_sop/exposure_curve_scene_R1.json \
        --base-label R1 --label X1 \
        --out-dir C:/Peter/3dgs-runs/house0305_sop \
        --copy-dir run_configs/house0305_tiles/diag_v2 --validate
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.exposure import load_curve_payload  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rename(text: str, base_label: str, label: str) -> str:
    """``diag_<region>_<count>_<base_label>[-_]<suffix>`` -> same with ``label``."""
    for sep in ("_", "-"):
        token = f"_{base_label}{sep}"
        if token in text:
            return text.replace(token, f"_{label}{sep}", 1)
    if text.endswith(f"_{base_label}"):
        return text[: -len(base_label)] + label
    raise ValueError(f"cannot find label {base_label!r} in {text!r}")


def derive_x1_config(
    base: dict[str, Any],
    *,
    base_path: Path,
    frozen_curve: Path,
    base_label: str,
    label: str,
    knot_seconds: float,
    prior_weight: float,
    mean_anchor_weight: float,
) -> dict[str, Any]:
    payload = load_curve_payload(frozen_curve)
    if abs(float(payload["knot_seconds"]) - knot_seconds) > 1e-9:
        raise ValueError(
            f"frozen curve knot_seconds {payload['knot_seconds']} != requested {knot_seconds}"
        )
    exposure = dict(base.get("exposure_compensation") or {})
    if not exposure.get("enabled", False):
        raise ValueError("the base arm must have exposure compensation enabled (X0 = per_image)")
    if exposure.get("mode", "per_image") != "per_image":
        raise ValueError("the base arm must be a per_image arm")
    derived = copy.deepcopy(base)
    derived["run_id"] = _rename(str(base["run_id"]), base_label, label)
    output_dir = Path(str(base["output_dir"]))
    derived["output_dir"] = str(output_dir.with_name(_rename(output_dir.name, base_label, label)))
    derived["exposure_compensation"] = {
        **exposure,
        "mode": "camera_curve",
        "knot_seconds": float(knot_seconds),
        # Inert while frozen; recorded so a later learnable arm inherits them.
        "prior_weight": float(prior_weight),
        "mean_anchor_weight": float(mean_anchor_weight),
        "frozen_curve": str(frozen_curve),
    }
    diag = dict(derived.get("diag") or {})
    diag["variant"] = label
    diag["variant_fields"] = {
        **(diag.get("variant_fields") or {}),
        "exposure_compensation": {
            "base": exposure,
            "diag": derived["exposure_compensation"],
        },
    }
    diag["x1"] = {
        "base_arm": str(base_path),
        "base_arm_sha256": _sha256(base_path),
        "base_run_id": str(base["run_id"]),
        "frozen_curve": str(frozen_curve),
        "frozen_curve_sha256": _sha256(frozen_curve),
        "frozen_curve_generator": (payload.get("provenance") or {}).get("generator"),
        "frozen_curve_source_csv_sha256": (payload.get("provenance") or {}).get("source_csv_sha256"),
        "knot_count_by_camera": {c: e["knot_count"] for c, e in payload["cameras"].items()},
    }
    derived["diag"] = diag
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, action="append", required=True)
    parser.add_argument("--frozen-curve", type=Path, required=True)
    parser.add_argument("--base-label", default="R1")
    parser.add_argument("--label", default="X1")
    parser.add_argument("--knot-seconds", type=float, default=10.0)
    parser.add_argument("--prior-weight", type=float, default=1e-2)
    parser.add_argument("--mean-anchor-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy-dir", type=Path, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    written: list[Path] = []
    for base_path in args.base:
        base = json.loads(base_path.read_text(encoding="utf-8"))
        derived = derive_x1_config(
            base,
            base_path=base_path,
            frozen_curve=args.frozen_curve.resolve(),
            base_label=args.base_label,
            label=args.label,
            knot_seconds=args.knot_seconds,
            prior_weight=args.prior_weight,
            mean_anchor_weight=args.mean_anchor_weight,
        )
        name = _rename(base_path.name, args.base_label, args.label)
        text = json.dumps(derived, indent=1)
        targets = [args.out_dir / name]
        if args.copy_dir is not None:
            targets.append(args.copy_dir / name)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            written.append(target)
            print(f"wrote {target}")
        if args.validate:
            from cloudstudio_3dgs.training.trainer import TrainerConfig

            config = TrainerConfig.from_dict(json.loads(text))
            config.validate()
            record = config.contract_dict()["exposure_compensation"]
            print(
                f"validated {derived['run_id']}: mode {record['mode']} frozen sha "
                f"{record['frozen_curve_sha256'][:12]} contract sha {config.contract_sha256()[:12]}"
                if hasattr(config, "contract_sha256")
                else f"validated {derived['run_id']}: mode {record['mode']} frozen sha {record['frozen_curve_sha256'][:12]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
