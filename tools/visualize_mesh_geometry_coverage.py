"""Render representative RGB/mesh-depth/normal coverage contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _face_records(manifest: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for image in manifest["images"]:
        for face in image["faces"]:
            result[f"{image['image_id']}::{face['face_id']}"] = face
    return result


def _resize(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _title(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((42, image.shape[1], 3), 26, np.uint8)
    cv2.putText(
        bar, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (245, 245, 245), 1, cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def _depth_color(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = depth[valid]
    lo, hi = np.quantile(values, [0.02, 0.98]) if len(values) else (0.0, 1.0)
    scaled = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((255.0 * (1.0 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def _normal_color(normal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    color = np.clip((normal[..., ::-1] * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    color[~valid] = 0
    return color


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--da2-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=560)
    args = parser.parse_args()

    mesh_manifest = _read(args.mesh_manifest)
    face_records = _face_records(_read(args.face_manifest))
    da2 = _read(args.da2_manifest)
    da2_records = {str(r["sample_id"]).replace("__", "::"): r for r in da2["records"]}
    records = sorted(
        mesh_manifest["records"], key=lambda item: float(item["mesh_valid_fraction"])
    )
    selected: list[tuple[str, dict]] = []
    for label, quantile in (("low_p10", 0.10), ("median_p50", 0.50), ("high_p90", 0.90)):
        index = min(len(records) - 1, round(quantile * (len(records) - 1)))
        selected.append((label, records[index]))
    failed = [
        record for record in records
        if not da2_records[str(record["sample_id"])]["alignment"]["valid"]
    ]
    if failed:
        selected.append(("da2_rejected", max(failed, key=lambda r: r["mesh_valid_fraction"])))

    args.output.mkdir(parents=True, exist_ok=True)
    sheets: list[np.ndarray] = []
    report: list[dict] = []
    for label, record in selected:
        sample_id = str(record["sample_id"])
        face = face_records[sample_id]
        rgb = cv2.imread(str(args.face_root / str(face["rgb_path"])), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(face["rgb_path"])
        crop = record["crop"]
        x, y = int(crop["x"]), int(crop["y"])
        width, height = int(crop["width"]), int(crop["height"])
        rgb = rgb[y:y + height, x:x + width]
        with np.load(args.mesh_root / str(record["path"]), allow_pickle=False) as payload:
            depth = np.asarray(payload["depth_range_m"], np.float32)
            normal = np.asarray(payload["normal_camera"], np.float32)
            valid = np.asarray(payload["valid"], bool)
        overlay = rgb.copy()
        cyan = np.zeros_like(overlay)
        cyan[..., 0] = 255
        cyan[..., 1] = 210
        overlay[valid] = cv2.addWeighted(overlay[valid], 0.55, cyan[valid], 0.45, 0)
        panels = [
            _title(_resize(rgb, args.panel_width), "RGB Tile crop"),
            _title(_resize(overlay, args.panel_width), "Mesh valid overlay (cyan)"),
            _title(_resize(_depth_color(depth, valid), args.panel_width), "Mesh ray range (2-98%)"),
            _title(_resize(_normal_color(normal, valid), args.panel_width), "Mesh camera normal"),
        ]
        target_height = max(panel.shape[0] for panel in panels)
        panels = [
            cv2.copyMakeBorder(panel, 0, target_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(26, 26, 26))
            for panel in panels
        ]
        sheet = np.hstack(panels)
        header = np.full((62, sheet.shape[1], 3), 245, np.uint8)
        alignment = da2_records[sample_id]["alignment"]
        text = (
            f"{label} | {sample_id} | mesh={record['mesh_valid_fraction']:.1%} "
            f"| DA2 align={alignment['valid']} ratio={alignment['inlier_ratio']:.3f}"
        )
        cv2.putText(header, text, (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
        sheet = np.vstack([header, sheet])
        path = args.output / f"{label}_{sample_id.replace('::', '__')}.png"
        cv2.imwrite(str(path), sheet)
        sheets.append(_resize(sheet, 1200))
        report.append(
            {
                "label": label,
                "sample_id": sample_id,
                "mesh_valid_fraction": record["mesh_valid_fraction"],
                "da2_alignment": alignment,
                "path": path.name,
            }
        )
    overview = np.vstack(sheets)
    cv2.imwrite(str(args.output / "mesh_coverage_representative_overview.png"), overview)
    (args.output / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
