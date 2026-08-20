#!/usr/bin/env python3
"""Audit Stage-2 BA residual overlap with signed person masks, without rerunning BA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.ba.person_residual_audit import (
    PersonResidualAuditPolicy,
    audit_labeled_residuals,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mask_manifest import (
    verify_dataset_manifest,
    verify_mask_manifest,
)
from cloudstudio_3dgs.data.person_masks import verify_person_mask_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"COLMAP model directory contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").removeprefix("camera/")


def _safe_artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"artifact paths must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes its root: {value!r}")
    return resolved


def _write_overlay(
    source: Path,
    person_mask: Path,
    output: Path,
    points: list[tuple[float, float, bool]],
) -> str:
    with Image.open(source) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(person_mask) as opened:
        person = np.asarray(opened, dtype=np.uint8) > 0
    if person.shape != pixels.shape[:2]:
        raise ValueError(f"person mask shape does not match source image: {person_mask}")
    red = np.zeros_like(pixels)
    red[..., 0] = 255
    pixels[person] = np.rint(0.45 * pixels[person] + 0.55 * red[person]).astype(
        np.uint8
    )
    overlay = Image.fromarray(pixels)
    draw = ImageDraw.Draw(overlay)
    for x, y, on_person in points:
        color = (0, 255, 255) if on_person else (255, 0, 255)
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output, format="PNG", optimize=False, compress_level=9)
    return _sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-mask-manifest", required=True, type=Path)
    parser.add_argument("--person-mask-manifest", required=True, type=Path)
    parser.add_argument("--person-mask-root", required=True, type=Path)
    parser.add_argument("--recording-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--high-residual-threshold-px", type=float, default=5.0)
    parser.add_argument("--minimum-high-observations", type=int, default=20)
    parser.add_argument("--rerun-overlap-fraction", type=float, default=0.3)
    parser.add_argument("--overlay-count", type=int, default=24)
    args = parser.parse_args()

    if not args.model.is_dir():
        raise NotADirectoryError(f"COLMAP model is not a directory: {args.model}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"audit output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = json.loads(args.base_mask_manifest.read_text(encoding="utf-8"))
    person_manifest = json.loads(
        args.person_mask_manifest.read_text(encoding="utf-8")
    )
    dataset_sha = verify_dataset_manifest(dataset)
    base_sha = verify_mask_manifest(base)
    person_sha = verify_person_mask_manifest(person_manifest)
    if base.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("base mask manifest is bound to a different dataset")
    if person_manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("person mask manifest is bound to a different dataset")
    if person_manifest.get("base_mask_manifest_sha256") != base_sha:
        raise ValueError("person mask manifest is bound to a different base mask")

    dataset_by_name = {
        _normalise_name(str(record["path"])): record for record in dataset["images"]
    }
    person_by_id = {
        str(record["image_id"]): record for record in person_manifest["images"]
    }
    if set(person_by_id) != {
        str(record["image_id"]) for record in dataset["images"]
    }:
        raise ValueError("person mask coverage differs from the dataset")

    import pycolmap

    reconstruction = pycolmap.Reconstruction(args.model)
    model_images = sorted(
        reconstruction.images.values(), key=lambda image: _normalise_name(image.name)
    )
    unknown_names = {
        _normalise_name(image.name) for image in model_images
    } - set(dataset_by_name)
    if unknown_names:
        raise ValueError(f"COLMAP model has unknown images: {sorted(unknown_names)[:4]}")
    policy = PersonResidualAuditPolicy(
        high_residual_threshold_px=args.high_residual_threshold_px,
        minimum_high_residual_observations=args.minimum_high_observations,
        rerun_overlap_fraction=args.rerun_overlap_fraction,
    )
    overlay_points: dict[str, list[tuple[float, float, bool]]] = {}

    def observations() -> Iterator[dict[str, Any]]:
        for image_index, image in enumerate(model_images, start=1):
            image_record = dataset_by_name[_normalise_name(image.name)]
            image_id = str(image_record["image_id"])
            person_record = person_by_id[image_id]
            mask_path = _safe_artifact(
                args.person_mask_root, str(person_record["person_mask_path"])
            )
            if _sha256_file(mask_path) != str(person_record["person_mask_sha256"]):
                raise ValueError(f"person mask SHA256 mismatch for {image_id}")
            with Image.open(mask_path) as opened:
                person = np.asarray(opened, dtype=np.uint8) > 0
            points = overlay_points.setdefault(image_id, [])
            for point2D in image.points2D:
                if not point2D.has_point3D():
                    continue
                projected = image.project_point(
                    reconstruction.point3D(point2D.point3D_id).xyz
                )
                if projected is None:
                    continue
                xy = np.asarray(point2D.xy, dtype=np.float64)
                x = int(np.rint(xy[0]))
                y = int(np.rint(xy[1]))
                height, width = person.shape
                if x < 0 or y < 0 or x >= width or y >= height:
                    raise ValueError(f"BA observation is outside person mask for {image_id}")
                error = float(np.linalg.norm(np.asarray(projected) - xy))
                on_person = bool(person[y, x])
                if error >= policy.high_residual_threshold_px and len(points) < 2_000:
                    points.append((float(xy[0]), float(xy[1]), on_person))
                yield {
                    "image_id": image_id,
                    "xy": [float(xy[0]), float(xy[1])],
                    "error_px": error,
                    "on_person": on_person,
                }
            if image_index == 1 or image_index == len(model_images) or image_index % 100 == 0:
                print(
                    f"BA person audit progress: {image_index}/{len(model_images)}",
                    flush=True,
                )

    report = audit_labeled_residuals(observations(), policy)
    top_images = sorted(
        report["images"],
        key=lambda record: (
            -int(record["high_residual_observations"]),
            str(record["image_id"]),
        ),
    )[: max(0, args.overlay_count)]
    overlay_records = []
    dataset_by_id = {
        str(record["image_id"]): record for record in dataset["images"]
    }
    for record in top_images:
        image_id = str(record["image_id"])
        if not overlay_points.get(image_id):
            continue
        source_record = dataset_by_id[image_id]
        source = _safe_artifact(args.recording_root, str(source_record["path"]))
        if _sha256_file(source) != str(source_record["sha256"]):
            raise ValueError(f"source image SHA256 mismatch for overlay {image_id}")
        person_record = person_by_id[image_id]
        mask_path = _safe_artifact(
            args.person_mask_root, str(person_record["person_mask_path"])
        )
        relative = Path("overlays") / f"{image_id}.png"
        overlay_sha = _write_overlay(
            source, mask_path, args.output / relative, overlay_points[image_id]
        )
        overlay_records.append(
            {
                "image_id": image_id,
                "path": relative.as_posix(),
                "sha256": overlay_sha,
                "high_residual_observations": int(
                    record["high_residual_observations"]
                ),
                "plotted_points": len(overlay_points[image_id]),
                "legend": {
                    "red_fill": "person_dynamic_mask",
                    "cyan_circle": "high_residual_on_person",
                    "magenta_circle": "high_residual_off_person",
                },
            }
        )

    report.pop("person_residual_audit_sha256", None)
    report["inputs"] = {
        "dataset_manifest_sha256": dataset_sha,
        "base_mask_manifest_sha256": base_sha,
        "person_mask_manifest_sha256": person_sha,
        "ba_model_sha256": _directory_sha256(args.model),
    }
    report["overlay_samples"] = overlay_records
    report["person_residual_audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    report_path = args.output / "person_residual_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "BA person residual audit complete: "
        f"decision={report['decision']} "
        f"high={report['high_residual_observations']} "
        f"overlap={report['high_residual_person_overlap_fraction']:.6f} "
        f"sha256={report['person_residual_audit_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
