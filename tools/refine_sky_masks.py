"""Photometric refinement of a SegFormer sky-mask cache: keep the sky label only where the face
photo itself looks like sky.

Why: the ADE20K sky label is decided at 512 px model input and swallows whole tree crowns - on
house0305 44% of branch pixels (dark, thin structure against the sky) sit inside the eroded sky
mask (research doc 18 6.8). Under sky supervision those pixels lose their photometric terms, are
driven to alpha 0 and may not grow gaussians, so branches come out as haze. Erosion does not help
(38% at 16 px): the label boundary is far outside the crown.

Rule (per face, no ML): with ``g`` the photo luma and ``sky_ref`` the 75th percentile of ``g`` over
the raw sky label,
    bad  = (g < dark_ratio * sky_ref) | (blur3(|grad g| / 8) > edge_ratio * sky_ref)
    sky' = sky_raw & ~dilate(bad, dilate_px)
Bright, smooth pixels (open sky, clouds) stay sky; dark or high-gradient pixels (branches, twig
texture, roof edges) leave the label. ``sky'`` is a subset of the raw label, so every guard the
trainer applies afterwards (erosion, LiDAR proximity, rgb validity) still holds.

The output is a new signed cache (same layout and manifest schema as ``tools/build_sky_masks.py``;
``model`` and ``source_identity`` copied from the source, ``rule`` extended with ``refinement``),
so the trainer verifies it exactly like the original and binds it to the same Face4 cache.

    python tools/refine_sky_masks.py --source-manifest .../sky_mask_train/sky_mask_train.json \
        --face-cache-manifest .../face4_train/face_manifest.json --output-root .../sky_mask_train_pr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloudstudio_3dgs.data.sky_masks import (  # noqa: E402
    SKY_MASK_MANIFEST_SHA_KEY,
    SKY_MASK_VALUE,
    build_sky_mask_manifest,
    load_sky_mask_manifest,
    sky_mask_path_for,
)

DEFAULT_DARK_RATIO = 0.75
DEFAULT_EDGE_RATIO = 0.10
DEFAULT_DILATE_PX = 1
MIN_SKY_PIXELS_FOR_REFERENCE = 1000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refine_sky_mask(
    rgb: np.ndarray,
    sky_raw: np.ndarray,
    *,
    dark_ratio: float = DEFAULT_DARK_RATIO,
    edge_ratio: float = DEFAULT_EDGE_RATIO,
    dilate_px: int = DEFAULT_DILATE_PX,
) -> tuple[np.ndarray, dict[str, float]]:
    """Refine one face's boolean sky label with its (H, W, 3) uint8 RGB photo.

    Returns the refined boolean mask (a subset of ``sky_raw``) and per-face diagnostics.
    """
    import cv2

    if rgb.shape[:2] != sky_raw.shape:
        raise ValueError(f"rgb {rgb.shape[:2]} and sky mask {sky_raw.shape} sizes differ")
    if int(sky_raw.sum()) < MIN_SKY_PIXELS_FOR_REFERENCE:
        return sky_raw.copy(), {"sky_ref": float("nan"), "raw_fraction": float(sky_raw.mean()), "kept_fraction": 1.0}
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sky_ref = float(np.percentile(g[sky_raw], 75))
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    grad = cv2.blur(grad, (3, 3))
    bad = (g < dark_ratio * sky_ref) | (grad > edge_ratio * sky_ref)
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        bad = cv2.dilate(bad.astype(np.uint8), kernel) > 0
    refined = sky_raw & ~bad
    raw_count = int(sky_raw.sum())
    return refined, {
        "sky_ref": sky_ref,
        "raw_fraction": float(sky_raw.mean()),
        "kept_fraction": float(refined.sum()) / raw_count,
    }


def _process_record(job: tuple[dict[str, Any], str, str, str, float, float, int]) -> dict[str, Any]:
    record, source_root, rgb_path, output_root, dark_ratio, edge_ratio, dilate_px = job
    from PIL import Image

    with Image.open(Path(source_root) / str(record["mask_path"])) as source:
        sky_raw = np.asarray(source.convert("L"), dtype=np.uint8) != 0
    with Image.open(rgb_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    refined, diag = refine_sky_mask(rgb, sky_raw, dark_ratio=dark_ratio, edge_ratio=edge_ratio, dilate_px=dilate_px)
    rel = sky_mask_path_for(str(record["image_id"]), str(record["face_id"]))
    out_path = Path(output_root) / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".png.tmp")
    Image.fromarray(np.where(refined, SKY_MASK_VALUE, 0).astype(np.uint8), mode="L").save(tmp, format="PNG", optimize=False)
    os.replace(tmp, out_path)
    valid_pixels = int(record["valid_pixels"])
    sky_pixels = int(refined.sum())
    new_record = dict(record)
    new_record.update(
        {
            "mask_path": rel,
            "mask_sha256": _sha256_file(out_path),
            "sky_pixels": sky_pixels,
            "sky_fraction": sky_pixels / valid_pixels if valid_pixels else 0.0,
            "raw_sky_pixels": int(sky_raw.sum()),
            "refinement_kept_fraction": diag["kept_fraction"],
            "refinement_sky_ref": diag["sky_ref"],
        }
    )
    return new_record


def face_rgb_paths(face_manifest: dict[str, Any], face_root: Path) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for image in face_manifest.get("images", []):
        for face in image.get("faces", []):
            paths[(str(image["image_id"]), str(face["face_id"]))] = face_root / str(face["rgb_path"])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="default: the manifest's directory")
    parser.add_argument("--face-cache-manifest", type=Path, required=True)
    parser.add_argument("--face-cache-root", type=Path, help="default: the face manifest's directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dark-ratio", type=float, default=DEFAULT_DARK_RATIO)
    parser.add_argument("--edge-ratio", type=float, default=DEFAULT_EDGE_RATIO)
    parser.add_argument("--dilate-px", type=int, default=DEFAULT_DILATE_PX)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="first N records only (smoke)")
    args = parser.parse_args(argv)

    source_root = args.source_root or args.source_manifest.parent
    face_root = args.face_cache_root or args.face_cache_manifest.parent
    output_root: Path = args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output root exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source = load_sky_mask_manifest(args.source_manifest)
    face_manifest = json.loads(args.face_cache_manifest.read_text(encoding="utf-8"))
    if source.get("source_face_manifest_sha256") != face_manifest.get("face_manifest_sha256"):
        raise SystemExit("source sky mask manifest is bound to a different Face4 cache")
    rgb_paths = face_rgb_paths(face_manifest, face_root)
    records = list(source["masks"])
    if args.limit:
        records = records[: args.limit]
    jobs = []
    for record in records:
        key = (str(record["image_id"]), str(record["face_id"]))
        if key not in rgb_paths:
            raise SystemExit(f"face cache has no rgb for {key}")
        jobs.append((record, str(source_root), str(rgb_paths[key]), str(output_root), args.dark_ratio, args.edge_ratio, args.dilate_px))

    started = time.time()
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        # in-process (also what the tests use: a spec-loaded module cannot be pickled to workers)
        outputs = map(_process_record, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        outputs = pool.map(_process_record, jobs, chunksize=8)
    try:
        for index, new_record in enumerate(outputs, start=1):
            results.append(new_record)
            if index % 200 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} faces  {time.time() - started:.0f} s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    rule = dict(source["rule"])
    rule["refinement"] = {
        "tool": "tools/refine_sky_masks.py",
        "decision": (
            "sky_raw AND NOT dilate(dark OR edge, dilate_px); dark = luma < dark_ratio * p75(luma over sky_raw); "
            "edge = blur3(|Sobel3 luma| / 8) > edge_ratio * p75(luma over sky_raw); faces with fewer than "
            f"{MIN_SKY_PIXELS_FOR_REFERENCE} raw sky pixels are copied unchanged"
        ),
        "dark_ratio": float(args.dark_ratio),
        "edge_ratio": float(args.edge_ratio),
        "dilate_px": int(args.dilate_px),
        "source_sky_mask_manifest_sha256": str(source[SKY_MASK_MANIFEST_SHA_KEY]),
        "subset_of_source": True,
    }
    manifest = build_sky_mask_manifest(
        split=str(source["split"]),
        source_face_manifest_sha256=str(source["source_face_manifest_sha256"]),
        source_identity=dict(source["source_identity"]),
        model=dict(source["model"]),
        rule=rule,
        records=results,
    )
    out_manifest = output_root / args.source_manifest.name
    tmp = out_manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_manifest)

    raw_total = sum(int(r["raw_sky_pixels"]) for r in results)
    kept_total = sum(int(r["sky_pixels"]) for r in results)
    kept = [float(r["refinement_kept_fraction"]) for r in results if int(r["raw_sky_pixels"]) >= MIN_SKY_PIXELS_FOR_REFERENCE]
    stats = {
        "faces": len(results),
        "raw_sky_pixels": raw_total,
        "refined_sky_pixels": kept_total,
        "kept_fraction_overall": kept_total / raw_total if raw_total else 1.0,
        "kept_fraction_per_face": {
            "p10": float(np.percentile(kept, 10)) if kept else 1.0,
            "p50": float(np.percentile(kept, 50)) if kept else 1.0,
            "p90": float(np.percentile(kept, 90)) if kept else 1.0,
        },
        "seconds": time.time() - started,
        "manifest": str(out_manifest),
        "manifest_sha256": manifest[SKY_MASK_MANIFEST_SHA_KEY],
    }
    (output_root / "refine_stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
