#!/usr/bin/env python3
"""Build labeled contact sheets for the signed person-mask review sample."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.person_masks import verify_person_mask_manifest


def _artifact(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if "\\" in value or pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe review overlay path: {value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"review overlay escapes root: {value!r}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-mask-manifest", type=Path, required=True)
    parser.add_argument("--person-mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--thumbnail-pixels", type=int, default=384)
    args = parser.parse_args()
    if min(args.columns, args.rows, args.thumbnail_pixels) <= 0:
        raise ValueError("contact-sheet dimensions must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"contact-sheet output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.person_mask_manifest.read_text(encoding="utf-8"))
    verify_person_mask_manifest(manifest)
    samples = sorted(
        manifest["review_samples"], key=lambda record: str(record["image_id"])
    )
    page_size = args.columns * args.rows
    label_height = 30
    cell_width = args.thumbnail_pixels
    cell_height = args.thumbnail_pixels + label_height
    outputs = []
    for page_index in range(math.ceil(len(samples) / page_size)):
        page_samples = samples[page_index * page_size : (page_index + 1) * page_size]
        sheet = Image.new(
            "RGB",
            (args.columns * cell_width, args.rows * cell_height),
            color=(28, 28, 28),
        )
        draw = ImageDraw.Draw(sheet)
        for cell_index, sample in enumerate(page_samples):
            row, column = divmod(cell_index, args.columns)
            overlay = _artifact(args.person_mask_root, str(sample["overlay_path"]))
            with Image.open(overlay) as opened:
                thumbnail = opened.convert("RGB")
                thumbnail.thumbnail(
                    (args.thumbnail_pixels, args.thumbnail_pixels),
                    Image.Resampling.LANCZOS,
                )
            x = column * cell_width + (cell_width - thumbnail.width) // 2
            y = row * cell_height + (args.thumbnail_pixels - thumbnail.height) // 2
            sheet.paste(thumbnail, (x, y))
            draw.text(
                (column * cell_width + 6, row * cell_height + args.thumbnail_pixels + 7),
                str(sample["image_id"]),
                fill=(245, 245, 245),
            )
        path = args.output / f"contact_sheet_{page_index + 1:02d}.jpg"
        sheet.save(path, format="JPEG", quality=92, subsampling=0, optimize=True)
        outputs.append(path)
    print(f"person review contact sheets: pages={len(outputs)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
