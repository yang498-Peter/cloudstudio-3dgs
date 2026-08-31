#!/usr/bin/env python3
"""Crop each view's baked backdrop to the Tile crop the trainer will render.

The background library serves a backdrop at whatever size it is asked for, and
resizes when the stored image does not match. That is correct while a view is
rendered whole: the stored image is a downsampled copy of the same frame, so
resizing restores it. It is wrong once a Tile crops the view, because the
request then carries the crop's size and the library squeezes the ENTIRE frame
into the crop rectangle. The backdrop that reaches the loss is a squashed copy
of the whole view rather than the region behind the crop, and the mismatch is
largest exactly where the crop is smallest.

Nothing downstream can detect this: the shapes agree, the compositing is valid,
and the error is silent misregistration. Half of Tile_0's views and most of
Tile_1's are cropped, so the loss sees a wrong backdrop on the majority of
steps, and low-opacity Gaussians let it through into the photometric target.

This writes one background set per Tile, cropped to that Tile's rectangle, so
the library only ever upsamples - the operation it is correct for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.view_backgrounds import (  # noqa: E402
    write_view_background_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-manifest", type=Path, required=True)
    parser.add_argument("--background-root", type=Path, required=True)
    parser.add_argument("--tile-inputs", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--tile-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    payload = json.loads(args.background_manifest.read_text(encoding="utf-8"))
    stored = payload["views"]

    tiles = json.loads(args.tile_inputs.read_text(encoding="utf-8"))["tiles"]
    selected = [t for t in tiles if int(t["tile_id"]) == args.tile_id]
    if len(selected) != 1:
        raise ValueError("tile inputs do not contain a unique selected Tile")
    views = selected[0]["views"]

    # Full face dimensions decide how the crop maps onto the stored backdrop.
    faces = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    camera_of = {
        str(record["image_id"]): str(record["camera_id"])
        for record in dataset["images"]
    }
    face_size: dict[tuple[str, str], tuple[int, int]] = {}
    for camera_id, entry in faces["cameras"].items():
        for face in entry["faces"]:
            face_size[(str(camera_id), str(face["face_id"]))] = (
                int(face["width"]),
                int(face["height"]),
            )

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict] = {}
    cropped_count = 0
    for view in views:
        sample_id = str(view["sample_id"])
        entry = stored.get(sample_id)
        if entry is None:
            raise ValueError(f"no stored backdrop for {sample_id}")
        base, face_id = sample_id.split("::", 1)
        camera_id = camera_of.get(base)
        size = face_size.get((camera_id, face_id))
        if size is None:
            raise ValueError(f"no face dimensions for {sample_id}")
        full_width, full_height = size

        x, y = int(view["x"]), int(view["y"])
        width, height = int(view["width"]), int(view["height"])
        if x + width > full_width or y + height > full_height:
            raise ValueError(f"Tile crop exceeds face bounds for {sample_id}")

        source = out_root.parent / entry["file"]
        if not source.is_file():
            source = args.background_root / entry["file"]
        with Image.open(source) as image:
            backdrop = np.asarray(image.convert("RGB"), dtype=np.uint8)

        if (x, y, width, height) == (0, 0, full_width, full_height):
            # Whole-frame view: the stored backdrop already is this region.
            crop = backdrop
        else:
            # Restore the stored copy to face resolution before slicing, so the
            # crop lands on the same pixels the renderer will see.
            restored = np.asarray(
                Image.fromarray(backdrop).resize(
                    (full_width, full_height), Image.BILINEAR
                ),
                dtype=np.uint8,
            )
            crop = restored[y : y + height, x : x + width]
            cropped_count += 1

        name = f"{base}__{face_id}.png"
        Image.fromarray(crop).save(out_root / name)
        written[sample_id] = {
            "file": name,
            "height": int(crop.shape[0]),
            "width": int(crop.shape[1]),
        }

    metadata = {
        key: payload[key]
        for key in ("split", "dome_source", "dome_sha256", "background_rgb")
        if key in payload
    }
    metadata["tile_id"] = args.tile_id
    metadata["source_background_manifest_sha256"] = payload.get("manifest_sha256")
    metadata["cropped_view_count"] = cropped_count
    signature = write_view_background_manifest(
        out_root / "background_manifest.json", views=written, metadata=metadata
    )
    print(
        f"tile {args.tile_id}: {len(written)} backdrops "
        f"({cropped_count} cropped, {len(written) - cropped_count} whole-frame) "
        f"-> {out_root}  sha256={signature[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
