"""Decode MipMap MVSBlock files using descriptors embedded in divide_engine.exe.

This is a read-only research utility.  It does not require vendor protobuf
sources: the generated executable already contains serialized
FileDescriptorProto records for the messages it consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


DESCRIPTOR_FILES = (
    "engine_type_def.proto",
    "coordinate_system.proto",
    "meta_data.proto",
    "control_point.proto",
    "mvs.proto",
)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _find_descriptor_start(binary: bytes, name: str) -> int:
    encoded = name.encode("utf-8")
    for index, byte in enumerate(binary):
        if byte != 0x0A:
            continue
        try:
            length, payload = _read_varint(binary, index + 1)
        except (IndexError, ValueError):
            continue
        if length == len(encoded) and binary[payload : payload + length] == encoded:
            return index
    raise ValueError(f"embedded descriptor not found: {name}")


def _extract_descriptor(binary: bytes, name: str) -> descriptor_pb2.FileDescriptorProto:
    start = _find_descriptor_start(binary, name)
    best: descriptor_pb2.FileDescriptorProto | None = None
    # Vendor descriptors in this binary are all small.  Successful prefixes
    # occur at message boundaries; the longest valid prefix is the full file.
    for end in range(start + len(name) + 2, min(len(binary), start + 64 * 1024)):
        candidate = descriptor_pb2.FileDescriptorProto()
        try:
            consumed = candidate.MergeFromString(binary[start:end])
        except Exception:
            continue
        if consumed == end - start and candidate.name == name:
            best = candidate
    if best is None:
        raise ValueError(f"unable to recover descriptor: {name}")
    return best


def _build_pool(engine_path: Path) -> descriptor_pool.DescriptorPool:
    binary = engine_path.read_bytes()
    pending = {
        name: _extract_descriptor(binary, name)
        for name in DESCRIPTOR_FILES
    }
    pool = descriptor_pool.DescriptorPool()
    while pending:
        progressed = False
        for name, descriptor in list(pending.items()):
            if any(dependency in pending for dependency in descriptor.dependency):
                continue
            pool.Add(descriptor)
            del pending[name]
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(f"unresolved descriptor dependencies: {unresolved}")
    return pool


def _point3(point: object) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def summarize(engine_path: Path, mvs_path: Path) -> dict[str, object]:
    pool = _build_pool(engine_path)
    message_descriptor = pool.FindMessageTypeByName("mipmap.engine.message.MVSBlock")
    message_class = message_factory.GetMessageClass(message_descriptor)
    block = message_class()
    block.ParseFromString(mvs_path.read_bytes())

    image_rows = []
    pixel_sum = 0
    for image in block.image:
        width = int(image.image_rect.width)
        height = int(image.image_rect.height)
        pixels = width * height
        pixel_sum += pixels
        image_rows.append(
            {
                "id": int(image.img_id),
                "path": str(image.img_path),
                "camera_id": int(image.camera_id),
                "rect": [
                    int(image.image_rect.x),
                    int(image.image_rect.y),
                    width,
                    height,
                ],
                "pixels": pixels,
            }
        )

    bounding_box = [_point3(point) for point in block.bounding_box.corner]
    tight_bounding_box = [_point3(point) for point in block.tight_bounding_box.corner]
    widths = [row["rect"][2] for row in image_rows]
    heights = [row["rect"][3] for row in image_rows]
    summary = {
        "source": str(mvs_path),
        "camera_count": len(block.camera),
        "image_count": len(block.image),
        "point_count": len(block.point),
        "observation_count": len(block.observation),
        "image_meta_data_count": len(block.image_meta_data),
        "pixel_sum": pixel_sum,
        "unique_paths": len({row["path"] for row in image_rows}),
        "image_rect_extent": {
            "min_width": min(widths, default=0),
            "max_width": max(widths, default=0),
            "min_height": min(heights, default=0),
            "max_height": max(heights, default=0),
        },
        "image_rect_histogram": {
            f"{width}x{height}": count
            for (width, height), count in sorted(
                _histogram(
                    (row["rect"][2], row["rect"][3])
                    for row in image_rows
                ).items()
            )
        },
        "bounding_box_corners": bounding_box,
        "tight_bounding_box_corners": tight_bounding_box,
        "images": image_rows,
    }
    return summary


def _histogram(values: object) -> dict[object, int]:
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("mvs", nargs="+", type=Path)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--include-histogram", action="store_true")
    args = parser.parse_args()

    summaries = []
    for path in args.mvs:
        summary = summarize(args.engine, path)
        if not args.include_images:
            summary.pop("images")
        if not args.include_histogram:
            summary.pop("image_rect_histogram")
        summaries.append(summary)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
