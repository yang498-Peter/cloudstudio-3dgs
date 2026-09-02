"""Rasterize a LiDAR surface mesh into EVERY Face4 sample of a cache.

The tile probe (rasterize_lidar_mesh_face4_probe) validates the recipe on
selected crops; scenes that train on whole faces (house0305) need the same
depth/normal/valid sidecars at full face resolution for every sample, bound
to the face cache and signed for the trainer's mesh_geometry consumption.

Output: <output>/depth/<image>__<face>.npz per sample plus a signed
mesh_geometry_manifest.json (kind face4_lidar_surface_depth_normal).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cloudstudio_3dgs.data.mesh_geometry import (  # noqa: E402
    MESH_GEOMETRY_KIND,
    MESH_GEOMETRY_SCHEMA_VERSION,
    mesh_geometry_npz_bytes,
    sign_mesh_geometry_manifest,
)
from cloudstudio_3dgs.training.face_dataset import (  # noqa: E402
    SAMPLE_ID_SEPARATOR,
    verify_face_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        handle.write(payload)
        handle.close()
        Path(handle.name).replace(path)
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument(
        "--tile-inputs",
        type=Path,
        help="Tile inputs manifest: raster only the selected Tile's views at "
        "their training crops and record the crop, which the trainer requires",
    )
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument(
        "--raster-factor",
        type=int,
        default=1,
        help="raster at 1/N resolution (K scaled by 1/N); the trainer expands "
        "the arrays back to the crop with nearest-neighbour repetition. A full "
        "resolution cache is ~16 MB per face; 2 keeps it at a quarter of that",
    )
    args = parser.parse_args()
    if int(args.raster_factor) < 1:
        parser.error("--raster-factor must be >= 1")

    manifest_path = args.output / "mesh_geometry_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace {manifest_path}")

    import open3d as o3d

    face = json.loads(args.face_manifest.read_text(encoding="utf-8"))
    face_sha = verify_face_manifest(face)
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dataset_images = {str(item["image_id"]): item for item in dataset["images"]}
    face_specs = {
        (str(camera_id), str(payload["face_id"])): payload
        for camera_id, camera in face["cameras"].items()
        for payload in camera["faces"]
    }

    mesh_sha = _sha256_file(args.mesh)
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    print(
        f"mesh: {len(mesh.vertices):,} vertices / {len(mesh.triangles):,} "
        f"triangles, sha {mesh_sha[:12]}",
        flush=True,
    )

    crop_by_sample = None
    if args.tile_inputs is not None:
        tile_inputs = json.loads(args.tile_inputs.read_text(encoding="utf-8"))
        selected = [t for t in tile_inputs["tiles"] if int(t["tile_id"]) == int(args.tile_id)]
        if len(selected) != 1:
            parser.error("Tile inputs do not contain a unique selected Tile")
        crop_by_sample = {
            str(v["sample_id"]): {k: int(v[k]) for k in ("x", "y", "width", "height")}
            for v in selected[0]["views"]
        }
    factor = int(args.raster_factor)

    (args.output / "depth").mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.time()
    total = (
        len(crop_by_sample)
        if crop_by_sample is not None
        else sum(len(image["faces"]) for image in face["images"])
    )

    for image in face["images"]:
        image_id = str(image["image_id"])
        dataset_image = dataset_images[image_id]
        image_c2w = np.asarray(dataset_image["c2w"], dtype=np.float64)
        for face_entry in image["faces"]:
            face_id = str(face_entry["face_id"])
            sample_id = f"{image_id}{SAMPLE_ID_SEPARATOR}{face_id}"
            crop = None
            if crop_by_sample is not None:
                crop = crop_by_sample.get(sample_id)
                if crop is None:
                    continue
            spec = face_specs[(str(image["camera_id"]), face_id)]
            width = int(spec["width"])
            height = int(spec["height"])
            K = np.asarray(spec["K_face"], dtype=np.float64)
            crop_x = crop_y = 0
            if crop is not None:
                crop_x, crop_y = crop["x"], crop["y"]
                width, height = crop["width"], crop["height"]
                K = K.copy()
                K[0, 2] -= crop_x
                K[1, 2] -= crop_y
            face_shape = (height, width)
            if factor > 1:
                K = K.copy()
                K[0, 0] /= factor
                K[1, 1] /= factor
                K[0, 2] /= factor
                K[1, 2] /= factor
                width = -(-width // factor)
                height = -(-height // factor)
            face_to_base = np.eye(4, dtype=np.float64)
            face_to_base[:3, :3] = np.asarray(spec["R_face"], dtype=np.float64)
            w2c = np.linalg.inv(image_c2w @ face_to_base)

            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=o3d.core.Tensor(K.astype(np.float32)),
                extrinsic_matrix=o3d.core.Tensor(w2c.astype(np.float32)),
                width_px=width,
                height_px=height,
            )
            cast = scene.cast_rays(rays)
            t_hit = cast["t_hit"].numpy()
            ray_array = rays.numpy()
            valid = np.isfinite(t_hit) & (t_hit > 0.0)
            safe_t = np.where(valid, t_hit, 0.0)
            hit = ray_array[..., :3] + safe_t[..., None] * ray_array[..., 3:]
            depth_range = np.linalg.norm(
                hit - ray_array[..., :3], axis=-1
            ).astype(np.float32)
            normal_world = cast["primitive_normals"].numpy()
            normal_camera = np.einsum(
                "ij,hwj->hwi", w2c[:3, :3], normal_world
            ).astype(np.float32)

            with Image.open(args.face_root / str(face_entry["mask_path"])) as m:
                face_mask = np.asarray(m.convert("L")) > 0
            if crop is not None:
                face_mask = face_mask[crop_y : crop_y + face_shape[0], crop_x : crop_x + face_shape[1]]
            if factor > 1:
                face_mask = face_mask[::factor, ::factor]
            valid &= face_mask[: valid.shape[0], : valid.shape[1]]
            confidence = valid.astype(np.float32)

            relative = f"depth/{sample_id.replace(SAMPLE_ID_SEPARATOR, '__')}.npz"
            payload = mesh_geometry_npz_bytes(
                depth_range, normal_camera, confidence, valid, source_type=1
            )
            _atomic_bytes(args.output / relative, payload)
            record = {
                "sample_id": sample_id,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [height, width],
                "valid_fraction": float(valid.mean()),
            }
            if crop is not None:
                record["crop"] = dict(crop)
            if factor > 1:
                record["raster_factor"] = factor
                record["face_shape"] = [int(face_shape[0]), int(face_shape[1])]
            records.append(record)
            if len(records) % args.report_every == 0:
                rate = len(records) / max(1e-9, time.time() - started)
                print(
                    f"{len(records)}/{total} faces "
                    f"({rate:.1f}/s, eta {int((total - len(records)) / max(rate, 1e-9))}s)",
                    flush=True,
                )

    manifest = sign_mesh_geometry_manifest(
        {
            "schema_version": MESH_GEOMETRY_SCHEMA_VERSION,
            "kind": MESH_GEOMETRY_KIND,
            "complete_face_cache": True,
            "split": face.get("split"),
            "source_face_manifest_sha256": face_sha,
            "mesh_path": str(args.mesh),
            "mesh_sha256": mesh_sha,
            "raster_factor": factor,
            "expected_face_count": len(records),
            "tile_id": int(args.tile_id) if crop_by_sample is not None else None,
            "records": records,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    fractions = np.array([record["valid_fraction"] for record in records])
    print(
        f"DONE: {len(records)} samples, coverage p10/p50/p90 = "
        f"{np.percentile(fractions, 10):.2f}/{np.percentile(fractions, 50):.2f}/"
        f"{np.percentile(fractions, 90):.2f} -> {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
