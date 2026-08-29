"""Probe multi-view reprojection consistency of Face4 mesh depth sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_records(probe: dict, face_manifest: dict) -> dict[str, dict]:
    images = {str(item["image_id"]): item for item in face_manifest["images"]}
    face_specs = {
        (str(camera_id), str(face["face_id"])): face
        for camera_id, group in face_manifest["cameras"].items()
        for face in group["faces"]
    }
    result: dict[str, dict] = {}
    for record in probe["records"]:
        sample_id = str(record["sample_id"])
        image_id, face_id = sample_id.split("::", 1)
        image = images[image_id]
        spec = face_specs[(str(image["camera_id"]), face_id)]
        face_to_base = np.eye(4, dtype=np.float64)
        face_to_base[:3, :3] = np.asarray(spec["R_face"], dtype=np.float64)
        c2w = np.asarray(image["c2w"], dtype=np.float64) @ face_to_base
        result[sample_id] = {
            "c2w": c2w,
            "w2c": np.linalg.inv(c2w),
            "K": np.asarray(spec["K_face"], dtype=np.float64),
            "crop": record["crop"],
            "path": record["path"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-probe", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-pixels", type=int, default=20_000)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--threshold-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    probe = _read(args.mesh_probe)
    cameras = _camera_records(probe, _read(args.face_manifest))
    sample_ids = list(cameras)
    centres = np.stack([cameras[key]["c2w"][:3, 3] for key in sample_ids])
    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def arrays(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        if sample_id not in cached:
            with np.load(args.mesh_root / cameras[sample_id]["path"], allow_pickle=False) as payload:
                cached[sample_id] = (
                    np.asarray(payload["depth_range_m"], dtype=np.float32),
                    np.asarray(payload["valid"], dtype=bool),
                )
        return cached[sample_id]

    records: list[dict] = []
    for source_index, sample_id in enumerate(sample_ids):
        camera = cameras[sample_id]
        depth, valid = arrays(sample_id)
        yy, xx = np.nonzero(valid)
        rng = np.random.default_rng(args.seed + source_index)
        if len(xx) > args.max_source_pixels:
            chosen = rng.choice(len(xx), args.max_source_pixels, replace=False)
            yy, xx = yy[chosen], xx[chosen]
        crop = camera["crop"]
        u = xx.astype(np.float64) + float(crop["x"]) + 0.5
        v = yy.astype(np.float64) + float(crop["y"]) + 0.5
        K = camera["K"]
        rays_camera = np.stack(
            [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)],
            axis=1,
        )
        rays_camera /= np.linalg.norm(rays_camera, axis=1, keepdims=True)
        points_camera = rays_camera * depth[yy, xx, None]
        points_world = (
            points_camera @ camera["c2w"][:3, :3].T
            + camera["c2w"][:3, 3]
        )

        distance = np.linalg.norm(centres - camera["c2w"][:3, 3], axis=1)
        neighbor_indexes = np.argsort(distance)
        neighbor_indexes = [
            int(value) for value in neighbor_indexes if int(value) != source_index
        ][: args.max_neighbors]
        observed = np.zeros(len(xx), dtype=np.int16)
        consistent = np.zeros(len(xx), dtype=np.int16)
        occluded = np.zeros(len(xx), dtype=np.int16)
        conflict = np.zeros(len(xx), dtype=np.int16)
        for target_index in neighbor_indexes:
            target_id = sample_ids[target_index]
            target = cameras[target_id]
            target_depth, target_valid = arrays(target_id)
            point_h = np.concatenate(
                [points_world, np.ones((len(points_world), 1), dtype=np.float64)], axis=1
            )
            point_camera = point_h @ target["w2c"].T
            z = point_camera[:, 2]
            projected_u = target["K"][0, 0] * point_camera[:, 0] / np.maximum(z, 1e-12) + target["K"][0, 2]
            projected_v = target["K"][1, 1] * point_camera[:, 1] / np.maximum(z, 1e-12) + target["K"][1, 2]
            target_crop = target["crop"]
            tx = np.rint(projected_u - float(target_crop["x"]) - 0.5).astype(np.int64)
            ty = np.rint(projected_v - float(target_crop["y"]) - 0.5).astype(np.int64)
            inside = z > 0.0
            inside &= tx >= 0
            inside &= ty >= 0
            inside &= tx < int(target_crop["width"])
            inside &= ty < int(target_crop["height"])
            indexes = np.flatnonzero(inside)
            if not len(indexes):
                continue
            indexes = indexes[target_valid[ty[indexes], tx[indexes]]]
            if not len(indexes):
                continue
            expected = np.linalg.norm(point_camera[indexes, :3], axis=1)
            actual = target_depth[ty[indexes], tx[indexes]]
            delta = actual - expected
            observed[indexes] += 1
            consistent[indexes] += np.abs(delta) <= args.threshold_m
            occluded[indexes] += delta < -args.threshold_m
            conflict[indexes] += delta > args.threshold_m

        observable = observed > 0
        support_fraction = np.divide(
            consistent,
            observed,
            out=np.zeros_like(consistent, dtype=np.float64),
            where=observable,
        )
        record = {
            "sample_id": sample_id,
            "sampled_pixels": int(len(xx)),
            "observable_pixels": int(np.count_nonzero(observable)),
            "observable_fraction": float(np.mean(observable)),
            "consistent_observations": int(consistent.sum()),
            "occluded_observations": int(occluded.sum()),
            "conflicting_observations": int(conflict.sum()),
            "support_fraction_p50": float(np.median(support_fraction[observable])) if np.any(observable) else None,
            "supported_by_at_least_one_view_fraction": float(np.mean(consistent > 0)),
        }
        records.append(record)
        print(
            f"cross-view {source_index + 1}/{len(sample_ids)} {sample_id}: "
            f"observable={record['observable_fraction']:.3f} "
            f"support={record['supported_by_at_least_one_view_fraction']:.3f}",
            flush=True,
        )

    observable = np.asarray([item["observable_fraction"] for item in records])
    supported = np.asarray([item["supported_by_at_least_one_view_fraction"] for item in records])
    output = {
        "schema_version": 1,
        "kind": "mesh_depth_cross_view_reprojection_probe",
        "status": "PROBE_NOT_PRODUCTION_FILTER",
        "threshold_m": args.threshold_m,
        "neighbor_policy": f"nearest_{args.max_neighbors}_camera_centres_among_probe_views",
        "summary": {
            "observable_fraction_p50": float(np.median(observable)),
            "supported_by_at_least_one_view_fraction_p50": float(np.median(supported)),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
