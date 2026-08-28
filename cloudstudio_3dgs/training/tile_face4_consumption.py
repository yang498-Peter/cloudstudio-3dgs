"""Fail-closed structural audit for Tile-specific Face4 crop consumption."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mono_depth import verify_mono_depth_manifest
from cloudstudio_3dgs.data.renderer_masks import verify_renderer_mask_manifest
from cloudstudio_3dgs.training.face_dataset import (
    SAMPLE_ID_SEPARATOR,
    FaceCacheDataset,
    verify_face_manifest,
)
from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest


def audit_tile_face4_consumption(
    *,
    tile_inputs_path: Path,
    tile_inputs_root: Path,
    face_manifest_path: Path,
    face_cache_root: Path,
    renderer_mask_manifest_path: Path,
    mono_depth_manifest_path: Path,
    mono_depth_root: Path,
) -> dict[str, Any]:
    """Audit every Tile crop and load one real aligned sample per Tile on CPU."""

    tile_inputs_path = Path(tile_inputs_path).resolve()
    tile_inputs_root = Path(tile_inputs_root).resolve()
    face_manifest_path = Path(face_manifest_path).resolve()
    face_cache_root = Path(face_cache_root).resolve()
    renderer_mask_manifest_path = Path(renderer_mask_manifest_path).resolve()
    mono_depth_manifest_path = Path(mono_depth_manifest_path).resolve()
    mono_depth_root = Path(mono_depth_root).resolve()
    tile_inputs = json.loads(tile_inputs_path.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(
        tile_inputs,
        root=tile_inputs_root,
        verify_artifacts=True,
    )
    face_manifest = json.loads(face_manifest_path.read_text(encoding="utf-8"))
    face_manifest_sha = verify_face_manifest(face_manifest)
    renderer_mask_manifest = json.loads(
        renderer_mask_manifest_path.read_text(encoding="utf-8")
    )
    renderer_mask_manifest_sha = verify_renderer_mask_manifest(
        renderer_mask_manifest
    )
    if renderer_mask_manifest.get("source_face_manifest_sha256") != face_manifest_sha:
        raise ValueError("renderer mask manifest is bound to a different Face4 cache")
    mono_manifest = json.loads(
        mono_depth_manifest_path.read_text(encoding="utf-8")
    )
    mono_manifest_sha = verify_mono_depth_manifest(mono_manifest)
    if mono_manifest.get("source_face_manifest_sha256") != face_manifest_sha:
        raise ValueError("DA2 manifest is bound to a different Face4 cache")
    if mono_manifest.get("complete_face_cache") is not True:
        raise ValueError("DA2 manifest is incomplete")
    if mono_manifest.get("split") != face_manifest.get("split"):
        raise ValueError("DA2 and Face4 manifests use different splits")

    image_camera = {
        str(image["image_id"]): str(image["camera_id"])
        for image in face_manifest["images"]
    }
    face_intrinsics: dict[tuple[str, str], np.ndarray] = {}
    for camera_id, camera in face_manifest["cameras"].items():
        for face in camera["faces"]:
            face_intrinsics[(str(camera_id), str(face["face_id"]))] = np.asarray(
                face["K_face"], dtype=np.float64
            )
    mono_by_sample = {
        str(record["sample_id"]): record
        for record in mono_manifest.get("records", [])
    }
    if len(mono_by_sample) != len(mono_manifest.get("records", [])):
        raise ValueError("DA2 manifest contains duplicate sample IDs")

    audited_tiles: list[dict[str, Any]] = []
    all_sample_ids: set[str] = set()
    total_view_instances = 0
    total_pixel_load = 0
    for tile in tile_inputs["tiles"]:
        views = tile["views"]
        if int(tile.get("view_count", -1)) != len(views):
            raise ValueError(f"Tile {tile['tile_id']} view count is inconsistent")
        sample_ids = [str(view["sample_id"]) for view in views]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Tile {tile['tile_id']} contains duplicate views")
        for view in views:
            if int(view["pixel_load"]) != int(view["width"]) * int(view["height"]):
                raise ValueError(f"Tile {tile['tile_id']} has an invalid pixel load")
        dataset = FaceCacheDataset(
            face_manifest_path,
            face_cache_root,
            verify_artifacts=True,
            tile_views=views,
            renderer_mask_manifest_path=renderer_mask_manifest_path,
            mono_depth_manifest_path=mono_depth_manifest_path,
            mono_depth_root=mono_depth_root,
        )
        if set(dataset.image_ids) != set(sample_ids) or len(dataset) != len(views):
            raise ValueError(f"Tile {tile['tile_id']} Face4 selection differs from plan")
        index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.image_ids)}
        valid_da2_ids = [
            sample_id
            for sample_id in sample_ids
            if bool(
                mono_by_sample.get(sample_id.replace(SAMPLE_ID_SEPARATOR, "__"), {})
                .get("alignment", {})
                .get("valid")
            )
        ]
        probe_id = valid_da2_ids[len(valid_da2_ids) // 2] if valid_da2_ids else sample_ids[len(sample_ids) // 2]
        probe = dataset[index_by_id[probe_id]]
        crop = next(view for view in views if str(view["sample_id"]) == probe_id)
        base_id, face_id = probe_id.rsplit(SAMPLE_ID_SEPARATOR, 1)
        K_face = face_intrinsics[(image_camera[base_id], face_id)]
        expected_cx = float(K_face[0, 2]) - int(crop["x"])
        expected_cy = float(K_face[1, 2]) - int(crop["y"])
        if not np.isclose(float(probe.K[0, 2]), expected_cx) or not np.isclose(
            float(probe.K[1, 2]), expected_cy
        ):
            raise ValueError(f"Tile {tile['tile_id']} crop principal point was not shifted")
        if probe.image.shape[:2] != (int(crop["height"]), int(crop["width"])):
            raise ValueError(f"Tile {tile['tile_id']} crop raster shape differs from plan")
        tile_pixel_load = sum(int(view["pixel_load"]) for view in views)
        total_view_instances += len(views)
        total_pixel_load += tile_pixel_load
        all_sample_ids.update(sample_ids)
        audited_tiles.append(
            {
                "tile_id": int(tile["tile_id"]),
                "name": str(tile["name"]),
                "status": "CONSUMPTION_READY",
                "view_count": len(views),
                "pixel_load": tile_pixel_load,
                "da2_valid_view_count": len(valid_da2_ids),
                "probe": {
                    "sample_id": probe_id,
                    "shape": [int(value) for value in probe.image.shape],
                    "rgb_mask_true_pixels": int(np.count_nonzero(probe.rgb_mask)),
                    "da2_mask_true_pixels": None
                    if probe.mono_depth_mask is None
                    else int(np.count_nonzero(probe.mono_depth_mask)),
                    "crop_origin_xy": [int(crop["x"]), int(crop["y"])],
                    "principal_point_xy": [
                        float(probe.K[0, 2]),
                        float(probe.K[1, 2]),
                    ],
                },
            }
        )
    unsigned = {
        "schema_version": 1,
        "kind": "mipmap_tile_face4_crop_consumption_audit_v1",
        "status": "CONSUMPTION_READY",
        "evidence_boundary": (
            "all Tile IDs/crops/bounds/pixel loads are structurally checked; "
            "one SHA-verified RGB/mask/DA2 sample is loaded per Tile"
        ),
        "tile_inputs_manifest_sha256": tile_inputs_sha,
        "face_manifest_sha256": face_manifest_sha,
        "renderer_mask_manifest_sha256": renderer_mask_manifest_sha,
        "mono_depth_manifest_sha256": mono_manifest_sha,
        "tile_count": len(audited_tiles),
        "total_view_instances_with_overlap": total_view_instances,
        "unique_face4_sample_count": len(all_sample_ids),
        "total_pixel_load_with_overlap": total_pixel_load,
        "tiles": audited_tiles,
        "training_allowed": False,
        "next_required_artifact": "mesh_depth_normal_consumption_audit",
    }
    report = copy.deepcopy(unsigned)
    report["tile_face4_consumption_audit_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return report
