from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.geometry.mesh_completion import (
    MeshCompletionConfig,
    coverage_deficit_mask,
    depth_boundary_mask,
    deterministic_stride_mask,
    merge_voxel_candidates,
    verify_mesh_completion_initialization_manifest,
)


def test_coverage_deficit_requires_supported_mesh_and_real_deficit() -> None:
    config = MeshCompletionConfig(
        alpha_floor=0.9,
        confidence_min=0.8,
        depth_tolerance_m=0.05,
    )
    mesh_range = np.full((2, 3), 4.0, dtype=np.float32)
    result = coverage_deficit_mask(
        rgb_mask=np.ones((2, 3), dtype=bool),
        mesh_valid=np.ones((2, 3), dtype=bool),
        mesh_confidence=np.array(
            [[0.8335, 0.8335, 0.6665], [0.8335, 0.8335, 0.8335]],
            dtype=np.float32,
        ),
        source_type=np.array([[3, 4, 3], [3, 3, 3]], dtype=np.uint8),
        rendered_alpha=np.array(
            [[0.5, 0.5, 0.5], [0.95, 0.95, 0.95]], dtype=np.float32
        ),
        rendered_range_m=np.array(
            [[4.0, 4.0, 4.0], [4.0, 4.2, 4.0]], dtype=np.float32
        ),
        mesh_range_m=mesh_range,
        config=config,
    )
    assert result.tolist() == [[True, False, False], [False, True, False]]


def test_stride_mask_is_deterministic_and_bounded() -> None:
    mask = deterministic_stride_mask((9, 10), 4)
    assert np.argwhere(mask).tolist() == [[2, 2], [2, 6], [6, 2], [6, 6]]
    with pytest.raises(ValueError, match="positive"):
        deterministic_stride_mask((2, 2), 0)


def test_depth_boundary_marks_jump_and_dilates() -> None:
    depth = np.array(
        [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]], dtype=np.float32
    )
    boundary = depth_boundary_mask(
        depth, np.ones_like(depth, dtype=bool), threshold_m=0.1, dilation_pixels=0
    )
    assert boundary.tolist() == [
        [False, True, True, False],
        [False, True, True, False],
    ]
    dilated = depth_boundary_mask(
        depth, np.ones_like(depth, dtype=bool), threshold_m=0.1, dilation_pixels=1
    )
    assert dilated.all()


def test_merge_voxel_candidates_averages_and_excludes_lidar_cells() -> None:
    xyz = np.array(
        [[0.011, 0.0, 0.0], [0.014, 0.0, 0.0], [0.031, 0.0, 0.0]],
        dtype=np.float32,
    )
    normals = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    rgb = np.array([[100, 0, 0], [200, 0, 0], [0, 200, 0]], dtype=np.uint8)
    merged = merge_voxel_candidates(
        xyz,
        normals,
        rgb,
        np.array([1.0, 3.0, 1.0], dtype=np.float32),
        voxel_size_m=0.02,
        occupied_xyz=np.array([[0.039, 0.0, 0.0]], dtype=np.float32),
    )
    merged_xyz, merged_normals, merged_rgb, scores, counts = merged
    assert merged_xyz.shape == (1, 3)
    assert merged_xyz[0, 0] == pytest.approx(0.01325)
    assert merged_normals[0].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert merged_rgb[0].tolist() == [175, 0, 0]
    assert scores[0] > 0.0
    assert counts.tolist() == [2]


def test_merge_voxel_candidates_respects_deterministic_cap() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    result = merge_voxel_candidates(
        xyz,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        np.array([1.0, 5.0], dtype=np.float32),
        voxel_size_m=0.1,
        max_points=1,
    )
    assert result[0].tolist() == [[1.0, 0.0, 0.0]]


def test_mesh_completion_manifest_verifies_signature_and_artifacts(tmp_path) -> None:
    combined = tmp_path / "combined.ply"
    geometry = tmp_path / "geometry.npz"
    combined.write_bytes(b"ply")
    geometry.write_bytes(b"geometry")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "kind": "mesh_supported_surfel_completion_initialization_v1",
        "counts": {
            "original_lidar_gaussians": 10,
            "completion_surfels": 2,
            "combined_gaussians": 12,
        },
        "artifacts": {
            "combined_ply": combined.name,
            "combined_ply_sha256": digest(combined),
            "geometry": geometry.name,
            "geometry_sha256": digest(geometry),
        },
    }
    payload["initialization_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    assert (
        verify_mesh_completion_initialization_manifest(payload, root=tmp_path)
        == payload["initialization_manifest_sha256"]
    )
    payload["counts"]["combined_gaussians"] = 13
    with pytest.raises(ValueError, match="signature"):
        verify_mesh_completion_initialization_manifest(payload, root=tmp_path)
