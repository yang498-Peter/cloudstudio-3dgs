from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cloudstudio_3dgs.data.s1_reader import sha256_file
from cloudstudio_3dgs.training.mipmap_tile_geometry import (
    compute_mipmap_tile_geometry,
    explicit_mipmap_scale_calibration_report,
    load_mipmap_tile_geometry,
    sign_tile_geometry_manifest,
    verify_tile_geometry_manifest,
)
from cloudstudio_3dgs.training.scale_calibration import (
    verify_metric_scale_calibration_report,
)


def test_k7_k30_geometry_has_expected_axes_and_normal_rotation() -> None:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8))
    xyz = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    geometry, report = compute_mipmap_tile_geometry(xyz, batch_size=17, workers=1)
    assert geometry["scales_m"].shape == (64, 3)
    np.testing.assert_allclose(
        geometry["scales_m"][:, 2],
        0.5 * geometry["scales_m"][:, 0],
    )
    np.testing.assert_allclose(
        geometry["scales_m"][:, 1],
        geometry["scales_m"][:, 0],
    )
    assert np.all(np.abs(geometry["normals"][:, 2]) > 0.999)
    assert report["scale_knn_including_self"] == 7
    assert report["normal_knn"] == 30
    assert report["nonpositive_scale_replacement_count"] == 0


def test_tile_geometry_manifest_is_signed_and_tamper_evident() -> None:
    manifest = sign_tile_geometry_manifest(
        {
            "schema_version": 1,
            "kind": "mipmap_k7_k30_tile_initialization_geometry_v1",
            "tile_inputs_manifest_sha256": "i" * 64,
            "tile_count": 1,
            "tiles": [{"tile_id": 0, "geometry": {"path": "a.npz"}}],
            "training_allowed": False,
        }
    )
    assert verify_tile_geometry_manifest(manifest) == manifest[
        "tile_geometry_manifest_sha256"
    ]
    tampered = copy.deepcopy(manifest)
    tampered["tile_count"] = 2
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_tile_geometry_manifest(tampered)


def _write_geometry_fixture(root: Path) -> tuple[Path, str, int]:
    count = 32
    scales = np.full((count, 3), 0.02, dtype=np.float32)
    scales[:, 2] = 0.01
    quaternions = np.zeros((count, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    geometry_path = root / "Tile_0" / "initialization_geometry_k7_k30.npz"
    geometry_path.parent.mkdir(parents=True)
    np.savez(
        geometry_path,
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        eigenvalues=np.tile(
            np.asarray([[0.0, 0.1, 0.2]], dtype=np.float32), (count, 1)
        ),
        scales_m=scales,
        quaternions_wxyz=quaternions,
    )
    initialization_sha = "1" * 64
    manifest = sign_tile_geometry_manifest(
        {
            "schema_version": 1,
            "kind": "mipmap_k7_k30_tile_initialization_geometry_v1",
            "tile_inputs_manifest_sha256": "2" * 64,
            "tile_count": 1,
            "tiles": [
                {
                    "tile_id": 0,
                    "name": "Tile_0",
                    "initialization_ply_sha256": initialization_sha,
                    "point_count": count,
                    "geometry": {
                        "path": geometry_path.relative_to(root).as_posix(),
                        "sha256": sha256_file(geometry_path),
                    },
                    "report": {
                        "algorithm": "mipmap_compatible_k7_mean_distance_k30_pca_v1",
                        "evidence_boundary": "ALGORITHM_COMPATIBLE_NOT_VENDOR_BIT_EXACT",
                        "point_count": count,
                    },
                }
            ],
            "training_allowed": False,
        }
    )
    manifest_path = root / "tile_geometry_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, initialization_sha, count


def test_loader_binds_tile_ply_count_and_exact_geometry(tmp_path: Path) -> None:
    manifest_path, initialization_sha, count = _write_geometry_fixture(tmp_path)
    scales, quaternions, report = load_mipmap_tile_geometry(
        manifest_path,
        tmp_path,
        tile_id=0,
        expected_initialization_ply_sha256=initialization_sha,
        expected_count=count,
    )
    assert scales.shape == (count, 3)
    assert quaternions.shape == (count, 4)
    assert report["tile_id"] == 0
    assert report["initialization_ply_sha256"] == initialization_sha
    with pytest.raises(ValueError, match="different initialization PLY"):
        load_mipmap_tile_geometry(
            manifest_path,
            tmp_path,
            tile_id=0,
            expected_initialization_ply_sha256="3" * 64,
            expected_count=count,
        )


def test_explicit_scale_report_is_signed_and_metric_consistent() -> None:
    scales = np.asarray(
        [[0.01, 0.01, 0.005], [0.02, 0.02, 0.01], [0.03, 0.03, 0.015], [0.04, 0.04, 0.02]],
        dtype=np.float32,
    )
    report = explicit_mipmap_scale_calibration_report(
        scales,
        configured_means_lr=1.6e-4,
        configured_noise_lr=5.0,
    )
    assert verify_metric_scale_calibration_report(report) == report[
        "scale_calibration_sha256"
    ]
    assert report["policy"]["mode"] == "external_mipmap_k7_k30"
