from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cloudstudio_3dgs.data.mono_depth import (
    MONO_DEPTH_KIND,
    MONO_DEPTH_SCHEMA_VERSION,
    sign_mono_depth_manifest,
)
from cloudstudio_3dgs.data.sky_background import (
    IndependentSkyConfig,
    SkyEvidenceConfig,
    build_independent_sky_initialization,
    build_sky_evidence_cache,
    verify_sky_evidence_manifest,
)
from cloudstudio_3dgs.pipeline.adaptive_tiling import (
    AdaptiveTilingConfig,
    AxisAlignedBox,
    ProjectedObservationTable,
    build_adaptive_tile_plan,
    startup_budget_gib,
    verify_adaptive_tile_plan,
)
from cloudstudio_3dgs.training.face_dataset import sign_face_manifest
from cloudstudio_3dgs.training.tile_scheduler import (
    TileMemoryPolicy,
    assert_serial_trace,
    run_tiles_serially,
)


def _observation_table() -> ProjectedObservationTable:
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 20), np.linspace(-1.0, 1.0, 20))
    points = np.column_stack([x.ravel(), y.ravel(), np.linspace(-0.5, 0.5, 400)])
    observation_point = np.repeat(np.arange(len(points)), 4)
    observation_image = np.tile(np.arange(4), len(points))
    base_xy = np.column_stack([(points[:, 0] + 1.0) * 400 + 100, (points[:, 1] + 1.0) * 400 + 100])
    observation_xy = np.repeat(base_xy, 4, axis=0)
    observation_xy += np.tile(np.asarray([[0, 0], [5, 0], [0, 5], [5, 5]]), (len(points), 1))
    return ProjectedObservationTable(
        points=points,
        observation_xy=observation_xy,
        observation_image=observation_image,
        observation_point=observation_point,
        image_sizes=np.full((4, 2), 1100, dtype=np.int64),
    )


def test_adaptive_tile_plan_is_signed_and_has_four_serial_tiles() -> None:
    table = _observation_table()
    plan = build_adaptive_tile_plan(
        table,
        root_box=AxisAlignedBox(np.asarray([-1.1, -1.1, -0.6]), np.asarray([1.1, 1.1, 0.6])),
        force_depth=2,
        config=AdaptiveTilingConfig(
            minimum_anchor_count=50,
            minimum_image_rectangle_pixels=50,
        ),
        source_bindings={"face4_manifest_sha256": "a" * 64},
    )
    assert verify_adaptive_tile_plan(plan) == plan["tile_plan_manifest_sha256"]
    assert plan["leaf_count"] == 4
    assert plan["retained_tile_count"] == 4
    assert [tile["name"] for tile in plan["tiles"]] == [f"Tile_{i}" for i in range(4)]
    assert all(0 <= tile["valid_view_count"] <= 4 for tile in plan["tiles"])
    assert sum(tile["valid_view_count"] for tile in plan["tiles"]) > 0
    core = np.asarray(plan["tiles"][0]["core_box"])
    exported = np.asarray(plan["tiles"][0]["training_and_export_box"])
    np.testing.assert_allclose(core[0] - exported[0], (core[1] - core[0]) * 0.002)
    tampered = json.loads(json.dumps(plan))
    tampered["tiles"][0]["pixel_load"] += 1
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_adaptive_tile_plan(tampered)


def test_startup_budget_and_strict_serial_cache_lifecycle() -> None:
    budget = startup_budget_gib(
        gpu0_available_gib=9.0,
        system_available_gib=32.0,
    )
    assert budget["budget_gib"] == 9.0
    plan = build_adaptive_tile_plan(
        _observation_table(),
        root_box=AxisAlignedBox(np.asarray([-1.1, -1.1, -0.6]), np.asarray([1.1, 1.1, 0.6])),
        force_depth=2,
        config=AdaptiveTilingConfig(
            minimum_anchor_count=50,
            minimum_image_rectangle_pixels=50,
        ),
    )

    class FakeCuda:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            self.calls.append("synchronize")

        def empty_cache(self) -> None:
            self.calls.append("empty_cache")

    cuda = FakeCuda()
    active: list[int] = []

    def train(tile: dict, hook) -> dict:
        assert not active
        active.append(tile["tile_id"])
        released = [hook(step) for step in range(4)]
        assert released == [True, False, True, False]
        active.pop()
        return {"step_count": 4}

    trace = run_tiles_serially(
        plan,
        train,
        cuda=cuda,
        policy=TileMemoryPolicy(empty_cache_interval_steps=2),
    )
    assert trace["completed_tile_count"] == 4
    assert cuda.calls.count("empty_cache") == 12
    assert_serial_trace(trace)


def test_independent_sky_evidence_and_sh1_initialization(tmp_path: Path) -> None:
    face_root = tmp_path / "faces"
    mono_root = tmp_path / "mono"
    output_root = tmp_path / "sky"
    (face_root / "faces").mkdir(parents=True)
    (mono_root / "depth").mkdir(parents=True)
    rgb_path = face_root / "faces" / "sample_rgb.png"
    mask_path = face_root / "faces" / "sample_mask.png"
    Image.fromarray(np.full((32, 32, 3), [80, 140, 230], dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(mask_path)
    spec = {
        "face_id": "front",
        "width": 32,
        "height": 32,
        "half_fov_deg": 45.0,
        "K_face": [[16.0, 0.0, 16.0], [0.0, 16.0, 16.0], [0.0, 0.0, 1.0]],
        "R_face": np.eye(3).tolist(),
    }
    face_manifest = sign_face_manifest(
        {
            "schema_version": 1,
            "kind": "fisheye_face_cache",
            "split": "train",
            "source_identity": {"dataset_manifest_sha256": "d" * 64},
            "cameras": {"left": {"faces": [spec]}},
            "images": [
                {
                    "image_id": "image_0",
                    "camera_id": "left",
                    "c2w": np.eye(4).tolist(),
                    "faces": [
                        {
                            "face_id": "front",
                            "rgb_path": "faces/sample_rgb.png",
                            "mask_path": "faces/sample_mask.png",
                            "mask_true_pixels": 1024,
                        }
                    ],
                }
            ],
        }
    )
    face_manifest_path = face_root / "face_manifest.json"
    face_manifest_path.write_text(json.dumps(face_manifest), encoding="utf-8")
    depth_path = mono_root / "depth" / "sample.npz"
    np.savez_compressed(depth_path, relative_depth=np.full((16, 16), 40.0, dtype=np.float16))
    mono_manifest = sign_mono_depth_manifest(
        {
            "schema_version": MONO_DEPTH_SCHEMA_VERSION,
            "kind": MONO_DEPTH_KIND,
            "split": "train",
            "expected_face_count": 1,
            "complete_face_cache": True,
            "source_face_manifest_sha256": face_manifest["face_manifest_sha256"],
            "dataset_manifest_sha256": "d" * 64,
            "lidar_depth_manifest_sha256": "l" * 64,
            "records": [
                {
                    "sample_id": "image_0__front",
                    "image_id": "image_0",
                    "camera_id": "left",
                    "face_id": "front",
                    "path": "depth/sample.npz",
                    "alignment": {"valid": True, "scale": 1.0, "shift": 0.0},
                }
            ],
        }
    )
    mono_manifest_path = mono_root / "mono_depth_manifest.json"
    mono_manifest_path.write_text(json.dumps(mono_manifest), encoding="utf-8")
    evidence = build_sky_evidence_cache(
        face_manifest_path,
        face_root,
        mono_manifest_path,
        mono_root,
        output_root,
        config=SkyEvidenceConfig(
            far_aligned_range_m=30.0,
            minimum_world_z_direction=0.0,
            minimum_candidate_fraction_per_view=0.01,
            azimuth_bins=8,
            elevation_bins=4,
        ),
    )
    assert evidence["summary"]["accepted_view_count"] == 1
    assert verify_sky_evidence_manifest(evidence, root=output_root, verify_artifacts=True)
    initialization = build_independent_sky_initialization(
        evidence,
        np.zeros(3),
        output_root / "sky_init.npz",
        config=IndependentSkyConfig(count=100, sh_degree=1),
    )
    assert initialization["independent_from_surface_geometry"] is True
    with np.load(output_root / "sky_init.npz") as payload:
        assert payload["means"].shape == (100, 3)
        assert payload["sh0"].shape == (100, 1, 3)
        assert payload["shN"].shape == (100, 3, 3)
