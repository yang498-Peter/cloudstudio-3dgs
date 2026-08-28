from __future__ import annotations

import copy
import hashlib

import pytest

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    ORDERED_STAGES,
    UPSTREAM_DATA_READY_STATUS,
    sign_gate,
)
from cloudstudio_3dgs.training.mipmap_type2_contract import (
    BLOCKING_REQUIREMENTS,
    build_high_type2_parameter_spec,
    verify_parameter_spec,
)


def _signed(payload: dict, field: str) -> dict:
    result = copy.deepcopy(payload)
    result[field] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return result


def _fixtures() -> tuple[dict, dict, dict]:
    plan = _signed(
        {
            "schema_version": 1,
            "kind": "adaptive_projected_pixel_kd_xy_v1",
            "source_bindings": {},
            "leaf_count": 1,
            "retained_tile_count": 1,
            "execution_contract": {"strict_serial_tiles": True},
            "tiles": [
                {
                    "tile_id": 0,
                    "name": "Tile_0",
                    "core_box": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    "training_and_export_box": [
                        [-0.002, -0.002, -0.002],
                        [1.002, 1.002, 1.002],
                    ],
                    "valid_view_count": 12,
                    "estimated_memory_gib": 2.5,
                }
            ],
        },
        "tile_plan_manifest_sha256",
    )
    gate = sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": UPSTREAM_DATA_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": [],
            "bindings": {
                "spatial_tile_plan_manifest_sha256": plan[
                    "tile_plan_manifest_sha256"
                ],
                "training_dataset_manifest_sha256": "d" * 64,
                "face4_train_manifest_sha256": "f" * 64,
                "renderer_mask_train_manifest_sha256": "r" * 64,
                "da2_train_manifest_sha256": "m" * 64,
                "sky_initialization_manifest_sha256": "s" * 64,
            },
        }
    )
    tile_inputs = _signed(
        {
            "schema_version": 1,
            "kind": "lidar_adaptive_tile_training_inputs_v1",
            "tile_plan_manifest_sha256": plan["tile_plan_manifest_sha256"],
            "tile_count": 1,
            "tiles": [
                {
                    "tile_id": 0,
                    "name": "Tile_0",
                    "view_count": 12,
                    "initialization": {
                        "point_count": 1000,
                        "sha256": "i" * 64,
                    },
                    "recommended_training": {"steps": 240},
                }
            ],
        },
        "tile_inputs_manifest_sha256",
    )
    return gate, plan, tile_inputs


def test_parameter_spec_binds_tiles_and_stays_training_blocked() -> None:
    gate, plan, tile_inputs = _fixtures()
    spec = build_high_type2_parameter_spec(gate, plan, tile_inputs)
    assert verify_parameter_spec(spec) == spec["parameter_spec_sha256"]
    assert spec["training_allowed"] is False
    assert spec["tiles"][0]["total_steps"] == 240
    assert spec["tiles"][0]["stage_step_boundaries"] == [0, 60, 180, 240]
    assert tuple(
        item["id"] for item in spec["implementation_parity"]["requirements"]
    ) == BLOCKING_REQUIREMENTS


def test_parameter_spec_rejects_mismatched_tile_inputs() -> None:
    gate, plan, tile_inputs = _fixtures()
    tile_inputs["tile_plan_manifest_sha256"] = "x" * 64
    unsigned = copy.deepcopy(tile_inputs)
    unsigned.pop("tile_inputs_manifest_sha256")
    tile_inputs["tile_inputs_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(ValueError, match="different spatial Tile plan"):
        build_high_type2_parameter_spec(gate, plan, tile_inputs)


def test_parameter_spec_signature_detects_tampering() -> None:
    gate, plan, tile_inputs = _fixtures()
    spec = build_high_type2_parameter_spec(gate, plan, tile_inputs)
    spec["lidar_first_face4"]["loss"]["rgb_mean_l1"] = 0.8
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_parameter_spec(spec)
