"""The surface route defers monocular depth and sky without faking them.

Spatial tiling reads LiDAR visibility and the point-cloud bounds. It never
reads the monocular-depth cache or the sky evidence, so requiring them ahead of
it made the surface route wait on a signal whose training weight is zero. The
route removes that ordering, but only by recording the two stages as deferred
with a reason - never by listing them as complete.
"""

from __future__ import annotations

import copy

import pytest

from cloudstudio_3dgs.pipeline.mipmap_gate import (
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    LIDAR_DEPTH_READY_STATUS,
    ORDERED_STAGES,
    SURFACE_ONLY_DEFERRED_STAGES,
    UPSTREAM_DATA_READY_STATUS,
    advance_spatial_tile_gate_surface_only,
    sign_gate,
)

REASON = "monocular depth carries zero training weight on the LiDAR-first route"


def _lidar_gate(**overrides):
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "profile": GATE_PROFILE,
        "status": LIDAR_DEPTH_READY_STATUS,
        "training_allowed": False,
        "completed_stages": list(ORDERED_STAGES[:12]),
        "next_required_stage": ORDERED_STAGES[12],
        "bindings": {
            "training_dataset_manifest_sha256": "a" * 64,
            "face4_train_manifest_sha256": "b" * 64,
            "lidar_depth_manifest_sha256": "c" * 64,
        },
    }
    payload.update(overrides)
    return sign_gate(payload)


def _tile_plan(monkeypatch, **overrides):
    plan = {
        "retained_tile_count": 4,
        "source_bindings": {
            "training_dataset_manifest_sha256": "a" * 64,
            "face4_train_manifest_sha256": "b" * 64,
            "lidar_depth_manifest_sha256": "c" * 64,
        },
    }
    plan.update(overrides)
    return plan


@pytest.fixture(autouse=True)
def _stub_plan_verification(monkeypatch):
    monkeypatch.setattr(
        "cloudstudio_3dgs.pipeline.mipmap_gate.verify_adaptive_tile_plan",
        lambda plan: "d" * 64,
    )


def test_surface_route_reaches_upstream_ready(monkeypatch):
    gate = advance_spatial_tile_gate_surface_only(
        _lidar_gate(), _tile_plan(monkeypatch), deferral_reason=REASON
    )
    assert gate["status"] == UPSTREAM_DATA_READY_STATUS
    assert gate["training_allowed"] is False
    assert gate["route"] == "surface_only"


def test_deferred_stages_are_never_listed_as_complete(monkeypatch):
    gate = advance_spatial_tile_gate_surface_only(
        _lidar_gate(), _tile_plan(monkeypatch), deferral_reason=REASON
    )
    for stage in SURFACE_ONLY_DEFERRED_STAGES:
        assert stage not in gate["completed_stages"]
    assert tuple(gate["deferred_stages"]) == SURFACE_ONLY_DEFERRED_STAGES
    assert gate["deferral_reason"] == REASON
    # Everything the route did run is still there, in order.
    expected = [s for s in ORDERED_STAGES[:15] if s not in SURFACE_ONLY_DEFERRED_STAGES]
    assert gate["completed_stages"] == expected


def test_reason_is_mandatory(monkeypatch):
    with pytest.raises(ValueError, match="deferral reason"):
        advance_spatial_tile_gate_surface_only(
            _lidar_gate(), _tile_plan(monkeypatch), deferral_reason="   "
        )


def test_plan_claiming_sky_evidence_is_rejected(monkeypatch):
    plan = _tile_plan(monkeypatch)
    plan["source_bindings"]["sky_train_evidence_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="claims sky evidence"):
        advance_spatial_tile_gate_surface_only(
            _lidar_gate(), plan, deferral_reason=REASON
        )


def test_wrong_upstream_status_is_rejected(monkeypatch):
    stale = _lidar_gate(status=UPSTREAM_DATA_READY_STATUS)
    with pytest.raises(ValueError, match="LIDAR_DEPTH_READY"):
        advance_spatial_tile_gate_surface_only(
            stale, _tile_plan(monkeypatch), deferral_reason=REASON
        )


def test_mismatched_plan_bindings_are_rejected(monkeypatch):
    plan = _tile_plan(monkeypatch)
    plan["source_bindings"]["face4_train_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different training inputs"):
        advance_spatial_tile_gate_surface_only(
            _lidar_gate(), plan, deferral_reason=REASON
        )


def test_empty_tile_plan_is_rejected(monkeypatch):
    plan = _tile_plan(monkeypatch)
    plan["retained_tile_count"] = 0
    with pytest.raises(ValueError, match="no trainable tiles"):
        advance_spatial_tile_gate_surface_only(
            _lidar_gate(), plan, deferral_reason=REASON
        )


def test_tampered_gate_signature_is_rejected(monkeypatch):
    gate = _lidar_gate()
    tampered = copy.deepcopy(gate)
    tampered["bindings"]["lidar_depth_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        advance_spatial_tile_gate_surface_only(
            tampered, _tile_plan(monkeypatch), deferral_reason=REASON
        )
