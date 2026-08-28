import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio_3dgs.pipeline.mipmap_gate import (
    FRONTEND_READY_STATUS,
    DA2_DEPTH_READY_STATUS,
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    LIDAR_DEPTH_READY_STATUS,
    ORDERED_STAGES,
    MIPMAP_HIGH_TYPE2_PRESET,
    MIPMAP_HIGH_TYPE2_REQUIRED_IMPLEMENTATION,
    TRAINING_IMPLEMENTATION_CONTRACT_KIND,
    TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION,
    TRAINING_IMPLEMENTATION_READY_STATUS,
    FIXED_TOPOLOGY_EVALUATION_READY_STATUS,
    ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS,
    TRAINING_READY_STATUS,
    UPSTREAM_DATA_READY_STATUS,
    advance_training_implementation_gate,
    advance_fixed_topology_evaluation_gate,
    advance_adaptive_growth_gate,
    advance_lidar_depth_gate,
    advance_da2_depth_gate,
    advance_renderer_mask_gate,
    sign_gate,
    sign_training_implementation_contract,
    fixed_topology_evaluation_arm_fingerprint,
    verify_gate,
    verify_training_gate,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.renderer_masks import sign_renderer_mask_manifest
from cloudstudio_3dgs.data.mono_depth import sign_mono_depth_manifest
from cloudstudio_3dgs.training.trainer import TrainerConfig


DATASET_SHA = "1" * 64
SPLIT_SHA = "2" * 64
FACE_SHA = "3" * 64
MASK_SHA = "4" * 64
DA2_SHA = "5" * 64
TILE_SHA = "6" * 64


def _gate(*, ready: bool) -> dict:
    stage_count = 15 if ready else 10
    bindings = {
        "training_dataset_manifest_sha256": DATASET_SHA,
        "split_manifest_sha256": SPLIT_SHA,
        "face4_train_manifest_sha256": FACE_SHA,
        "face4_val_manifest_sha256": FACE_SHA,
        "training_circle_mask_manifest_sha256": MASK_SHA,
    }
    if ready:
        bindings["training_implementation_contract_sha256"] = "7" * 64
    return sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": (
                TRAINING_IMPLEMENTATION_READY_STATUS
                if ready
                else FRONTEND_READY_STATUS
            ),
            "training_allowed": ready,
            "completed_stages": list(ORDERED_STAGES[:stage_count]),
            "next_required_stage": ORDERED_STAGES[stage_count],
            "bindings": bindings,
        }
    )


def _upstream_data_gate() -> dict:
    return sign_gate(
        {
            "schema_version": GATE_SCHEMA_VERSION,
            "profile": GATE_PROFILE,
            "status": UPSTREAM_DATA_READY_STATUS,
            "training_allowed": False,
            "completed_stages": list(ORDERED_STAGES[:15]),
            "next_required_stage": ORDERED_STAGES[15],
            "blocking_reasons": ["training implementation is not aligned"],
            "bindings": {
                "training_dataset_manifest_sha256": DATASET_SHA,
                "split_manifest_sha256": SPLIT_SHA,
                "face4_train_manifest_sha256": FACE_SHA,
                "face4_val_manifest_sha256": FACE_SHA,
                "training_circle_mask_manifest_sha256": MASK_SHA,
                "da2_train_manifest_sha256": DA2_SHA,
                "spatial_tile_plan_manifest_sha256": TILE_SHA,
            },
        }
    )


def _implementation_contract(*, gpu_smoke: bool = True) -> dict:
    return sign_training_implementation_contract(
        {
            "schema_version": TRAINING_IMPLEMENTATION_CONTRACT_SCHEMA_VERSION,
            "kind": TRAINING_IMPLEMENTATION_CONTRACT_KIND,
            "preset": MIPMAP_HIGH_TYPE2_PRESET,
            "trainer_source_sha256": "8" * 64,
            "trainer_config_sha256": "9" * 64,
            "source_bindings": {
                "training_dataset_manifest_sha256": DATASET_SHA,
                "face4_train_manifest_sha256": FACE_SHA,
                "da2_train_manifest_sha256": DA2_SHA,
                "spatial_tile_plan_manifest_sha256": TILE_SHA,
            },
            "implementation": copy.deepcopy(
                MIPMAP_HIGH_TYPE2_REQUIRED_IMPLEMENTATION
            ),
            "verification": {
                "cpu_contract_tests_passed": True,
                "short_gpu_smoke_passed": gpu_smoke,
            },
            "unresolved_blockers": [],
        }
    )


class MipMapPipelineGateTests(unittest.TestCase):
    def test_adaptive_boundary_gate_binds_signed_v26_config(self) -> None:
        config = {
            "run_id": "v26-boundary",
            "mipmap_tile_id": 1,
            "topology_policy": {"mode": "adaptive_growth"},
            "densification_strategy": "default_3dgs",
            "densification_gradient_source": "rgb_only",
            "mcmc_refine_start_iter": 500,
            "mcmc_refine_every": 100,
            "mcmc_refine_stop_iter": 5610,
            "mcmc_noise_injection_stop_iter": 0,
            "mcmc_noise_lr": 0.0,
            "max_steps": 7480,
            "controlled_stop_after_steps": 502,
            "factor": 1,
            "cap_max": 2200000,
            "sh_degree": 0,
            "da2_depth_weight": 0.0,
            "error_weighted_sampling": {"enabled": False},
            "default_strategy": {
                "exact_mipmap_lifecycle": True,
                "grow_grad2d": 0.00015,
                "growth_min_opacity": 0.15,
                "split_scale_m": 0.2,
                "prune_scale_m": 0.2,
                "prune_opa": 0.1,
                "prune_opa_late": 0.05,
                "prune_switch_step": 3740,
                "prune_scale2d": 0.15,
                "reset_every": 300,
                "reset_opacity_cap": 0.2,
                "absgrad": True,
                "revised_opacity": True,
            },
            "tangent_proposal": {
                "enabled": True,
                "reject_unsupported_births": True,
            },
            "geometry_regularization": {
                "enabled": True,
                "opacity_sparsity_weight": 1e-4,
                "scale_upper_weight": 1e-4,
                "anisotropy_weight": 1e-4,
                "max_scale_ratio_to_reference": 8.0,
                "max_anisotropy": 10.0,
            },
        }
        config["config_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(config)
        ).hexdigest()
        gate = advance_adaptive_growth_gate(
            _upstream_data_gate(), config, stage="boundary"
        )
        self.assertEqual(gate["status"], ADAPTIVE_GROWTH_BOUNDARY_READY_STATUS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            verify_training_gate(
                path,
                dataset_manifest_sha256=DATASET_SHA,
                split_manifest_sha256=SPLIT_SHA,
                face_manifest_sha256=FACE_SHA,
                adaptive_growth_config_manifest_sha256=config[
                    "config_manifest_sha256"
                ],
            )

    def test_fixed_topology_evaluation_gate_authorizes_only_signed_arm(self) -> None:
        arm_config = {
            "run_id": "tile1-a0",
            "mipmap_tile_id": 1,
            "topology_policy": {"mode": "strict_fixed"},
            "fixed_topology_schedule": {
                "enabled": True,
                "phase_a_steps": 5,
                "phase_b_steps": 10,
                "audit_steps": [1, 5, 15, 20],
            },
            "max_steps": 20,
            "factor": 1,
            "color_model": "sh",
            "sh_degree": 0,
            "da2_depth_weight": 0.0,
            "lidar_admission": {"enabled": False},
            "tangent_proposal": {"enabled": False},
            "densification_strategy": "default_3dgs",
        }
        plan = {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_plan_v1",
            "tile_id": 1,
            "steps": {"total": 20},
            "arms": [{"arm": "A0", "path": "a0.json"}],
            "training_allowed": False,
            "adaptive_growth_remains_blocked_by": ["gap evidence"],
        }
        plan["evaluation_plan_sha256"] = hashlib.sha256(
            canonical_json_bytes(plan)
        ).hexdigest()
        upstream = _upstream_data_gate()
        readiness = {
            "schema_version": 1,
            "kind": "fixed_topology_evaluation_readiness_v1",
            "status": "FIXED_TOPOLOGY_EVALUATION_PREPARED",
            "upstream_gate_manifest_sha256": upstream["gate_manifest_sha256"],
            "evaluation_plan_sha256": plan["evaluation_plan_sha256"],
            "evidence": {
                "directional_pass": True,
                "phase_a_geometry_frozen": True,
                "core_only_merge_contract": True,
            },
            "training_allowed": False,
            "adaptive_growth_allowed": False,
        }
        readiness["readiness_sha256"] = hashlib.sha256(
            canonical_json_bytes(readiness)
        ).hexdigest()
        gate = advance_fixed_topology_evaluation_gate(
            upstream, readiness, plan, {"A0": arm_config}
        )
        self.assertEqual(gate["status"], FIXED_TOPOLOGY_EVALUATION_READY_STATUS)
        self.assertTrue(gate["training_allowed"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            verify_training_gate(
                path,
                dataset_manifest_sha256=DATASET_SHA,
                split_manifest_sha256=SPLIT_SHA,
                face_manifest_sha256=FACE_SHA,
                fixed_topology_arm_fingerprint_sha256=(
                    fixed_topology_evaluation_arm_fingerprint(arm_config)
                ),
            )
            with self.assertRaisesRegex(ValueError, "not an authorized"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                    fixed_topology_arm_fingerprint_sha256="0" * 64,
                )

    def test_frontend_gate_is_valid_but_blocks_training(self) -> None:
        gate = _gate(ready=False)
        self.assertEqual(verify_gate(gate), gate["gate_manifest_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data/implementation gate is only"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_training_gate_requires_all_stages_and_exact_bindings(self) -> None:
        gate = _gate(ready=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            self.assertEqual(
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                ),
                gate["gate_manifest_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "different inputs"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256="4" * 64,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_upstream_data_ready_does_not_authorize_training(self) -> None:
        gate = _upstream_data_gate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not authorize training"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_upstream_data_ready_allows_only_explicit_implementation_smoke(self) -> None:
        gate = _upstream_data_gate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            self.assertEqual(
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                    allow_implementation_smoke=True,
                ),
                gate["gate_manifest_sha256"],
            )

    def test_implementation_smoke_is_limited_to_two_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "run_id": "bounded-smoke",
                "trainer_preset": "custom",
                "dataset_manifest": str(root / "dataset.json"),
                "recording_root": str(root / "recording"),
                "mask_manifest": str(root / "mask.json"),
                "mask_root": str(root / "masks"),
                "split_manifest": str(root / "split.json"),
                "initialization_ply": str(root / "init.ply"),
                "output_dir": str(root / "output"),
                "gsplat_lock": str(root / "gsplat.lock.json"),
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
                "implementation_smoke_only": True,
                "final_evaluation_artifacts": False,
                "factor": 4,
                "max_steps": 3,
                "checkpoint_every": 3,
            }
            with self.assertRaisesRegex(ValueError, "at most 2 steps"):
                TrainerConfig.from_dict(config).validate()

    def test_legacy_training_ready_literal_is_rejected(self) -> None:
        gate = _upstream_data_gate()
        gate["status"] = "TRAINING_READY"
        gate["training_allowed"] = True
        gate = sign_gate(gate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data/implementation gate is only"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=FACE_SHA,
                )

    def test_implementation_contract_requires_short_gpu_smoke(self) -> None:
        with self.assertRaisesRegex(ValueError, "short_gpu_smoke_passed=true"):
            advance_training_implementation_gate(
                _upstream_data_gate(),
                _implementation_contract(gpu_smoke=False),
            )

    def test_implementation_contract_cannot_omit_closed_source_blockers(self) -> None:
        contract = _implementation_contract()
        unsigned = copy.deepcopy(contract)
        unsigned.pop("training_implementation_contract_sha256")
        del unsigned["implementation"]["face4_lidar_geometry"]
        contract = sign_training_implementation_contract(unsigned)
        with self.assertRaisesRegex(ValueError, "face4_lidar_geometry"):
            advance_training_implementation_gate(_upstream_data_gate(), contract)

    def test_verified_implementation_contract_advances_training_gate(self) -> None:
        advanced = advance_training_implementation_gate(
            _upstream_data_gate(),
            _implementation_contract(),
        )
        self.assertEqual(advanced["status"], TRAINING_IMPLEMENTATION_READY_STATUS)
        self.assertTrue(advanced["training_allowed"])
        self.assertEqual(
            advanced["bindings"]["training_implementation_contract_sha256"],
            _implementation_contract()["training_implementation_contract_sha256"],
        )

    def test_training_requires_face4_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(_gate(ready=True)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires a signed Face4"):
                verify_training_gate(
                    path,
                    dataset_manifest_sha256=DATASET_SHA,
                    split_manifest_sha256=SPLIT_SHA,
                    face_manifest_sha256=None,
                )

    def test_renderer_mask_stage_advances_exactly_one_step(self) -> None:
        gate = _gate(ready=False)

        def renderer(split: str, face_sha: str) -> dict:
            return sign_renderer_mask_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_renderer_mask_cache",
                    "split": split,
                    "source_face_manifest_sha256": face_sha,
                    "policy": {
                        "profile": "mipmap_renderer_visibility_compat_v1",
                        "keep_expression": "face_cache_combined_mask != 0",
                        "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                        "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                    },
                    "masks": [
                        {
                            "image_id": f"{split}-image",
                            "face_id": "front",
                        }
                    ],
                    "summary": {
                        "face_sample_count": 1,
                        "empty_mask_count": 0,
                        "missing_mask_count": 0,
                    },
                }
            )

        train = renderer("train", FACE_SHA)
        val = renderer("val", FACE_SHA)
        advanced = advance_renderer_mask_gate(gate, train, val)
        self.assertEqual(advanced["status"], "RENDERER_MASK_READY")
        self.assertFalse(advanced["training_allowed"])
        self.assertEqual(advanced["next_required_stage"], "new_at_lidar_depth")
        self.assertEqual(len(advanced["completed_stages"]), 11)
        verify_gate(advanced)

    def test_lidar_depth_stage_requires_complete_bound_manifest(self) -> None:
        gate = _gate(ready=False)

        def renderer(split: str) -> dict:
            return sign_renderer_mask_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_renderer_mask_cache",
                    "split": split,
                    "source_face_manifest_sha256": FACE_SHA,
                    "policy": {
                        "profile": "mipmap_renderer_visibility_compat_v1",
                        "keep_expression": "face_cache_combined_mask != 0",
                        "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                        "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                    },
                    "masks": [{"image_id": split, "face_id": "front"}],
                    "summary": {
                        "face_sample_count": 1,
                        "empty_mask_count": 0,
                        "missing_mask_count": 0,
                    },
                }
            )

        renderer_gate = advance_renderer_mask_gate(
            gate, renderer("train"), renderer("val")
        )
        depth = {
            "schema_version": 1,
            "dataset_manifest_sha256": DATASET_SHA,
            "mask_manifest_sha256": MASK_SHA,
            "point_cloud_sha256": "5" * 64,
            "point_cloud_points": 100,
            "complete_dataset": True,
            "total_dataset_images": 1,
            "images": [{"image_id": "image", "path": "depth/image.npz"}],
            "summary": {"image_count": 1},
        }
        import hashlib

        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        depth["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(depth)
        ).hexdigest()
        advanced = advance_lidar_depth_gate(renderer_gate, depth)
        self.assertEqual(advanced["status"], LIDAR_DEPTH_READY_STATUS)
        self.assertEqual(advanced["next_required_stage"], "da2_monocular_depth")
        self.assertEqual(len(advanced["completed_stages"]), 12)
        verify_gate(advanced)
        broken = dict(depth)
        broken["complete_dataset"] = False
        broken.pop("depth_manifest_sha256")
        broken["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(broken)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            advance_lidar_depth_gate(renderer_gate, broken)

    def test_da2_stage_requires_complete_train_and_val(self) -> None:
        gate = _gate(ready=False)

        def renderer(split: str) -> dict:
            return sign_renderer_mask_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_renderer_mask_cache",
                    "split": split,
                    "source_face_manifest_sha256": FACE_SHA,
                    "policy": {
                        "profile": "mipmap_renderer_visibility_compat_v1",
                        "keep_expression": "face_cache_combined_mask != 0",
                        "competitor_reference_expression": "(seg != 255) & (seg != 33)",
                        "label_33_semantics": "UNKNOWN_NOT_INFERRED",
                    },
                    "masks": [{"image_id": split, "face_id": "front"}],
                    "summary": {
                        "face_sample_count": 1,
                        "empty_mask_count": 0,
                        "missing_mask_count": 0,
                    },
                }
            )

        renderer_gate = advance_renderer_mask_gate(
            gate, renderer("train"), renderer("val")
        )
        import hashlib

        from cloudstudio_3dgs.data.manifest import canonical_json_bytes

        depth = {
            "schema_version": 1,
            "dataset_manifest_sha256": DATASET_SHA,
            "mask_manifest_sha256": MASK_SHA,
            "point_cloud_sha256": "5" * 64,
            "point_cloud_points": 100,
            "complete_dataset": True,
            "total_dataset_images": 1,
            "images": [{"image_id": "image", "path": "depth/image.npz"}],
            "summary": {"image_count": 1},
        }
        depth["depth_manifest_sha256"] = hashlib.sha256(
            canonical_json_bytes(depth)
        ).hexdigest()
        lidar_gate = advance_lidar_depth_gate(renderer_gate, depth)

        def mono(split: str) -> dict:
            return sign_mono_depth_manifest(
                {
                    "schema_version": 1,
                    "kind": "face4_da2_relative_depth_cache",
                    "split": split,
                    "source_face_manifest_sha256": FACE_SHA,
                    "dataset_manifest_sha256": DATASET_SHA,
                    "lidar_depth_manifest_sha256": depth["depth_manifest_sha256"],
                    "model": {"checkpoint_sha256": "6" * 64},
                    "metric_alignment": {"iterations": 2000},
                    "complete_face_cache": True,
                    "expected_face_count": 1,
                    "records": [{"sample_id": split}],
                    "summary": {"face_count": 1},
                }
            )

        advanced = advance_da2_depth_gate(lidar_gate, mono("train"), mono("val"))
        self.assertEqual(advanced["status"], DA2_DEPTH_READY_STATUS)
        self.assertEqual(advanced["next_required_stage"], "independent_sky_background")
        self.assertEqual(len(advanced["completed_stages"]), 13)
        verify_gate(advanced)

    def test_reordered_or_tampered_gate_is_rejected(self) -> None:
        gate = _gate(ready=False)
        gate["completed_stages"][1:3] = reversed(gate["completed_stages"][1:3])
        gate = sign_gate(gate)
        with self.assertRaisesRegex(ValueError, "missing, reordered, or skipped"):
            verify_gate(gate)
        gate = _gate(ready=False)
        gate["status"] = "TAMPERED"
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            verify_gate(gate)

    def test_trainer_rejects_shared_kb4_dataset_before_ready_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            split = root / "split.json"
            face = root / "face.json"
            gate_path = root / "gate.json"
            dataset.write_text(
                json.dumps(
                    {
                        "manifest_sha256": DATASET_SHA,
                        "training_lineage": {
                            "independent_at_algorithm_version": (
                                "independent_pos_prior_shared_single_focal_kb4_at_v2"
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            split.write_text(
                json.dumps({"split_manifest_sha256": SPLIT_SHA}), encoding="utf-8"
            )
            face.write_text(
                json.dumps({"face_manifest_sha256": FACE_SHA}), encoding="utf-8"
            )
            gate_path.write_text(json.dumps(_gate(ready=False)), encoding="utf-8")
            base = {
                "run_id": "shared-kb4-gate",
                "dataset_manifest": str(dataset),
                "recording_root": str(root / "recording"),
                "mask_manifest": str(root / "mask.json"),
                "mask_root": str(root / "masks"),
                "split_manifest": str(split),
                "initialization_ply": str(root / "initialization.ply"),
                "output_dir": str(root / "output"),
                "gsplat_lock": str(root / "gsplat.lock.json"),
                "require_person_masks": False,
                "lidar_range_weight": 0.0,
            }
            with self.assertRaisesRegex(ValueError, "requires mipmap_pipeline_gate"):
                TrainerConfig.from_dict(base).validate()
            with self.assertRaisesRegex(ValueError, "data/implementation gate is only"):
                TrainerConfig.from_dict(
                    {
                        **base,
                        "face_cache_manifest": str(face),
                        "face_cache_root": str(root / "faces"),
                        "mipmap_pipeline_gate": str(gate_path),
                    }
                ).validate()


if __name__ == "__main__":
    unittest.main()
