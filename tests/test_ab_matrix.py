import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cloudstudio_3dgs.training.ab_matrix import (
    build_trainer_ab_matrix,
    verify_trainer_ab_matrix,
)
from cloudstudio_3dgs.training.ab_results import (
    classify_metric_deltas,
    verify_trainer_ab_report,
)
from cloudstudio_3dgs.data.manifest import canonical_json_bytes


def _fixture(root: Path) -> tuple[dict, Path]:
    for name in ("recording", "masks"):
        (root / name).mkdir()
    for name in (
        "dataset.json",
        "masks.json",
        "split.json",
        "sparse_pc.ply",
        "gsplat.lock.json",
    ):
        content = "{}" if name.endswith(".json") else name
        (root / name).write_text(content, encoding="utf-8")
    base = {
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "sparse_pc.ply",
        "gsplat_lock": "gsplat.lock.json",
        "require_person_masks": False,
        "lidar_range_weight": 0.0,
        "seed": 42,
        "factor": 4,
        "max_steps": 4_000,
        "checkpoint_every": 1_000,
    }
    path = root / "base.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return base, path


class TrainerAbMatrixTests(unittest.TestCase):
    def test_metric_delta_classification_uses_positive_as_better(self) -> None:
        self.assertEqual(classify_metric_deltas({"psnr": 0.1, "depth": 0.2}), "IMPROVED")
        self.assertEqual(classify_metric_deltas({"psnr": 0.0, "depth": 0.0}), "TIED")
        self.assertEqual(classify_metric_deltas({"psnr": -0.1, "depth": -0.2}), "REGRESSED")
        self.assertEqual(classify_metric_deltas({"psnr": 0.1, "depth": -0.2}), "MIXED")

    def test_ab_report_signature_is_bound_to_matrix(self) -> None:
        matrix = {"ab_matrix_sha256": "matrix-a"}
        report = {
            "schema_version": 1,
            "ab_matrix_sha256": "matrix-a",
            "gate2_quality_candidate": {"status": "PASS"},
        }
        report["ab_report_sha256"] = hashlib.sha256(
            canonical_json_bytes(report)
        ).hexdigest()
        self.assertEqual(
            verify_trainer_ab_report(report, matrix), report["ab_report_sha256"]
        )
        with self.assertRaisesRegex(ValueError, "another matrix"):
            verify_trainer_ab_report(report, {"ab_matrix_sha256": "matrix-b"})

    def test_matrix_has_signed_shared_inputs_and_single_variable_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, base_path = _fixture(root)
            output = root / "matrix"
            manifest = build_trainer_ab_matrix(
                base,
                base_config_path=base_path,
                output_dir=output,
                experiment_id="d1-factor4",
            )
            actual = verify_trainer_ab_matrix(manifest, output)
            loaded = json.loads(
                (output / "ab_matrix_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(actual, manifest["ab_matrix_sha256"])
            self.assertEqual(loaded, manifest)
            self.assertEqual(len(manifest["arms"]), 5)
            by_arm = {record["arm"]: record for record in manifest["arms"]}
            self.assertTrue(
                all(
                    path.startswith("initialization")
                    or path.startswith("optimizer.means_step_fraction")
                    or path.startswith("strategy.noise_std_fraction")
                    for path in by_arm["knn_only"]["contract_differences_from_reference"]
                )
            )
            self.assertTrue(
                all(
                    path.startswith("color_model")
                    for path in by_arm["sh_only"]["contract_differences_from_reference"]
                )
            )
            self.assertTrue(
                all(
                    path.startswith("loss_contract.rgb_ssim")
                    for path in by_arm["local_ssim_only"][
                        "contract_differences_from_reference"
                    ]
                )
            )

            config_path = output / "configs" / "sh_only.json"
            config_path.write_text(
                config_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config file identity"):
                verify_trainer_ab_matrix(manifest, output)

    def test_matrix_rejects_arm_fields_in_shared_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, base_path = _fixture(root)
            base["background_color"] = [1.0, 1.0, 1.0]
            with self.assertRaisesRegex(ValueError, "arm-specific"):
                build_trainer_ab_matrix(
                    base,
                    base_config_path=base_path,
                    output_dir=root / "matrix",
                    experiment_id="invalid",
                )
