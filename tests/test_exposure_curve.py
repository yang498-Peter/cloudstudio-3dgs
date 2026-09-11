"""ExposureCurve (X1): per-camera time curve, anchor, frozen curve, per_image byte-identity.

Pins:
* the curve evaluates to the knot value at a knot and to the linear blend
  between knots, with constant extrapolation outside the grid;
* the soft anchor drives each camera's mean log gain (over its images) to
  zero on its own, and a globally-balanced left/right drift is still penalised;
* a frozen curve reproduces the gains written in the file, is not optimised,
  and rejects a grid that does not match the config;
* the per_image default is unchanged: the same config dict, the same contract
  keys, the same compensator state and the same gained forward as before the
  ``mode`` field existed;
* the trainer builds the curve under ``camera_curve`` and registers it as the
  ``exposure_curve_knots`` auxiliary parameter without an optimizer when frozen.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # torch is an optional training dependency
    torch = None

from cloudstudio_3dgs.training.exposure import (
    ExposureCompensationConfig,
    ExposureCompensator,
    ExposureCurve,
    build_curve_payload,
    curve_interpolation_weights,
    curve_knot_count,
    evaluate_curve,
    load_curve_payload,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "cloudstudio_3dgs" / "training" / "trainer.py"
LEGACY_CONTRACT_KEYS = [
    "enabled",
    "learning_rate",
    "regularization_weight",
    "max_abs_log_gain",
    "zero_mean_projection",
    "mean_anchor_weight",
    "mean_anchor_beta",
]
T0 = 1_772_726_380_342_686_976  # house0305 first frame, ns


def _rig(count: int, *, step_s: float = 0.5) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """count images alternating left/right every step_s seconds."""
    ids = [f"img_{k:03d}" for k in range(count)]
    cameras = {i: ("left" if k % 2 == 0 else "right") for k, i in enumerate(ids)}
    times = {i: T0 + int(k * step_s * 1e9) for k, i in enumerate(ids)}
    return ids, cameras, times


def _curve_config(**overrides: object) -> ExposureCompensationConfig:
    values: dict[str, object] = {
        "enabled": True,
        "mode": "camera_curve",
        "knot_seconds": 10.0,
        "prior_weight": 0.0,
        "mean_anchor_weight": 0.0,
    }
    values.update(overrides)
    return ExposureCompensationConfig(**values)


class CurveMathTests(unittest.TestCase):
    def test_knot_count_covers_the_span(self) -> None:
        self.assertEqual(curve_knot_count(221.0, 10.0), 24)
        self.assertEqual(curve_knot_count(0.0, 10.0), 2)
        self.assertEqual(curve_knot_count(10.0, 10.0), 3)
        with self.assertRaises(ValueError):
            curve_knot_count(-1.0, 10.0)

    def test_weights_at_between_and_beyond_knots(self) -> None:
        taps = curve_interpolation_weights([0.0, 10.0, 5.0, 25.0, -3.0, 99.0], knot_seconds=10.0, knot_count=4)
        self.assertEqual(taps[0], (0, 0.0))
        self.assertEqual(taps[1], (1, 0.0))
        self.assertEqual(taps[2], (0, 0.5))
        self.assertEqual(taps[3], (2, 0.5))
        self.assertEqual(taps[4], (0, 0.0))  # clamped below the grid
        self.assertEqual(taps[5], (2, 1.0))  # clamped to the last knot
        knots = [0.0, 1.0, -1.0, 3.0]
        values = evaluate_curve(knots, [0.0, 10.0, 5.0, 25.0, -3.0, 99.0], knot_seconds=10.0)
        self.assertEqual(values, [0.0, 1.0, 0.5, 1.0, 0.0, 3.0])

    def test_payload_roundtrip_and_validation(self) -> None:
        payload = build_curve_payload(
            knot_seconds=10.0, time_origin_ns=T0, cameras={"left": [0.1, 0.2], "right": [0.0, 0.0, 0.0]},
            max_abs_log_gain=math.log(2.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_curve_payload(path)
            self.assertEqual(loaded["cameras"]["left"]["knot_log_gains"], [0.1, 0.2])
            path.write_text(json.dumps({"kind": "other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an exposure camera curve"):
                load_curve_payload(path)
        with self.assertRaisesRegex(ValueError, "at least two knots"):
            build_curve_payload(knot_seconds=10.0, time_origin_ns=T0, cameras={"left": [0.1]}, max_abs_log_gain=0.7)


class ConfigTests(unittest.TestCase):
    def test_per_image_default_contract_is_byte_identical(self) -> None:
        legacy = {
            "enabled": False,
            "learning_rate": 5e-3,
            "regularization_weight": 1e-2,
            "max_abs_log_gain": 0.6931471805599453,
            "zero_mean_projection": False,
            "mean_anchor_weight": 0.0,
            "mean_anchor_beta": 0.1,
        }
        self.assertEqual(ExposureCompensationConfig().to_dict(), legacy)
        self.assertEqual(list(ExposureCompensationConfig().to_dict()), LEGACY_CONTRACT_KEYS)
        # The R1 arms set only enabled + learning_rate: same keys, same values.
        r1 = ExposureCompensationConfig(**{"enabled": True, "learning_rate": 0.005}).to_dict()
        self.assertEqual(r1, dict(legacy, enabled=True))
        explicit = ExposureCompensationConfig(enabled=True, learning_rate=0.005, mode="per_image").to_dict()
        self.assertEqual(explicit, r1)
        self.assertEqual(
            json.dumps(r1, sort_keys=True), json.dumps(explicit, sort_keys=True)
        )

    def test_curve_contract_only_grows_off_default(self) -> None:
        curve = _curve_config().to_dict()
        self.assertEqual(list(curve)[: len(LEGACY_CONTRACT_KEYS)], LEGACY_CONTRACT_KEYS)
        self.assertEqual(
            list(curve)[len(LEGACY_CONTRACT_KEYS):],
            ["mode", "knot_seconds", "prior_weight", "frozen_curve", "frozen_curve_sha256"],
        )
        self.assertEqual(curve["mode"], "camera_curve")
        self.assertIsNone(curve["frozen_curve"])
        self.assertIsNone(curve["frozen_curve_sha256"])

    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            ExposureCompensationConfig(mode="per_camera").validate()
        with self.assertRaisesRegex(ValueError, "frozen_curve requires mode camera_curve"):
            ExposureCompensationConfig(frozen_curve="x.json").validate()
        with self.assertRaisesRegex(ValueError, "knot_seconds must be positive"):
            _curve_config(knot_seconds=0.0).validate()
        with self.assertRaisesRegex(ValueError, "prior_weight must be non-negative"):
            _curve_config(prior_weight=-1.0).validate()
        with self.assertRaisesRegex(ValueError, "zero_mean_projection is a per_image knob"):
            _curve_config(zero_mean_projection=True).validate()
        with self.assertRaisesRegex(ValueError, "frozen_curve is not a file"):
            _curve_config(frozen_curve="does-not-exist.json").validate()
        _curve_config().validate()


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class ExposureCurveTests(unittest.TestCase):
    def test_evaluates_at_knots_and_between_knots(self) -> None:
        # left at 0, 10, 20 s (knots) and 5 s (midpoint); right at 0 and 15 s.
        ids = ["l0", "l10", "l20", "l5", "r0", "r15"]
        cameras = {"l0": "left", "l10": "left", "l20": "left", "l5": "left", "r0": "right", "r15": "right"}
        times = {"l0": T0, "l10": T0 + 10_000_000_000, "l20": T0 + 20_000_000_000, "l5": T0 + 5_000_000_000,
                 "r0": T0, "r15": T0 + 15_000_000_000}
        curve = ExposureCurve(ids, config=_curve_config(), device="cpu", camera_by_image=cameras, timestamp_ns_by_image=times)
        self.assertEqual(curve.time_origin_ns, T0)
        # left spans 20 s -> 4 knots (0,10,20,30); right spans 15 s -> 3 knots.
        self.assertEqual(curve.camera_slices, {"left": (0, 4), "right": (4, 3)})
        self.assertEqual(int(curve.knot_log_gains.shape[0]), 7)
        with torch.no_grad():
            curve.knot_log_gains.copy_(torch.tensor([0.0, 0.2, -0.4, 0.6, 0.1, -0.1, 0.3]))
        self.assertAlmostEqual(float(curve.gain("l0")), math.exp(0.0), places=6)
        self.assertAlmostEqual(float(curve.gain("l10")), math.exp(0.2), places=6)
        self.assertAlmostEqual(float(curve.gain("l20")), math.exp(-0.4), places=6)
        self.assertAlmostEqual(float(curve.gain("l5")), math.exp(0.1), places=6)
        self.assertAlmostEqual(float(curve.gain("r0")), math.exp(0.1), places=6)
        self.assertAlmostEqual(float(curve.gain("r15")), math.exp(0.1), places=6)  # blend of -0.1 and 0.3
        self.assertAlmostEqual(curve.log_gain_at("right", T0 + 5_000_000_000), 0.0, places=6)
        self.assertAlmostEqual(curve.log_gain_at("right", T0 + 99_000_000_000), 0.3, places=6)
        with self.assertRaises(KeyError):
            curve.gain("unknown")
        # Gradient reaches both taps of an interior image and nothing else.
        grad = torch.autograd.grad(curve.gain("l5"), curve.knot_log_gains)[0]
        self.assertGreater(float(grad[0]), 0.0)
        self.assertGreater(float(grad[1]), 0.0)
        self.assertEqual(float(grad[2]), 0.0)
        self.assertEqual(float(grad[6]), 0.0)
        # Clamp: a knot beyond ln 2 is applied at the bound.
        with torch.no_grad():
            curve.knot_log_gains[0] = 5.0
        self.assertAlmostEqual(float(curve.gain("l0")), 2.0, places=6)

    def test_anchor_keeps_each_camera_mean_at_zero(self) -> None:
        ids, cameras, times = _rig(80)  # 39.5 s: 5 knots per camera (0..40 s)
        curve = ExposureCurve(
            ids, config=_curve_config(mean_anchor_weight=1.0), device="cpu",
            camera_by_image=cameras, timestamp_ns_by_image=times,
        )
        self.assertEqual(curve.camera_slices, {"left": (0, 5), "right": (5, 5)})
        left, right = slice(0, 5), slice(5, 10)
        self.assertAlmostEqual(float(curve.prior_loss().detach()), 0.0, places=7)
        with torch.no_grad():
            curve.knot_log_gains[left] += 0.3
            curve.knot_log_gains[right] -= 0.3
        means = curve.mean_log_gain_by_camera()
        self.assertAlmostEqual(means["left"], 0.3, places=5)
        self.assertAlmostEqual(means["right"], -0.3, places=5)
        # Globally balanced but per-camera wrong: still penalised.
        self.assertGreater(float(curve.prior_loss().detach()), 0.1)
        optimizer = curve.make_optimizer()
        self.assertIsNotNone(optimizer)
        # Give the curves shape so we can see the anchor leaves it alone.
        with torch.no_grad():
            curve.knot_log_gains[left] += torch.tensor([0.05, -0.05, 0.0, 0.05, -0.05])
        before_shape = curve.knot_log_gains.detach()[left] - curve.knot_log_gains.detach()[left].mean()
        for _ in range(3000):
            optimizer.zero_grad()
            curve.prior_loss().backward()
            optimizer.step()
        means = curve.mean_log_gain_by_camera()
        self.assertLess(abs(means["left"]), 1e-3)
        self.assertLess(abs(means["right"]), 1e-3)
        after_shape = curve.knot_log_gains.detach()[left] - curve.knot_log_gains.detach()[left].mean()
        # The anchor is a rank-one pull along the mean; the wiggle survives.
        self.assertGreater(float(after_shape.abs().max()), 0.03)
        self.assertLess(float((after_shape - before_shape).abs().max()), 0.02)

    def test_smoothness_prior_penalises_knot_differences_only(self) -> None:
        ids, cameras, times = _rig(40)
        curve = ExposureCurve(
            ids, config=_curve_config(prior_weight=1.0), device="cpu",
            camera_by_image=cameras, timestamp_ns_by_image=times,
        )
        with torch.no_grad():
            curve.knot_log_gains.fill_(0.25)  # constant curve: smooth, no penalty
        self.assertAlmostEqual(float(curve.prior_loss()), 0.0, places=7)
        with torch.no_grad():
            curve.knot_log_gains[1] = 0.45
        self.assertGreater(float(curve.prior_loss()), 0.0)
        report = curve.report()
        self.assertEqual(report["mode"], "camera_curve")
        self.assertFalse(report["frozen"])
        self.assertEqual(report["image_count"], 40)
        self.assertEqual(report["image_count_by_camera"], {"left": 20, "right": 20})
        self.assertEqual(report["parameter_count"], int(curve.knot_log_gains.shape[0]))

    def test_frozen_curve_reproduces_given_gains(self) -> None:
        ids, cameras, times = _rig(60)  # 30 s
        knots = {
            "left": [0.1, -0.2, 0.3, 0.05, -0.1],
            "right": [-0.15, 0.25, 0.0, 0.2, 0.1],
        }
        payload = build_curve_payload(
            knot_seconds=10.0, time_origin_ns=T0 - 3_000_000_000, cameras=knots, max_abs_log_gain=math.log(2.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene_curve.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = _curve_config(frozen_curve=str(path), prior_weight=1.0, mean_anchor_weight=1.0)
            self.assertTrue(config.is_frozen_curve)
            self.assertEqual(len(config.to_dict()["frozen_curve_sha256"]), 64)
            curve = ExposureCurve(ids, config=config, device="cpu", camera_by_image=cameras, timestamp_ns_by_image=times)
            self.assertTrue(curve.frozen)
            self.assertEqual(curve.time_origin_ns, T0 - 3_000_000_000)  # the file's origin, not the data's
            self.assertFalse(curve.knot_log_gains.requires_grad)
            self.assertIsNone(curve.make_optimizer())
            self.assertEqual(float(curve.prior_loss()), 0.0)
            for image_id in ids:
                expected = evaluate_curve(
                    knots[cameras[image_id]], [(times[image_id] - curve.time_origin_ns) / 1e9], knot_seconds=10.0,
                )[0]
                self.assertAlmostEqual(float(curve.gain(image_id)), math.exp(expected), places=5)
            report = curve.report()
            self.assertTrue(report["frozen"])
            self.assertEqual(report["frozen_curve_sha256"], config.frozen_curve_sha256())
            # The same file produces the same gains in a second "tile" that
            # sees a different subset of images: shared by construction.
            subset = ids[10:25]
            other = ExposureCurve(subset, config=config, device="cpu", camera_by_image=cameras, timestamp_ns_by_image=times)
            for image_id in subset:
                self.assertAlmostEqual(float(other.gain(image_id)), float(curve.gain(image_id)), places=6)
            # Grid mismatch and missing camera are refused.
            with self.assertRaisesRegex(ValueError, "knot_seconds"):
                ExposureCurve(ids, config=_curve_config(frozen_curve=str(path), knot_seconds=5.0), device="cpu",
                              camera_by_image=cameras, timestamp_ns_by_image=times)
            with self.assertRaisesRegex(ValueError, "no camera 'top'"):
                ExposureCurve(["x"], config=config, device="cpu", camera_by_image={"x": "top"}, timestamp_ns_by_image={"x": T0})
            with self.assertRaisesRegex(ValueError, "timestamps missing"):
                ExposureCurve(ids, config=config, device="cpu", camera_by_image=cameras, timestamp_ns_by_image={})

    def test_per_image_compensator_forward_unchanged(self) -> None:
        torch.manual_seed(3)
        render = torch.rand(6, 5, 3)
        legacy = ExposureCompensator(["b", "a", "c"], config=ExposureCompensationConfig(enabled=True), device="cpu",
                                     group_by_image={"a": "left", "b": "right", "c": "left"})
        explicit = ExposureCompensator(["b", "a", "c"], config=ExposureCompensationConfig(enabled=True, mode="per_image"),
                                       device="cpu", group_by_image={"a": "left", "b": "right", "c": "left"})
        self.assertEqual(legacy.index, explicit.index)
        self.assertEqual(legacy.group_members, explicit.group_members)
        with torch.no_grad():
            legacy.log_gains.copy_(torch.tensor([0.1, -0.3, 0.9]))
            explicit.log_gains.copy_(torch.tensor([0.1, -0.3, 0.9]))
        torch.testing.assert_close(legacy.log_gains, explicit.log_gains)
        for image_id, expected in (("a", 0.1), ("b", -0.3), ("c", math.log(2.0))):
            torch.testing.assert_close(render * legacy.gain(image_id), render * math.exp(expected))
            torch.testing.assert_close(render * explicit.gain(image_id), render * legacy.gain(image_id))
        torch.testing.assert_close(legacy.prior_loss(), explicit.prior_loss())
        self.assertEqual(legacy.report(), explicit.report())


@unittest.skipUnless(torch is not None, "torch is an optional training dependency")
class TrainerWiringTests(unittest.TestCase):
    def _config(self, exposure: ExposureCompensationConfig):
        from cloudstudio_3dgs.training.trainer import TrainerConfig

        return TrainerConfig(
            run_id="exposure-curve",
            dataset_manifest=Path("dataset.json"),
            recording_root=Path("recording"),
            mask_manifest=Path("masks.json"),
            mask_root=Path("masks"),
            split_manifest=Path("split.json"),
            initialization_ply=Path("init.ply"),
            output_dir=Path("output"),
            gsplat_lock=Path("lock.json"),
            require_person_masks=False,
            lidar_range_weight=0.0,
            color_model="sh",
            sh_degree=3,
            sh_degree_interval=1000,
            means_lr_final_factor=0.01,
            background_color=(1.0, 1.0, 1.0),
            exposure_compensation=exposure,
        )

    def test_contract_dict_keys(self) -> None:
        per_image = self._config(ExposureCompensationConfig(enabled=True, learning_rate=0.005))
        per_image.validate()
        self.assertEqual(list(per_image.contract_dict()["exposure_compensation"]), LEGACY_CONTRACT_KEYS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.json"
            path.write_text(
                json.dumps(build_curve_payload(knot_seconds=10.0, time_origin_ns=T0, cameras={"left": [0.0, 0.0]}, max_abs_log_gain=0.7)),
                encoding="utf-8",
            )
            curve = self._config(_curve_config(frozen_curve=str(path)))
            curve.validate()
            record = curve.contract_dict()["exposure_compensation"]
            self.assertEqual(record["mode"], "camera_curve")
            self.assertEqual(record["frozen_curve"], str(path))
            self.assertEqual(len(record["frozen_curve_sha256"]), 64)
            # Warm-start fresh names follow the mode.
            from cloudstudio_3dgs.training.trainer import TrainerConfig

            warm = Path(tmp) / "warm.pt"
            warm.write_bytes(b"")
            values = dict(curve.__dict__)
            values.update(warm_start_checkpoint=warm, warm_start_fresh_auxiliary=("exposure_log_gains",))
            with self.assertRaisesRegex(ValueError, "per_image exposure"):
                TrainerConfig(**values).validate()
            values.update(warm_start_fresh_auxiliary=("exposure_curve_knots",))
            with self.assertRaisesRegex(ValueError, "learnable camera_curve"):
                TrainerConfig(**values).validate()
            learnable = dict(self._config(_curve_config()).__dict__)
            learnable.update(warm_start_checkpoint=warm, warm_start_fresh_auxiliary=("exposure_curve_knots",))
            TrainerConfig(**learnable).validate()

    def test_trainer_source_builds_the_curve_and_registers_knots(self) -> None:
        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(
                r"if config\.exposure_compensation\.enabled and config\.exposure_compensation\.is_curve:.*?"
                r"exposure = ExposureCurve\(.*?timestamp_ns_by_image=timestamp_ns_by_image,.*?"
                r'auxiliary_params\["exposure_curve_knots"\] = exposure\.knot_log_gains\s*\n'
                r"\s*if exposure_optimizer is not None:\s*\n"
                r'\s*auxiliary_optimizers\["exposure_curve_knots"\] = exposure_optimizer\s*\n'
                r"\s*elif config\.exposure_compensation\.enabled:.*?"
                r"exposure = ExposureCompensator\(",
                re.DOTALL,
            ),
            "camera_curve must be the explicit alternative to the per_image compensator",
        )
        self.assertIn('auxiliary_params["exposure_log_gains"] = exposure.log_gains', source)
        self.assertIn("rgb_gain=None\n            if exposure is None\n            else exposure.gain(sample.image_id.split(\"::\")[0])", source)


if __name__ == "__main__":
    unittest.main()
