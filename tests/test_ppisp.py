# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tests for the pure-PyTorch PPISP port (cloudstudio_3dgs/training/ppisp.py),
# ported from nv-tlabs/ppisp commit df33809f7b3b20ac06de088dfc871b144b8fb54d.

from __future__ import annotations

import unittest

import torch

from cloudstudio_3dgs.training.ppisp import (
    COLOR_PARAMS,
    CRF_PARAMS_PER_CHANNEL,
    VIGNETTING_PARAMS_PER_CHANNEL,
    PpispConfig,
    PpispCorrector,
)

_IMAGE_IDS = ["left_000", "left_001", "right_000", "right_001"]
_CAMERA_BY_IMAGE = {
    "left_000": "left",
    "left_001": "left",
    "right_000": "right",
    "right_001": "right",
}


def _make_corrector(**config_overrides) -> PpispCorrector:
    config = PpispConfig(enabled=True, **config_overrides)
    return PpispCorrector(
        _IMAGE_IDS,
        config=config,
        device="cpu",
        camera_by_image=_CAMERA_BY_IMAGE,
    )


def _make_rgb(height: int = 8, width: int = 12, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(height, width, 3, generator=generator) * 0.8 + 0.1


class PpispConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_no_crf_per_camera(self) -> None:
        config = PpispConfig()
        config.validate()
        self.assertFalse(config.enabled)
        self.assertEqual(config.param_type, "no_crf")
        self.assertEqual(config.mode, "per_camera")

    def test_validate_rejects_illegal_values(self) -> None:
        with self.assertRaises(ValueError):
            PpispConfig(param_type="crf_only").validate()
        with self.assertRaises(ValueError):
            PpispConfig(mode="per_pixel").validate()
        with self.assertRaises(ValueError):
            PpispConfig(learning_rate=0.0).validate()
        with self.assertRaises(ValueError):
            PpispConfig(learning_rate=-1e-3).validate()
        with self.assertRaises(ValueError):
            PpispConfig(vig_center_weight=-0.1).validate()
        with self.assertRaises(ValueError):
            PpispConfig(color_mean_weight=-1.0).validate()

    def test_to_dict_round_trips(self) -> None:
        config = PpispConfig(enabled=True, param_type="crf", learning_rate=1e-2)
        rebuilt = PpispConfig(**config.to_dict())
        self.assertEqual(config, rebuilt)

    def test_disabled_config_rejects_construction(self) -> None:
        with self.assertRaises(ValueError):
            PpispCorrector(
                _IMAGE_IDS,
                config=PpispConfig(enabled=False),
                device="cpu",
                camera_by_image=_CAMERA_BY_IMAGE,
            )

    def test_missing_camera_mapping_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PpispCorrector(
                _IMAGE_IDS,
                config=PpispConfig(enabled=True),
                device="cpu",
                camera_by_image={"left_000": "left"},
            )


class PpispIdentityTest(unittest.TestCase):
    def test_zero_params_no_crf_is_identity(self) -> None:
        corrector = _make_corrector()
        rgb = _make_rgb(seed=1)
        out = corrector.apply(rgb, "left_000")
        # Not bit-exact: upstream renormalizes intensity with a 1e-5 epsilon
        # and the homography with 1e-10, so identity holds to ~1e-4.
        self.assertTrue(torch.allclose(out, rgb, atol=1e-4))

    def test_default_init_crf_is_identity(self) -> None:
        corrector = _make_corrector(param_type="crf")
        rgb = _make_rgb(seed=2)
        out = corrector.apply(rgb, "right_001")
        self.assertTrue(torch.allclose(out, rgb, atol=1e-4))

    def test_parameter_count_no_crf_is_24_per_camera(self) -> None:
        corrector = _make_corrector()
        expected_per_camera = 1 + 3 * VIGNETTING_PARAMS_PER_CHANNEL + COLOR_PARAMS
        self.assertEqual(expected_per_camera, 24)
        self.assertEqual(
            corrector.report()["parameter_count"], 2 * expected_per_camera
        )

    def test_parameter_count_crf_adds_12_per_camera(self) -> None:
        corrector = _make_corrector(param_type="crf")
        expected_per_camera = (
            1 + 3 * VIGNETTING_PARAMS_PER_CHANNEL + COLOR_PARAMS
            + 3 * CRF_PARAMS_PER_CHANNEL
        )
        self.assertEqual(expected_per_camera, 36)
        self.assertEqual(
            corrector.report()["parameter_count"], 2 * expected_per_camera
        )


class PpispExposureTest(unittest.TestCase):
    def test_exposure_scales_by_exp2(self) -> None:
        corrector = _make_corrector()
        rgb = _make_rgb(seed=3)
        with torch.no_grad():
            slot = corrector.frame_index["left"]
            corrector.exposure_params[slot] = 1.0  # 2^1
        out = corrector.apply(rgb, "left_000")
        self.assertTrue(torch.allclose(out, rgb * 2.0, atol=1e-3))

    def test_exposure_is_differentiable_with_nonzero_grad(self) -> None:
        corrector = _make_corrector()
        rgb = _make_rgb(seed=4)
        out = corrector.apply(rgb, "left_000")
        loss = (out - rgb * 1.5).abs().mean()
        loss.backward()
        grad = corrector.exposure_params.grad
        self.assertIsNotNone(grad)
        slot = corrector.frame_index["left"]
        self.assertNotEqual(float(grad[slot]), 0.0)
        self.assertTrue(torch.isfinite(grad).all())
        # The other camera's slot never participated in this image.
        other = corrector.frame_index["right"]
        self.assertEqual(float(grad[other]), 0.0)

    def test_all_parameters_receive_finite_gradients(self) -> None:
        corrector = _make_corrector(param_type="crf")
        rgb = _make_rgb(seed=5)
        with torch.no_grad():
            for param in corrector.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        out = corrector.apply(rgb, "left_000")
        out.mean().backward()
        for param in corrector.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()), 0.0)


class PpispVignettingTest(unittest.TestCase):
    def test_negative_alphas_darken_corners_only(self) -> None:
        corrector = _make_corrector()
        rgb = torch.full((9, 9, 3), 0.5)
        camera_slot = corrector.camera_index["left"]
        with torch.no_grad():
            corrector.vignetting_params[camera_slot, :, 2] = -1.0  # a0 * r^2
        out = corrector.apply(rgb, "left_000")
        center = out[4, 4]
        corner = out[0, 0]
        self.assertTrue(torch.all(corner < center))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= rgb + 1e-4))

    def test_falloff_clamp_lower_bound_prevents_negative_rgb(self) -> None:
        corrector = _make_corrector()
        rgb = torch.full((16, 16, 3), 0.5)
        camera_slot = corrector.camera_index["left"]
        with torch.no_grad():
            # Massive negative polynomial: unclamped falloff would go far
            # below zero at the corners and flip the sign of the image.
            corrector.vignetting_params[camera_slot, :, 2:] = -1000.0
        out = corrector.apply(rgb, "left_000")
        self.assertTrue(torch.all(out >= 0.0))

    def test_falloff_clamp_upper_bound_prevents_brightening(self) -> None:
        corrector = _make_corrector()
        rgb = torch.full((16, 16, 3), 0.5)
        camera_slot = corrector.camera_index["left"]
        with torch.no_grad():
            # Positive alphas would brighten the corners; clamp caps at 1.
            corrector.vignetting_params[camera_slot, :, 2:] = 1000.0
        out = corrector.apply(rgb, "left_000")
        self.assertTrue(torch.allclose(out, rgb, atol=1e-4))

    def test_crop_pixel_coords_match_full_frame_subwindow(self) -> None:
        corrector = _make_corrector()
        camera_slot = corrector.camera_index["left"]
        with torch.no_grad():
            corrector.vignetting_params[camera_slot, :, :2] = 0.05
            corrector.vignetting_params[camera_slot, :, 2] = -0.8
        full = torch.full((12, 16, 3), 0.5)
        out_full = corrector.apply(full, "left_000")
        # Render the same sensor window as a crop with explicit coordinates.
        ys = torch.arange(4, 10, dtype=torch.float32) + 0.5
        xs = torch.arange(6, 14, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1)
        crop = full[4:10, 6:14]
        out_crop = corrector.apply(
            crop, "left_000", pixel_coords=coords, resolution=(16, 12)
        )
        self.assertTrue(torch.allclose(out_crop, out_full[4:10, 6:14], atol=1e-5))


class PpispGroupingTest(unittest.TestCase):
    def test_per_camera_groups_are_independent(self) -> None:
        corrector = _make_corrector()
        rgb = _make_rgb(seed=6)
        baseline_right = corrector.apply(rgb, "right_000").detach()
        left_slot = corrector.camera_index["left"]
        with torch.no_grad():
            corrector.exposure_params[corrector.frame_index["left"]] = 0.7
            corrector.vignetting_params[left_slot, :, 2] = -0.5
            corrector.color_params[corrector.frame_index["left"], 0] = 0.4
        changed_left = corrector.apply(rgb, "left_001").detach()
        unchanged_right = corrector.apply(rgb, "right_000").detach()
        self.assertFalse(torch.allclose(changed_left, rgb, atol=1e-3))
        self.assertTrue(torch.allclose(unchanged_right, baseline_right, atol=1e-6))

    def test_per_camera_images_share_parameters(self) -> None:
        corrector = _make_corrector()
        rgb = _make_rgb(seed=7)
        with torch.no_grad():
            corrector.exposure_params[corrector.frame_index["left"]] = 0.3
        out_a = corrector.apply(rgb, "left_000")
        out_b = corrector.apply(rgb, "left_001")
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-7))

    def test_per_image_mode_gives_each_image_its_own_slot(self) -> None:
        corrector = _make_corrector(mode="per_image")
        self.assertEqual(corrector.exposure_params.shape[0], len(_IMAGE_IDS))
        rgb = _make_rgb(seed=8)
        with torch.no_grad():
            corrector.exposure_params[corrector.frame_index["left_000"]] = 1.0
        out_a = corrector.apply(rgb, "left_000")
        out_b = corrector.apply(rgb, "left_001")
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-3))
        self.assertTrue(torch.allclose(out_b, rgb, atol=1e-4))

    def test_unknown_image_id_raises(self) -> None:
        corrector = _make_corrector()
        with self.assertRaises(KeyError):
            corrector.apply(_make_rgb(), "mystery_cam_007")


class PpispRegularizationTest(unittest.TestCase):
    def test_zero_params_have_zero_regularization(self) -> None:
        for param_type in ("no_crf", "crf"):
            corrector = _make_corrector(param_type=param_type)
            loss = corrector.regularization_loss()
            # CRF init is identical across channels, so its variance term is
            # exactly zero as well.
            self.assertEqual(float(loss.detach()), 0.0)

    def test_offset_params_have_positive_regularization(self) -> None:
        cases = {
            "exposure": lambda c: c.exposure_params.add_(0.5),
            "vig_center": lambda c: c.vignetting_params[:, :, :2].add_(0.2),
            "vig_positive_alpha": lambda c: c.vignetting_params[:, :, 2:].add_(0.3),
            "color": lambda c: c.color_params.add_(1.0),
        }
        for name, mutate in cases.items():
            corrector = _make_corrector()
            with torch.no_grad():
                mutate(corrector)
            loss = corrector.regularization_loss()
            self.assertGreater(float(loss.detach()), 0.0, msg=name)

    def test_crf_channel_variance_penalized_only_with_crf(self) -> None:
        corrector = _make_corrector(param_type="crf")
        with torch.no_grad():
            corrector.crf_params[:, 0, :].add_(0.5)  # de-synchronize channels
        self.assertGreater(float(corrector.regularization_loss().detach()), 0.0)

    def test_exposure_mean_anchor_ignores_balanced_gains(self) -> None:
        # Same anchoring family as the scalar exposure module's zero-mean
        # projection: only the MEAN log gain is pulled to zero, per-group
        # differences stay free.
        corrector = _make_corrector(
            vig_center_weight=0.0,
            vig_channel_weight=0.0,
            vig_non_pos_weight=0.0,
            color_mean_weight=0.0,
        )
        with torch.no_grad():
            corrector.exposure_params.copy_(torch.tensor([0.4, -0.4]))
        balanced = float(corrector.regularization_loss().detach())
        with torch.no_grad():
            corrector.exposure_params.copy_(torch.tensor([0.4, 0.4]))
        shifted = float(corrector.regularization_loss().detach())
        self.assertAlmostEqual(balanced, 0.0, places=7)
        self.assertGreater(shifted, 0.0)

    def test_negative_alphas_are_not_penalized_by_non_pos_term(self) -> None:
        corrector = _make_corrector(
            exposure_mean_weight=0.0,
            vig_center_weight=0.0,
            vig_channel_weight=0.0,
            color_mean_weight=0.0,
        )
        with torch.no_grad():
            corrector.vignetting_params[:, :, 2:] = -0.5
        self.assertEqual(float(corrector.regularization_loss().detach()), 0.0)

    def test_regularization_is_differentiable(self) -> None:
        corrector = _make_corrector(param_type="crf")
        with torch.no_grad():
            for param in corrector.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        loss = corrector.regularization_loss()
        loss.backward()
        for param in corrector.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())


class PpispOptimizerAndReportTest(unittest.TestCase):
    def test_make_optimizer_covers_all_parameters(self) -> None:
        corrector = _make_corrector(param_type="crf")
        optimizer = corrector.make_optimizer()
        covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for param in corrector.parameters():
            self.assertIn(id(param), covered)

    def test_optimizer_step_reduces_regularization(self) -> None:
        corrector = _make_corrector()
        with torch.no_grad():
            corrector.exposure_params.fill_(0.5)
        optimizer = corrector.make_optimizer()
        initial = float(corrector.regularization_loss().detach())
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            loss = corrector.regularization_loss()
            loss.backward()
            optimizer.step()
        final = float(corrector.regularization_loss().detach())
        self.assertLess(final, initial)

    def test_report_contains_expected_keys(self) -> None:
        corrector = _make_corrector(param_type="crf")
        report = corrector.report()
        for key in (
            "mode",
            "param_type",
            "camera_count",
            "parameter_count",
            "exposure_log2_mean",
            "exposure_gain_min",
            "exposure_gain_max",
            "vig_center_offset_max",
            "vig_positive_alpha_fraction",
            "color_offset_abs_max",
            "crf_raw_abs_max",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["camera_count"], 2)


if __name__ == "__main__":
    unittest.main()
