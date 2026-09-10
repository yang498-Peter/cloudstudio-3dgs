"""RenderSpec: one render description shared by every evaluator.

Torch-free. The backend plumbing test substitutes a stub GsplatBackend and a
stub ``torch`` module, so it checks exactly what ``_load_backend`` sets on the
object it returns and nothing about CUDA.
"""

from __future__ import annotations

import json
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.render_spec import (  # noqa: E402
    PROPAGATED,
    UNPROPAGATED,
    RenderSpec,
    SpecField,
    model_sh_degree_from_params,
    resolve_pinhole_render_mode,
)

REPO_CONFIG = ROOT / "run_configs" / "house0305_tiles" / "v9" / "tile0_R1_range0_20k.json"

EVALUATORS = (
    "tools/sharpness_metrics.py",
    "tools/evaluate_probe_views.py",
    "tools/build_three_way_compare.py",
    "tools/build_offtrajectory_compare.py",
    "tools/roundtrip_checkpoint_ply.py",
)


def _params(degree: int, count: int = 5) -> dict:
    bands = (degree + 1) ** 2
    return {
        "means": np.zeros((count, 3), np.float32),
        "sh0": np.zeros((count, 1, 3), np.float32),
        "shN": np.zeros((count, bands - 1, 3), np.float32),
    }


def _config(**overrides) -> dict:
    base = {
        "device": "cuda:0",
        "cap_max": 1000,
        "gsplat_lock": "lock.json",
        "factor": 1,
        "sh_degree": 1,
        "color_model": "sh",
        "background_color": [1.0, 1.0, 1.0],
        "face_cache_manifest": "C:/faces/face4_train/face_manifest.json",
        "dataset_manifest": "C:/data/dataset_manifest.json",
        "exposure_compensation": {"enabled": True, "learning_rate": 0.005},
    }
    base.update(overrides)
    return base


class RenderSpecResolutionTest(unittest.TestCase):
    def test_pinhole_defaults_resolve_classic_without_ut(self):
        spec = RenderSpec.from_config_and_params(_config(), _params(1))
        self.assertEqual(spec.camera_model.value, "pinhole")
        self.assertEqual(spec.rasterize_mode.value, {"applied": "classic", "trained": "classic"})
        self.assertEqual(spec.with_ut.value, {"applied": False, "trained": False})
        self.assertFalse(spec.with_eval3d.value)
        self.assertEqual(spec.rasterize_mode.status, PROPAGATED)
        self.assertEqual(spec.sh_degree.value["model"], 1)
        self.assertEqual(spec.sh_degree.value["rendered"], 1)
        self.assertEqual(spec.sh_degree.status, PROPAGATED)
        self.assertEqual(spec.alpha_rule.value["global_z_order"], True)
        self.assertEqual(spec.alpha_rule.value["range_semantics"], "pinhole_z_depth_m")
        self.assertEqual(spec.distortion.value, "none")
        self.assertEqual(spec.coordinate_frame.value["face_rotation"], True)

    def test_always_unpropagated_fields_are_marked(self):
        spec = RenderSpec.from_config_and_params(_config(), _params(1))
        for name in ("pixel_centre_convention", "colour_space", "clipping_near_far"):
            self.assertEqual(getattr(spec, name).status, UNPROPAGATED, name)
        self.assertIn("pixel_centre_convention", spec.unpropagated())
        self.assertEqual(spec.clipping_near_far.value, [0.01, 1e10])

    def test_antialiased_config_is_recorded_but_not_applied_by_default(self):
        config = _config(pinhole_rasterize_mode="antialiased")
        spec = RenderSpec.from_config_and_params(config, _params(1))
        self.assertEqual(spec.rasterize_mode.value, {"applied": "classic", "trained": "antialiased"})
        self.assertEqual(spec.rasterize_mode.status, UNPROPAGATED)
        honoured = RenderSpec.from_config_and_params(config, _params(1), honour_render_mode=True)
        self.assertEqual(honoured.rasterize_mode.value, {"applied": "antialiased", "trained": "antialiased"})
        self.assertEqual(honoured.rasterize_mode.status, PROPAGATED)
        self.assertNotEqual(spec.fingerprint(), honoured.fingerprint())

    def test_pinhole_with_ut_config_drives_eval3d_and_range_semantics(self):
        config = _config(pinhole_with_ut=True)
        default = RenderSpec.from_config_and_params(config, _params(1))
        self.assertEqual(default.with_ut.value, {"applied": False, "trained": True})
        self.assertEqual(default.with_ut.status, UNPROPAGATED)
        honoured = RenderSpec.from_config_and_params(config, _params(1), honour_render_mode=True)
        self.assertTrue(honoured.with_eval3d.value)
        self.assertEqual(honoured.alpha_rule.value["range_semantics"], "euclidean_ray_range_m")

    def test_invalid_pinhole_modes_are_rejected_like_the_trainer(self):
        with self.assertRaises(ValueError):
            resolve_pinhole_render_mode({"pinhole_rasterize_mode": "mip"})
        with self.assertRaises(ValueError):
            resolve_pinhole_render_mode(
                {"pinhole_rasterize_mode": "antialiased", "pinhole_with_ut": True}
            )

    def test_fisheye_forces_ut_and_classic(self):
        config = _config(pinhole_rasterize_mode="antialiased")
        spec = RenderSpec.from_config_and_params(config, _params(1), camera_model="fisheye")
        self.assertEqual(spec.rasterize_mode.value, {"applied": "classic", "trained": "classic"})
        self.assertEqual(spec.with_ut.value, {"applied": True, "trained": True})
        self.assertTrue(spec.with_eval3d.value)
        self.assertEqual(spec.distortion.value, "OPENCV_FISHEYE_k1_k4")
        self.assertEqual(spec.alpha_rule.value["global_z_order"], False)
        self.assertEqual(spec.intrinsics_source.value["kind"], "dataset_manifest_intrinsic_over_factor")
        self.assertEqual(spec.coordinate_frame.value["face_rotation"], False)

    def test_camera_model_defaults_to_trainer_rule(self):
        no_faces = _config()
        del no_faces["face_cache_manifest"]
        spec = RenderSpec.from_config_and_params(no_faces, _params(1))
        self.assertEqual(spec.camera_model.value, "fisheye")

    def test_sh_degree_is_model_derived(self):
        self.assertEqual(model_sh_degree_from_params(_params(0)), 0)
        self.assertEqual(model_sh_degree_from_params(_params(1)), 1)
        self.assertEqual(model_sh_degree_from_params(_params(3)), 3)
        self.assertIsNone(model_sh_degree_from_params(None))
        self.assertIsNone(model_sh_degree_from_params({"colors": np.zeros((2, 3))}))
        # An eval config that says 0 does not change what the model carries.
        spec = RenderSpec.from_config_and_params(_config(sh_degree=0), _params(1), applied_sh_degree=1)
        self.assertEqual(spec.sh_degree.value, {"rendered": 1, "model": 1, "config": 0, "color_model": "sh"})
        self.assertEqual(spec.sh_degree.status, PROPAGATED)
        # Rendering below the model's degree is a different render and says so.
        clamped = RenderSpec.from_config_and_params(_config(sh_degree=0), _params(1), applied_sh_degree=0)
        self.assertEqual(clamped.sh_degree.status, UNPROPAGATED)
        self.assertNotEqual(spec.fingerprint(), clamped.fingerprint())

    def test_sh_degree_without_params_is_unpropagated(self):
        spec = RenderSpec.from_config_and_params(_config(), None, applied_sh_degree=1)
        self.assertEqual(spec.sh_degree.status, UNPROPAGATED)
        self.assertIsNone(spec.sh_degree.value["model"])

    def test_exposure_policy_resolution(self):
        enabled = _config()
        disabled = _config(exposure_compensation={"enabled": False})
        single_tile_meta = {"step": 20000, "identity": {}}
        merged_meta = {
            "step": 20000,
            "merge": {
                "exposure_harmonized": True,
                "records": [{"tile_id": 0, "exposure_gain_applied": 0.9},
                            {"tile_id": 1, "exposure_gain_applied": 0.95}],
            },
        }
        unharmonized_meta = {"step": 20000, "merge": {"exposure_harmonized": False, "records": []}}

        none = RenderSpec.from_config_and_params(disabled, _params(1), checkpoint_meta=single_tile_meta)
        self.assertEqual(none.exposure_policy.value["policy"], "none")
        self.assertEqual(none.exposure_policy.status, PROPAGATED)

        canonical = RenderSpec.from_config_and_params(enabled, _params(1), checkpoint_meta=single_tile_meta)
        self.assertEqual(canonical.exposure_policy.value["policy"], "canonical")
        self.assertEqual(canonical.exposure_policy.status, PROPAGATED)
        self.assertTrue(canonical.exposure_policy.value["training"]["enabled"])

        unknown = RenderSpec.from_config_and_params(enabled, _params(1), checkpoint_meta=None)
        self.assertEqual(unknown.exposure_policy.value["policy"], "canonical")
        self.assertEqual(unknown.exposure_policy.status, UNPROPAGATED)

        baked = RenderSpec.from_config_and_params(enabled, _params(1), checkpoint_meta=merged_meta)
        self.assertEqual(baked.exposure_policy.value["policy"], "per_tile_gain")
        self.assertEqual(baked.exposure_policy.value["baked_tile_gains"], [0.9, 0.95])
        self.assertEqual(baked.exposure_policy.status, PROPAGATED)

        plain_merge = RenderSpec.from_config_and_params(enabled, _params(1), checkpoint_meta=unharmonized_meta)
        self.assertEqual(plain_merge.exposure_policy.value["policy"], "canonical")

        fingerprints = {s.fingerprint() for s in (none, canonical, unknown, baked)}
        self.assertEqual(len(fingerprints), 4)

    def test_background_policy_follows_manifest_unless_overridden(self):
        with_manifest = _config(background_image_manifest="C:/bg/manifest.json")
        spec = RenderSpec.from_config_and_params(with_manifest, _params(1))
        self.assertEqual(spec.background.value["policy"], "view_background_library")
        self.assertEqual(spec.background.value["manifest"], "C:/bg/manifest.json")
        constant = RenderSpec.from_config_and_params(
            with_manifest, _params(1), background_policy="constant", background_rgb=(0.0, 0.0, 0.0)
        )
        self.assertEqual(constant.background.value, {"policy": "constant", "constant_rgb": [0.0, 0.0, 0.0], "manifest": None})
        with self.assertRaises(ValueError):
            RenderSpec.from_config_and_params(with_manifest, _params(1), background_policy="sky")

    def test_real_tile0_config_resolves(self):
        config = json.loads(REPO_CONFIG.read_text(encoding="utf-8"))
        spec = RenderSpec.from_config_and_params(config, _params(1), checkpoint_meta={"step": 20000})
        self.assertEqual(spec.rasterize_mode.value["applied"], "classic")
        self.assertEqual(spec.with_ut.value["applied"], False)
        self.assertEqual(spec.exposure_policy.value["policy"], "canonical")
        self.assertEqual(spec.background.value["policy"], "view_background_library")
        self.assertEqual(spec.resolution.value, {"factor": 1, "tile_crops": True})
        self.assertEqual(spec.sh_degree.value["config"], 1)
        self.assertEqual(
            sorted(spec.unpropagated()),
            ["clipping_near_far", "colour_space", "pixel_centre_convention"],
        )


class RenderSpecFingerprintTest(unittest.TestCase):
    def test_fingerprint_is_stable_and_pure(self):
        a = RenderSpec.from_config_and_params(_config(), _params(1), checkpoint_meta={"step": 1})
        b = RenderSpec.from_config_and_params(_config(), _params(1), checkpoint_meta={"step": 1})
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.fingerprint(), a.fingerprint())
        self.assertRegex(a.fingerprint(), r"^[0-9a-f]{64}$")
        # Params of the same degree but different count leave the spec alone.
        c = RenderSpec.from_config_and_params(_config(), _params(1, count=50), checkpoint_meta={"step": 1})
        self.assertEqual(a.fingerprint(), c.fingerprint())
        # Documentation (source / note) is not part of comparability.
        documented = a.with_field(
            "alpha_rule", SpecField(a.alpha_rule.value, source="reworded", status=a.alpha_rule.status)
        )
        self.assertEqual(a.fingerprint(), documented.fingerprint())

    def test_fingerprint_is_sensitive_to_every_field(self):
        base = RenderSpec.from_config_and_params(_config(), _params(1), checkpoint_meta={"step": 1})
        seen = {base.fingerprint()}
        for name in RenderSpec.field_names():
            original = getattr(base, name)
            changed = base.with_field(
                name, SpecField({"changed": name}, source=original.source, status=original.status)
            )
            digest = changed.fingerprint()
            self.assertNotIn(digest, seen, f"{name} did not move the fingerprint")
            seen.add(digest)
            # A status flip alone (same value) is also a different protocol.
            flipped = base.with_field(
                name,
                SpecField(
                    original.value,
                    source=original.source,
                    status=UNPROPAGATED if original.status == PROPAGATED else PROPAGATED,
                ),
            )
            self.assertNotEqual(flipped.fingerprint(), base.fingerprint(), name)

    def test_record_is_json_serialisable_and_carries_fingerprint(self):
        spec = RenderSpec.from_config_and_params(_config(), _params(1), checkpoint_meta={"step": 1})
        record = spec.record()
        text = json.dumps(record)
        self.assertEqual(json.loads(text)["fingerprint"], spec.fingerprint())
        self.assertEqual(set(record["fields"]), set(RenderSpec.field_names()))
        self.assertEqual(record["schema_version"], RenderSpec.SCHEMA_VERSION)
        self.assertEqual(record["unpropagated"], spec.unpropagated())

    def test_with_field_rejects_unknown_names(self):
        spec = RenderSpec.from_config_and_params(_config(), _params(1))
        with self.assertRaises(KeyError):
            spec.with_field("view_set", SpecField("x", source="y"))


class _StubBackend:
    """Records constructor kwargs; carries none of GsplatBackend's defaults so
    every attribute the test reads must have been set by _load_backend."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)


class LoadBackendPlumbingTest(unittest.TestCase):
    def _load(self, config, **kwargs):
        stub_backend_module = types.ModuleType("cloudstudio_3dgs.training.backend")
        stub_backend_module.GsplatBackend = _StubBackend
        stub_torch = types.ModuleType("torch")
        _StubBackend.instances = []
        with mock.patch.dict(
            sys.modules,
            {"torch": stub_torch, "cloudstudio_3dgs.training.backend": stub_backend_module},
        ):
            from tools.sharpness_metrics import _load_backend

            with mock.patch("sys.stderr") as stderr:
                backend, torch_module = _load_backend(config, **kwargs)
        self.assertIs(torch_module, stub_torch)
        self.assertIs(backend, _StubBackend.instances[0])
        return backend, stderr

    def test_defaults_match_trainer_defaults(self):
        backend, stderr = self._load(_config())
        self.assertEqual(backend.kwargs["cap_max"], 1000)
        self.assertEqual(backend.kwargs["device"], "cuda:0")
        self.assertEqual(backend.kwargs["mcmc_config"], {"noise_injection_stop_iter": 0})
        self.assertEqual(backend.color_model, "sh")
        self.assertEqual(backend.sh_degree, 1)
        self.assertEqual(backend.pinhole_rasterize_mode, "classic")
        self.assertIs(backend.pinhole_with_ut, False)
        self.assertIs(backend.honour_render_mode, False)
        self.assertEqual(backend.render_spec.rasterize_mode.status, PROPAGATED)
        stderr.write.assert_not_called()

    def test_sh_degree_argument_overrides_config(self):
        backend, _ = self._load(_config(sh_degree=0), sh_degree=3)
        self.assertEqual(backend.sh_degree, 3)
        self.assertEqual(backend.render_spec.sh_degree.value["rendered"], 3)
        self.assertEqual(backend.render_spec.sh_degree.value["config"], 0)

    def test_non_classic_config_is_kept_classic_and_logged(self):
        backend, stderr = self._load(_config(pinhole_rasterize_mode="antialiased"))
        self.assertEqual(backend.pinhole_rasterize_mode, "classic")
        self.assertIs(backend.pinhole_with_ut, False)
        self.assertEqual(backend.render_spec.rasterize_mode.value["trained"], "antialiased")
        self.assertEqual(backend.render_spec.rasterize_mode.status, UNPROPAGATED)
        logged = "".join(str(call.args[0]) for call in stderr.write.call_args_list)
        self.assertIn("antialiased", logged)
        self.assertIn("--honour-render-mode", logged)

    def test_honour_render_mode_applies_config_like_the_trainer(self):
        backend, stderr = self._load(
            _config(pinhole_rasterize_mode="antialiased"), honour_render_mode=True
        )
        self.assertEqual(backend.pinhole_rasterize_mode, "antialiased")
        self.assertIs(backend.pinhole_with_ut, False)
        self.assertIs(backend.honour_render_mode, True)
        self.assertEqual(backend.render_spec.rasterize_mode.status, PROPAGATED)
        stderr.write.assert_not_called()
        with_ut, _ = self._load(_config(pinhole_with_ut=True), honour_render_mode=True)
        self.assertIs(with_ut.pinhole_with_ut, True)
        self.assertEqual(with_ut.pinhole_rasterize_mode, "classic")

    def test_resolve_render_spec_reads_applied_state_off_the_backend(self):
        backend, _ = self._load(_config(pinhole_rasterize_mode="antialiased"))
        backend.sh_degree = 1
        from tools.sharpness_metrics import checkpoint_meta, resolve_render_spec

        payload = {"step": 7, "params": {"means": "tensor"}, "optimizers": {}, "identity": {"a": 1}}
        meta = checkpoint_meta(payload)
        self.assertEqual(meta, {"step": 7, "identity": {"a": 1}})
        spec = resolve_render_spec(
            _config(pinhole_rasterize_mode="antialiased"), _params(1), backend,
            camera_model="pinhole", background_policy="constant",
            background_rgb=(1.0, 1.0, 1.0), tile_crops=False, checkpoint_meta=meta,
        )
        self.assertEqual(spec.rasterize_mode.value, {"applied": "classic", "trained": "antialiased"})
        self.assertEqual(spec.exposure_policy.value["policy"], "canonical")
        self.assertEqual(spec.exposure_policy.status, PROPAGATED)


class EvaluatorSchemaTest(unittest.TestCase):
    """Every evaluator writes render_spec next to its scores and exposes the
    render-mode flag; a new evaluator that forgets either fails here."""

    def test_every_evaluator_records_render_spec(self):
        for relative in EVALUATORS:
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertRegex(
                source, r'"render_spec":\s*render_spec\.record\(\)',
                f"{relative} does not write render_spec into its output",
            )
            self.assertIn("--honour-render-mode", source, relative)
            self.assertIn("resolve_render_spec", source, relative)

    def test_load_backend_callers_pass_the_flag(self):
        for relative in EVALUATORS:
            source = (ROOT / relative).read_text(encoding="utf-8")
            calls = re.findall(r"_load_backend\(([^)]*)\)", source)
            calls = [c for c in calls if "config: dict" not in c]
            self.assertTrue(calls, relative)
            for call in calls:
                self.assertIn("honour_render_mode", call, f"{relative}: {call}")


if __name__ == "__main__":
    unittest.main()
