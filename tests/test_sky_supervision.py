"""sky_supervision: sky pixels belong to the backdrop (research/quality_recovery_v2/12).

Pins:
* the default path is byte-identical: no sky tensor, the loss equals the
  inline oracle, the contract has no sky key and equals an explicit
  ``enabled: false`` config;
* the effective sky mask (erosion + no-LiDAR-within-window + renderer mask)
  agrees between the torch and numpy twins and a plain-loop oracle;
* the sky alpha term equals ``mean |alpha - target|`` over the effective sky
  pixels and its gradient pushes alpha (and the opacity of a gaussian
  covering a sky pixel) down, while a gaussian covering a non-sky pixel gets
  no gradient from it;
* with ``exclude_photometric`` the RGB terms see no sky pixel (zero gradient
  there, mean over the remaining pixels); with ``exclude_mono_depth`` the DA2
  term ignores sky pixels;
* the dataset binds the signed manifest to the Face4 cache, crops the mask
  with the Tile view, and fails closed on a missing record / file / wrong
  binding / tampered signature;
* the contract gains keys only when enabled, validation refuses a missing
  manifest (naming it), a mismatched Face4 binding and the growth block
  outside the classic exact lifecycle, and the adaptive-growth gate refuses
  an enabled arm;
* the growth block counts sky observations per gaussian, blocks parents seen
  in sky in >= 50% of their observations, rides the topology ops and is
  zeroed at the refine event.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.sky_masks import build_sky_mask_manifest, sky_mask_path_for
from cloudstudio_3dgs.pipeline.mipmap_gate import (
    GATE_PROFILE,
    GATE_SCHEMA_VERSION,
    ORDERED_STAGES,
    UPSTREAM_DATA_READY_STATUS,
    advance_adaptive_growth_gate,
    sign_gate,
)
from cloudstudio_3dgs.training.sky_supervision import (
    SKY_BLOCKED_TOTAL_KEY,
    SKY_HIT_KEY,
    SKY_MASK_INFO_KEY,
    SKY_MASK_KIND,
    SKY_SEEN_KEY,
    SkyGrowthBlock,
    SkySupervisionConfig,
    sign_sky_mask_manifest,
    sky_supervision_mask_numpy,
    verify_sky_mask_manifest,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is an optional training dependency")

ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH = 32, 32
SKY_ROWS = 12  # raw sky label: rows [0, 12)
RETURN_ROW_START = 20  # sparse LiDAR returns in rows [20, 32)
EROSION = 2
WINDOW = 3


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------


def _base_config(**overrides):
    base = {
        "run_id": "sky-supervision",
        "trainer_preset": "custom",
        "dataset_manifest": "dataset.json",
        "recording_root": "recording",
        "mask_manifest": "masks.json",
        "mask_root": "masks",
        "split_manifest": "split.json",
        "initialization_ply": "sparse_pc.ply",
        "output_dir": "run",
        "gsplat_lock": "upstream/gsplat.lock.json",
        "require_person_masks": False,
        "rgb_l1_weight": 0.6,
        "rgb_ssim_weight": 0.4,
        "lidar_range_weight": 0.0,
        "depth_manifest": "depth.json",
        "depth_root": "depth",
    }
    base.update(overrides)
    return base


def _sky_settings(**overrides):
    settings = {
        "enabled": True,
        "mask_manifest": "sky.json",
        "mask_root": "sky",
        "alpha_weight": 0.5,
        "alpha_target": 0.0,
        "mask_erosion_px": EROSION,
        "require_no_lidar_within_px": WINDOW,
    }
    settings.update(overrides)
    return settings


def _synthetic_view(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rgb_mask = np.ones((HEIGHT, WIDTH), dtype=bool)
    rgb_mask[:, -2:] = False  # renderer mask excludes a border strip
    sky = np.zeros((HEIGHT, WIDTH), dtype=bool)
    sky[:SKY_ROWS] = True
    depth_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    depth_mask[RETURN_ROW_START::2, ::2] = True
    confidence = np.where(depth_mask, np.float32(0.5), np.float32(0.0)).astype(np.float32)
    return {
        "rgb": rng.random((HEIGHT, WIDTH, 3), dtype=np.float32),
        "rendered": rng.random((HEIGHT, WIDTH, 3), dtype=np.float32),
        "alpha": rng.random((HEIGHT, WIDTH), dtype=np.float32),
        "rgb_mask": rgb_mask,
        "sky": sky,
        "depth_mask": depth_mask,
        "confidence": confidence,
    }


def _expected_effective(view: dict[str, np.ndarray], *, erosion: int = EROSION, window: int = WINDOW) -> np.ndarray:
    """Plain-loop oracle: erode(sky) & ~dilate(returns) & rgb_mask.

    Erosion treats pixels beyond the image as sky (the border does not
    erode); the LiDAR guard treats them as no return.
    """
    sky = view["sky"]
    returns = view["depth_mask"]
    out = np.zeros_like(sky)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ys, ye = max(0, y - erosion), min(HEIGHT, y + erosion + 1)
            xs, xe = max(0, x - erosion), min(WIDTH, x + erosion + 1)
            eroded = bool(sky[ys:ye, xs:xe].all())
            ys, ye = max(0, y - window), min(HEIGHT, y + window + 1)
            xs, xe = max(0, x - window), min(WIDTH, x + window + 1)
            guarded = not bool(returns[ys:ye, xs:xe].any()) if window > 0 else True
            out[y, x] = eroded and guarded
    return out & view["rgb_mask"]


def _run_loss(config, view, *, torch, with_sky=True, alpha=None, extra_tensors=None, rendered_range=None):
    from cloudstudio_3dgs.training.trainer import _render_supervision_loss

    rendered_leaf = torch.tensor(view["rendered"], requires_grad=True)
    rendered = rendered_leaf * 1.0
    if alpha is None:
        alpha = torch.tensor(view["alpha"])

    class Backend:
        @staticmethod
        def render(*args, **kwargs):
            return rendered, rendered_range, alpha, {}

    Backend.torch = torch
    tensors = {
        "rgb": torch.tensor(view["rgb"]),
        "rgb_mask": torch.tensor(view["rgb_mask"]),
        "depth_mask": torch.tensor(view["depth_mask"]),
        "confidence": torch.tensor(view["confidence"]),
    }
    if with_sky:
        tensors["sky_mask"] = torch.tensor(view["sky"])
    if extra_tensors:
        tensors.update(extra_tensors)
    loss, l1, ssim, range_loss, info = _render_supervision_loss(
        backend=Backend(),
        params={},
        sample=SimpleNamespace(camera_model="fisheye", K=np.eye(3, dtype=np.float32)),
        tensors=tensors,
        config=config,
    )
    return loss, l1.detach(), ssim.detach(), info, rendered_leaf


# ----------------------------------------------------------------------------
# default path
# ----------------------------------------------------------------------------


@requires_torch
def test_default_path_is_byte_identical() -> None:
    import torch
    from cloudstudio_3dgs.training.losses import masked_rgb_l1, masked_rgb_ssim_loss
    from cloudstudio_3dgs.training.rgb_supervision import rgb_supervision_mask
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view()
    default = TrainerConfig.from_dict(_base_config())
    assert default.sky_supervision == SkySupervisionConfig()
    assert not default.sky_supervision.enabled
    loss, l1, ssim, info, _ = _run_loss(default, view, torch=torch, with_sky=False)
    rendered = torch.tensor(view["rendered"])
    target = torch.tensor(view["rgb"])
    mask = torch.tensor(view["rgb_mask"])
    oracle_l1 = masked_rgb_l1(rendered, target, mask)
    oracle_ssim = masked_rgb_ssim_loss(rendered, target, mask)
    assert float(l1) == float(oracle_l1)
    assert float(ssim) == float(oracle_ssim)
    assert float(loss.detach()) == pytest.approx(0.6 * float(oracle_l1) + 0.4 * float(oracle_ssim), abs=1e-7)
    assert info["cloudstudio_sky_alpha_loss"] is None
    assert info["cloudstudio_sky_pixel_fraction"] is None
    assert info["cloudstudio_sky_alpha_mean"] is None
    assert SKY_MASK_INFO_KEY not in info
    # A sky tensor that happens to be present is ignored when disabled.
    loss_ignored, l1_ignored, _, info_ignored, _ = _run_loss(default, view, torch=torch, with_sky=True)
    assert float(l1_ignored) == float(l1)
    assert float(loss_ignored.detach()) == float(loss.detach())
    assert SKY_MASK_INFO_KEY not in info_ignored
    # "all" without exclude still hands back the renderer mask object itself.
    same = rgb_supervision_mask(
        torch, rgb_mask=mask, depth_mask=None, confidence=None, mode="all", dilation_radius_px=0
    )
    assert same.mask is mask and same.excluded_pixels == 0


# ----------------------------------------------------------------------------
# effective mask twins
# ----------------------------------------------------------------------------


@requires_torch
def test_effective_mask_twins_match_the_oracle_and_the_lidar_guard() -> None:
    import torch
    from cloudstudio_3dgs.training.sky_supervision import sky_supervision_mask

    view = _synthetic_view(seed=3)
    expected = _expected_effective(view)
    assert 0 < expected.sum() < view["sky"].sum()
    # Erosion took the two rows next to the label boundary, not the top edge.
    assert expected[0, :-2].all() and not expected[SKY_ROWS - 1].any()
    numpy_mask = sky_supervision_mask_numpy(
        sky_mask=view["sky"], rgb_mask=view["rgb_mask"], depth_mask=view["depth_mask"],
        erosion_px=EROSION, lidar_window_px=WINDOW,
    )
    torch_mask = sky_supervision_mask(
        torch,
        sky_mask=torch.tensor(view["sky"]), rgb_mask=torch.tensor(view["rgb_mask"]),
        depth_mask=torch.tensor(view["depth_mask"]), erosion_px=EROSION, lidar_window_px=WINDOW,
    )
    assert np.array_equal(numpy_mask.mask, expected)
    assert np.array_equal(torch_mask.mask.numpy(), expected)
    assert numpy_mask.sky_pixels == torch_mask.sky_pixels == int(expected.sum())
    assert numpy_mask.raw_sky_pixels == torch_mask.raw_sky_pixels == int((view["sky"] & view["rgb_mask"]).sum())
    assert numpy_mask.fraction == pytest.approx(expected.sum() / view["rgb_mask"].sum())

    # A LiDAR return inside the sky band (an overexposed wall labelled sky)
    # clears its whole window; nothing else changes.
    guarded = {key: value.copy() for key, value in view.items()}
    guarded["depth_mask"][4, 10] = True
    expected_guarded = _expected_effective(guarded)
    assert not expected_guarded[1:8, 7:14].any()
    assert expected_guarded.sum() == expected.sum() - int(expected[1:8, 7:14].sum())
    for twin in (
        sky_supervision_mask_numpy(
            sky_mask=guarded["sky"], rgb_mask=guarded["rgb_mask"], depth_mask=guarded["depth_mask"],
            erosion_px=EROSION, lidar_window_px=WINDOW,
        ).mask,
        sky_supervision_mask(
            torch,
            sky_mask=torch.tensor(guarded["sky"]), rgb_mask=torch.tensor(guarded["rgb_mask"]),
            depth_mask=torch.tensor(guarded["depth_mask"]), erosion_px=EROSION, lidar_window_px=WINDOW,
        ).mask.numpy(),
    ):
        assert np.array_equal(twin, expected_guarded)

    # Both guards off: the raw label within the renderer mask; no returns
    # (depth_mask None) skips the LiDAR guard only.
    raw = sky_supervision_mask_numpy(
        sky_mask=view["sky"], rgb_mask=view["rgb_mask"], depth_mask=view["depth_mask"],
        erosion_px=0, lidar_window_px=0,
    )
    assert np.array_equal(raw.mask, view["sky"] & view["rgb_mask"])
    no_returns = sky_supervision_mask_numpy(
        sky_mask=view["sky"], rgb_mask=view["rgb_mask"], depth_mask=None,
        erosion_px=EROSION, lidar_window_px=WINDOW,
    )
    assert np.array_equal(no_returns.mask, _expected_effective(view, window=0))
    with pytest.raises(ValueError, match="differs from rgb_mask"):
        sky_supervision_mask_numpy(
            sky_mask=view["sky"][:8], rgb_mask=view["rgb_mask"], depth_mask=None,
            erosion_px=0, lidar_window_px=0,
        )


# ----------------------------------------------------------------------------
# loss behaviour
# ----------------------------------------------------------------------------


@requires_torch
def test_sky_alpha_loss_value_and_gradient_direction() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view(seed=1)
    expected = _expected_effective(view)
    config = TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings(exclude_photometric=False)))
    alpha_leaf = torch.tensor(view["alpha"], requires_grad=True)
    loss, l1, ssim, info, rendered_leaf = _run_loss(config, view, torch=torch, alpha=alpha_leaf * 1.0)
    sky_loss = info["cloudstudio_sky_alpha_loss"]
    assert float(sky_loss) == pytest.approx(float(np.abs(view["alpha"][expected]).mean()), rel=1e-6)
    assert float(info["cloudstudio_sky_alpha_mean"]) == pytest.approx(float(view["alpha"][expected].mean()), rel=1e-6)
    assert info["cloudstudio_sky_pixel_fraction"] == pytest.approx(expected.sum() / view["rgb_mask"].sum())
    assert info["cloudstudio_sky_pixels"] == int(expected.sum())
    assert np.array_equal(info[SKY_MASK_INFO_KEY].numpy(), expected)
    # Total = photometric (unmasked here) + 0.5 * sky alpha term.
    assert float(loss.detach()) == pytest.approx(0.6 * float(l1) + 0.4 * float(ssim) + 0.5 * float(sky_loss), abs=1e-6)
    loss.backward()
    grad = alpha_leaf.grad.numpy()
    # Descent lowers alpha on every effective sky pixel and touches nothing else.
    assert (grad[expected] > 0.0).all()
    assert np.abs(grad[~expected]).max() == 0.0
    assert np.allclose(grad[expected], 0.5 / expected.sum(), rtol=1e-5)

    # Two "gaussians": A paints a patch inside the sky band, B a patch on the
    # wall. Only A's opacity is pushed down by the sky term.
    opacity_a = torch.tensor(0.0, requires_grad=True)  # sigmoid -> 0.5
    opacity_b = torch.tensor(0.0, requires_grad=True)
    region_a = torch.zeros((HEIGHT, WIDTH), dtype=torch.bool)
    region_a[2:6, 4:12] = True
    region_b = torch.zeros((HEIGHT, WIDTH), dtype=torch.bool)
    region_b[24:30, 4:12] = True
    assert bool(torch.tensor(expected)[region_a].all()) and not bool(torch.tensor(expected)[region_b].any())
    alpha_map = (
        torch.sigmoid(opacity_a) * region_a.float()
        + torch.sigmoid(opacity_b) * region_b.float()
    )
    loss_ab, _, _, info_ab, _ = _run_loss(config, view, torch=torch, alpha=alpha_map)
    info_ab["cloudstudio_sky_alpha_loss"].backward()
    assert float(opacity_a.grad) > 0.0  # d(loss)/d(logit) > 0: descent lowers A's opacity
    assert float(opacity_b.grad) == 0.0

    # alpha_target is honoured: at target the term is zero.
    at_target = TrainerConfig.from_dict(
        _base_config(sky_supervision=_sky_settings(alpha_target=0.25, exclude_photometric=False))
    )
    _, _, _, info_t, _ = _run_loss(at_target, view, torch=torch, alpha=torch.full((HEIGHT, WIDTH), 0.25))
    assert float(info_t["cloudstudio_sky_alpha_loss"]) == 0.0


@requires_torch
def test_photometric_exclusion_masks_the_rgb_terms() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view(seed=2)
    expected = _expected_effective(view)
    supervised = view["rgb_mask"] & ~expected
    config = TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings()))
    assert config.sky_supervision.exclude_photometric
    loss, l1, ssim, info, leaf = _run_loss(config, view, torch=torch)
    # Only the photometric part reaches the render leaf: the alpha term is a
    # function of the (fixed) alpha tensor here.
    loss.backward()
    grad = leaf.grad.detach().numpy()
    assert np.abs(grad[expected]).max() == 0.0
    assert np.abs(grad[supervised]).sum() > 0.0
    diff = np.abs(view["rendered"] - view["rgb"])
    assert float(l1) == pytest.approx(float(diff[supervised].mean()), rel=1e-6)
    assert float(l1) != pytest.approx(float(diff[view["rgb_mask"]].mean()), rel=1e-3)
    assert info["cloudstudio_rgb_supervised_pixels"] == int(supervised.sum())
    assert info["cloudstudio_rgb_supervised_fraction"] == pytest.approx(supervised.sum() / view["rgb_mask"].sum())
    assert float(ssim) > 0.0

    # Exclusion off: the RGB terms are the full-mask ones again, while the
    # alpha term is unchanged.
    unmasked = TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings(exclude_photometric=False)))
    loss_u, l1_u, _, info_u, leaf_u = _run_loss(unmasked, view, torch=torch)
    assert float(l1_u) == pytest.approx(float(diff[view["rgb_mask"]].mean()), rel=1e-6)
    assert info_u["cloudstudio_rgb_supervised_fraction"] == 1.0
    assert float(info_u["cloudstudio_sky_alpha_loss"]) == float(info["cloudstudio_sky_alpha_loss"])
    loss_u.backward()
    assert np.abs(leaf_u.grad.detach().numpy()[expected]).sum() > 0.0

    # Composes with lidar_support: sky pixels never re-enter the support.
    composed = TrainerConfig.from_dict(
        _base_config(
            rgb_supervision_mask="lidar_support",
            rgb_supervision_dilation_radius_px=WINDOW,
            sky_supervision=_sky_settings(),
        )
    )
    _, _, _, info_c, _ = _run_loss(composed, view, torch=torch)
    assert info_c["cloudstudio_rgb_supervised_pixels"] <= int(supervised.sum())

    # An all-sky view keeps a graph-connected zero photometric loss.
    all_sky = dict(view)
    all_sky["sky"] = np.ones_like(view["sky"])
    all_sky["depth_mask"] = np.zeros_like(view["depth_mask"])
    all_sky["confidence"] = np.zeros_like(view["confidence"])
    zero_config = TrainerConfig.from_dict(
        _base_config(sky_supervision=_sky_settings(mask_erosion_px=0, require_no_lidar_within_px=0))
    )
    loss_z, l1_z, ssim_z, info_z, leaf_z = _run_loss(zero_config, all_sky, torch=torch)
    assert float(l1_z) == 0.0 and float(ssim_z) == 0.0
    assert info_z["cloudstudio_rgb_psnr"] is None
    assert info_z["cloudstudio_rgb_supervised_fraction"] == 0.0
    assert float(info_z["cloudstudio_sky_alpha_loss"]) == pytest.approx(float(np.abs(all_sky["alpha"][all_sky["rgb_mask"]]).mean()), rel=1e-6)
    loss_z.backward()  # must not raise
    assert float(leaf_z.grad.abs().sum()) == 0.0

    # Enabled without a sky tensor is a hard error, never a silent skip.
    with pytest.raises(ValueError, match="carries no sky mask"):
        _run_loss(config, view, torch=torch, with_sky=False)


@requires_torch
def test_da2_term_ignores_sky_pixels() -> None:
    import torch
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    view = _synthetic_view(seed=4)
    expected = _expected_effective(view)
    rendered_range = torch.full((HEIGHT, WIDTH), 3.0)
    # DA2 target agrees off the sky and is wildly off on the sky pixels.
    da2 = np.where(expected, np.float32(30.0), np.float32(3.0)).astype(np.float32)
    da2_tensors = {"da2_range_m": torch.tensor(da2), "da2_mask": torch.tensor(view["rgb_mask"])}

    def run(**sky_overrides):
        config = TrainerConfig.from_dict(
            _base_config(
                da2_depth_weight=0.15,
                mono_depth_manifest="da2.json",
                mono_depth_root="da2",
                sky_supervision=_sky_settings(**sky_overrides),
            )
        )
        _, _, _, info, _ = _run_loss(
            config, view, torch=torch, extra_tensors=da2_tensors, rendered_range=rendered_range
        )
        return info["cloudstudio_da2_depth_loss"]

    assert float(run(exclude_mono_depth=True, exclude_photometric=False)) == 0.0
    assert float(run(exclude_mono_depth=True, exclude_photometric=True)) == 0.0
    assert float(run(exclude_mono_depth=False, exclude_photometric=False)) > 0.0


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------


def _load_face_cache_fixture():
    spec = importlib.util.spec_from_file_location("_face_dataset_fixture", ROOT / "tests" / "test_face_dataset.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model_stub() -> dict:
    return {
        "id": "synthetic/segformer",
        "revision": "0",
        "config_sha256": "c" * 64,
        "weights_sha256": "d" * 64,
        "license_note": "synthetic test stub",
    }


def _rule_stub() -> dict:
    return {
        "decision": "argmax",
        "sky_probability_threshold": 0.5,
        "model_input": "face",
        "upsampling": "nearest",
        "valid_mask": "face_cache_combined_mask",
        "sky_fraction_denominator": "valid_pixels",
    }


def _write_sky_cache(face_manifest_path: Path, root: Path, *, sky_rows: int = 3, drop_face: str | None = None) -> Path:
    """Sky PNGs (top ``sky_rows`` rows AND face-valid) for every face + manifest.

    Built through the data-side ``build_sky_mask_manifest`` so the fixture
    satisfies the authoritative contract (records, summary, signature key).
    """
    face = json.loads(face_manifest_path.read_text(encoding="utf-8"))
    cache_root = face_manifest_path.parent
    faces_dir = root / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        (camera_id, spec["face_id"]): spec
        for camera_id, entry in face["cameras"].items()
        for spec in entry["faces"]
    }
    records = []
    for image in face["images"]:
        for entry in image["faces"]:
            if entry["face_id"] == drop_face:
                continue
            spec = specs[(image["camera_id"], entry["face_id"])]
            with Image.open(cache_root / entry["mask_path"]) as source:
                valid = np.asarray(source.convert("L"), dtype=np.uint8) > 0
            sky = np.zeros((spec["height"], spec["width"]), dtype=bool)
            sky[:sky_rows] = True
            sky &= valid
            relative = sky_mask_path_for(image["image_id"], entry["face_id"])
            path = root / relative
            Image.fromarray(np.where(sky, 255, 0).astype(np.uint8)).save(path)
            valid_pixels = int(valid.sum())
            sky_pixels = int(sky.sum())
            records.append(
                {
                    "image_id": image["image_id"],
                    "camera_id": image["camera_id"],
                    "face_id": entry["face_id"],
                    "width": spec["width"],
                    "height": spec["height"],
                    "mask_path": relative,
                    "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "valid_pixels": valid_pixels,
                    "sky_pixels": sky_pixels,
                    "sky_fraction": sky_pixels / valid_pixels if valid_pixels else 0.0,
                }
            )
    manifest = build_sky_mask_manifest(
        split=face["split"],
        source_face_manifest_sha256=face["face_manifest_sha256"],
        source_identity=face.get("source_identity", {}),
        model=_model_stub(),
        rule=_rule_stub(),
        records=records,
    )
    manifest_path = root / "sky_mask_train.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest_path


@requires_torch
def test_dataset_binds_crops_and_fails_closed() -> None:
    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

    fixture = _load_face_cache_fixture()
    with tempfile.TemporaryDirectory(prefix="sky-cache-") as temporary:
        cache_root = Path(temporary) / "face4"
        cache_root.mkdir()
        manifest_path, _record, _skipped, _grids = fixture.build_cache(cache_root)
        sky_root = Path(temporary) / "sky"
        sky_manifest = _write_sky_cache(manifest_path, sky_root)

        dataset = FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=sky_manifest, sky_mask_root=sky_root)
        assert dataset.identity["sky_mask_manifest_sha256"] == verify_sky_mask_manifest(
            json.loads(sky_manifest.read_text(encoding="utf-8"))
        )
        plain = FaceCacheDataset(manifest_path, cache_root)
        assert "sky_mask_manifest_sha256" not in plain.identity
        assert plain[0].sky_mask is None
        full = dataset[0]
        assert full.sky_mask is not None and full.sky_mask.shape == full.rgb_mask.shape
        expected_sky = np.zeros_like(full.rgb_mask)
        expected_sky[:3] = True
        expected_sky &= full.rgb_mask
        assert expected_sky.any()
        np.testing.assert_array_equal(full.sky_mask, expected_sky)

        crop = {"sample_id": full.image_id, "x": 2, "y": 1, "width": 4, "height": 5}
        cropped = FaceCacheDataset(
            manifest_path, cache_root, tile_views=[crop],
            sky_mask_manifest_path=sky_manifest, sky_mask_root=sky_root,
        )[0]
        np.testing.assert_array_equal(cropped.sky_mask, full.sky_mask[1:6, 2:6])
        np.testing.assert_array_equal(cropped.rgb_mask, full.rgb_mask[1:6, 2:6])

        # Missing record for a selected face: refused at construction.
        first_face = full.image_id.split("::")[1]
        missing_root = Path(temporary) / "sky_missing"
        missing_manifest = _write_sky_cache(manifest_path, missing_root, drop_face=first_face)
        with pytest.raises(ValueError, match="sky mask manifest does not cover"):
            FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=missing_manifest, sky_mask_root=missing_root)

        # Missing file: refused at access, never a silent "no sky here".
        gone_root = Path(temporary) / "sky_gone"
        gone_manifest = _write_sky_cache(manifest_path, gone_root)
        gone = FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=gone_manifest, sky_mask_root=gone_root)
        shutil.rmtree(gone_root / "faces")
        with pytest.raises(FileNotFoundError, match="missing sky mask"):
            gone[0]

        # Tampered artifact: SHA mismatch.
        bad_root = Path(temporary) / "sky_bad"
        bad_manifest = _write_sky_cache(manifest_path, bad_root)
        bad = FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=bad_manifest, sky_mask_root=bad_root)
        target = bad_root / json.loads(bad_manifest.read_text(encoding="utf-8"))["masks"][0]["mask_path"]
        Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(target)
        with pytest.raises(ValueError, match="sky mask SHA256 mismatch"):
            bad[0]

        # Wrong Face4 binding and tampered signature.
        payload = json.loads(sky_manifest.read_text(encoding="utf-8"))
        rebound = dict(payload, source_face_manifest_sha256="0" * 64)
        rebound_path = Path(temporary) / "rebound.json"
        rebound_path.write_text(json.dumps(sign_sky_mask_manifest(rebound)), encoding="utf-8")
        with pytest.raises(ValueError, match="different Face4 cache"):
            FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=rebound_path, sky_mask_root=sky_root)
        tampered = dict(payload, split="val")
        tampered_path = Path(temporary) / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="signature mismatch"):
            FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=tampered_path, sky_mask_root=sky_root)
        with pytest.raises(ValueError, match="provided together"):
            FaceCacheDataset(manifest_path, cache_root, sky_mask_manifest_path=sky_manifest)


# ----------------------------------------------------------------------------
# config, contract, validation, gate
# ----------------------------------------------------------------------------


def _classic_config(**overrides) -> dict:
    config = _base_config(
        max_steps=3000,
        checkpoint_every=500,
        color_model="sh",
        sh_degree=1,
        sh_degree_interval=0,
        densification_strategy="default_3dgs",
        topology_policy={"mode": "adaptive_growth"},
        mcmc_refine_start_iter=500,
        mcmc_refine_stop_iter=2100,
        mcmc_refine_every=100,
        default_strategy={
            "exact_mipmap_lifecycle": True,
            "refine_start_iter": 500,
            "refine_stop_iter": 2100,
            "refine_every": 100,
            "refine_scale2d_stop_iter": 2100,
            "reset_every": 300,
            "absgrad": True,
            "grow_grad2d": 0.00015,
            "split_scale_m": 0.2,
            "prune_scale_m": 0.2,
            "prune_opa": 0.1,
            "prune_opa_late": 0.05,
            "prune_switch_step": 1500,
            "prune_scale2d": 0.15,
            "reset_opacity_cap": 0.2,
        },
    )
    config.update(overrides)
    return config


def _minimal_face_cache(root: Path) -> Path:
    from cloudstudio_3dgs.training.face_dataset import sign_face_manifest

    face = sign_face_manifest(
        {
            "schema_version": 1,
            "kind": "fisheye_face_cache",
            "split": "train",
            "source_identity": {"dataset_manifest_sha256": "synthetic"},
            "cameras": {"left": {"faces": [{"face_id": "front"}]}},
            "images": [{"image_id": "image", "camera_id": "left", "faces": []}],
        }
    )
    path = root / "face_manifest.json"
    path.write_text(json.dumps(face), encoding="utf-8")
    return path


def _minimal_sky_manifest(root: Path, source_sha: str, *, split: str = "train") -> Path:
    manifest = build_sky_mask_manifest(
        split=split,
        source_face_manifest_sha256=source_sha,
        source_identity={"dataset_manifest_sha256": "synthetic"},
        model=_model_stub(),
        rule=_rule_stub(),
        records=[
            {
                "image_id": "image", "camera_id": "left", "face_id": "front",
                "width": 8, "height": 8, "mask_path": sky_mask_path_for("image", "front"),
                "mask_sha256": "a" * 64, "valid_pixels": 64, "sky_pixels": 0, "sky_fraction": 0.0,
            }
        ],
    )
    assert manifest["kind"] == SKY_MASK_KIND
    path = root / "sky_mask_train.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@requires_torch
def test_contract_gains_keys_only_when_enabled() -> None:
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    default = TrainerConfig.from_dict(_classic_config())
    explicit = TrainerConfig.from_dict(
        _classic_config(sky_supervision={"enabled": False, "alpha_weight": 0.9, "growth_block": False})
    )
    explicit.sky_supervision.validate()
    assert "sky_supervision" not in default.contract_dict()["loss_contract"]
    assert "sky_growth_block" not in default.contract_dict()["strategy"]
    assert default.contract_dict() == explicit.contract_dict()

    with tempfile.TemporaryDirectory(prefix="sky-contract-") as temporary:
        root = Path(temporary)
        face_path = _minimal_face_cache(root)
        face_sha = json.loads(face_path.read_text(encoding="utf-8"))["face_manifest_sha256"]
        sky_path = _minimal_sky_manifest(root, face_sha)
        enabled = TrainerConfig.from_dict(
            _classic_config(
                face_cache_manifest=str(face_path),
                face_cache_root=str(root),
                sky_supervision=_sky_settings(mask_manifest=str(sky_path), mask_root=str(root)),
            )
        )
        contract = enabled.contract_dict()
        sky_contract = contract["loss_contract"]["sky_supervision"]
        assert sky_contract["manifest_sha256"] == verify_sky_mask_manifest(
            json.loads(sky_path.read_text(encoding="utf-8"))
        )
        assert sky_contract["alpha_weight"] == 0.5 and sky_contract["alpha_target"] == 0.0
        assert sky_contract["excluded_photometric_terms"] and sky_contract["excluded_depth_terms"] == ["da2_depth"]
        assert sky_contract["alpha_source"] == "accumulated_alpha_before_backdrop_composite"
        assert sky_contract["growth_block_rule"]["min_sky_fraction"] == 0.5
        assert contract["strategy"]["sky_growth_block"] == SkyGrowthBlock().state_dict()
        # No path leaks into the identity; the manifest is bound by its SHA.
        assert "mask_manifest" not in sky_contract and "mask_root" not in sky_contract
        without = copy.deepcopy(contract)
        without["loss_contract"].pop("sky_supervision")
        without["strategy"].pop("sky_growth_block")
        base = TrainerConfig.from_dict(
            _classic_config(face_cache_manifest=str(face_path), face_cache_root=str(root))
        ).contract_dict()
        assert without == base
        # growth_block off drops only the strategy key.
        no_block = TrainerConfig.from_dict(
            _classic_config(
                face_cache_manifest=str(face_path),
                face_cache_root=str(root),
                sky_supervision=_sky_settings(mask_manifest=str(sky_path), mask_root=str(root), growth_block=False),
            )
        ).contract_dict()
        assert "sky_growth_block" not in no_block["strategy"]
        assert no_block["loss_contract"]["sky_supervision"]["growth_block_rule"] is None


@requires_torch
def test_validation_refusals() -> None:
    from cloudstudio_3dgs.training.trainer import TrainerConfig

    with pytest.raises(ValueError, match="must be an object"):
        TrainerConfig.from_dict(_base_config(sky_supervision=[]))
    with pytest.raises(ValueError, match="unknown field"):
        TrainerConfig.from_dict(_base_config(sky_supervision={"enabled": False, "weight": 1.0}))
    with pytest.raises(ValueError, match="requires mask_manifest and mask_root"):
        TrainerConfig.from_dict(_base_config(sky_supervision={"enabled": True})).validate()
    with pytest.raises(ValueError, match="set but enabled is false"):
        TrainerConfig.from_dict(_base_config(sky_supervision={"enabled": False, "mask_manifest": "x"})).validate()
    with pytest.raises(ValueError, match="alpha_weight must be non-negative"):
        TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings(alpha_weight=-1.0))).validate()
    with pytest.raises(ValueError, match="alpha_target must lie within"):
        TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings(alpha_target=2.0))).validate()
    with pytest.raises(ValueError, match="mask_erosion_px must be an integer"):
        TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings(mask_erosion_px=-1))).validate()
    with pytest.raises(ValueError, match="would do nothing"):
        TrainerConfig.from_dict(
            _base_config(
                sky_supervision=_sky_settings(
                    alpha_weight=0.0, exclude_photometric=False, exclude_mono_depth=False, growth_block=False
                )
            )
        ).validate()
    # Face-cache training is required, and the growth block needs the
    # classic exact lifecycle (a silent no-op elsewhere is refused).
    with pytest.raises(ValueError, match="requires face-cache training"):
        TrainerConfig.from_dict(_base_config(sky_supervision=_sky_settings())).validate()
    with pytest.raises(ValueError, match="growth_block requires"):
        TrainerConfig.from_dict(
            _base_config(
                face_cache_manifest="face.json",
                face_cache_root="face",
                sky_supervision=_sky_settings(),
            )
        ).validate()

    # File-level checks on the MCMC base with growth_block off: the exact
    # lifecycle base needs a signed renderer mask manifest first (an
    # unrelated precondition), and the growth_block refusal is pinned above.
    with tempfile.TemporaryDirectory(prefix="sky-validate-") as temporary:
        root = Path(temporary)
        face_path = _minimal_face_cache(root)
        face_sha = json.loads(face_path.read_text(encoding="utf-8"))["face_manifest_sha256"]
        missing = root / "sky_mask_train.json"
        with pytest.raises(FileNotFoundError, match="sky mask manifest is missing: .*sky_mask_train.json"):
            TrainerConfig.from_dict(
                _base_config(
                    face_cache_manifest=str(face_path),
                    face_cache_root=str(root),
                    sky_supervision=_sky_settings(mask_manifest=str(missing), mask_root=str(root), growth_block=False),
                )
            ).validate()
        rebound = _minimal_sky_manifest(root, "0" * 64)
        with pytest.raises(ValueError, match="bound to different Face4 inputs"):
            TrainerConfig.from_dict(
                _base_config(
                    face_cache_manifest=str(face_path),
                    face_cache_root=str(root),
                    sky_supervision=_sky_settings(mask_manifest=str(rebound), mask_root=str(root), growth_block=False),
                )
            ).validate()
        wrong_split = _minimal_sky_manifest(root, face_sha, split="val")
        with pytest.raises(ValueError, match="different splits"):
            TrainerConfig.from_dict(
                _base_config(
                    face_cache_manifest=str(face_path),
                    face_cache_root=str(root),
                    sky_supervision=_sky_settings(mask_manifest=str(wrong_split), mask_root=str(root), growth_block=False),
                )
            ).validate()
        bound = _minimal_sky_manifest(root, face_sha)
        tampered = json.loads(bound.read_text(encoding="utf-8"))
        tampered["rule"] = "other"
        bound.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="signature mismatch"):
            TrainerConfig.from_dict(
                _base_config(
                    face_cache_manifest=str(face_path),
                    face_cache_root=str(root),
                    sky_supervision=_sky_settings(mask_manifest=str(bound), mask_root=str(root), growth_block=False),
                )
            ).validate()
        # A correctly bound manifest passes: the synthetic base has nothing
        # else validate() checks on disk, so this is the positive path.
        good = _minimal_sky_manifest(root, face_sha)
        TrainerConfig.from_dict(
            _base_config(
                face_cache_manifest=str(face_path),
                face_cache_root=str(root),
                sky_supervision=_sky_settings(mask_manifest=str(good), mask_root=str(root), growth_block=False),
            )
        ).validate()


def _upstream_data_gate() -> dict:
    sha = {name: value * 64 for name, value in (("dataset", "1"), ("split", "2"), ("face", "3"), ("mask", "4"), ("da2", "5"), ("tile", "6"))}
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
                "training_dataset_manifest_sha256": sha["dataset"],
                "split_manifest_sha256": sha["split"],
                "face4_train_manifest_sha256": sha["face"],
                "face4_val_manifest_sha256": sha["face"],
                "training_circle_mask_manifest_sha256": sha["mask"],
                "da2_train_manifest_sha256": sha["da2"],
                "spatial_tile_plan_manifest_sha256": sha["tile"],
            },
        }
    )


def _signed(config: dict) -> dict:
    signed = copy.deepcopy(config)
    signed.pop("config_manifest_sha256", None)
    signed["config_manifest_sha256"] = hashlib.sha256(canonical_json_bytes(signed)).hexdigest()
    return signed


# The parity arm the schedule-contract test signs (tests/test_schedule_contract.py).
PARITY_ARM = {
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
    "tangent_proposal": {"enabled": True, "reject_unsupported_births": True},
    "geometry_regularization": {
        "enabled": True,
        "opacity_sparsity_weight": 1e-4,
        "scale_upper_weight": 1e-4,
        "anisotropy_weight": 1e-4,
        "max_scale_ratio_to_reference": 8.0,
        "max_anisotropy": 10.0,
    },
}


def test_adaptive_growth_gate_refuses_a_sky_supervision_arm() -> None:
    gate = advance_adaptive_growth_gate(_upstream_data_gate(), _signed(PARITY_ARM), stage="boundary")
    assert gate["training_allowed"]
    explicit_off = dict(PARITY_ARM, sky_supervision={"enabled": False})
    assert advance_adaptive_growth_gate(_upstream_data_gate(), _signed(explicit_off), stage="boundary")["training_allowed"]
    research = dict(PARITY_ARM, sky_supervision=_sky_settings())
    with pytest.raises(ValueError, match="sky_supervision is a research departure"):
        advance_adaptive_growth_gate(_upstream_data_gate(), _signed(research), stage="boundary")
    tampered = _signed(research)
    tampered["max_steps"] = 7481
    with pytest.raises(ValueError, match="signature mismatch"):
        advance_adaptive_growth_gate(_upstream_data_gate(), tampered, stage="boundary")


# ----------------------------------------------------------------------------
# growth block
# ----------------------------------------------------------------------------


def _adapter(**overrides):
    from cloudstudio_3dgs.training.default_strategy_adapter import DefaultStrategyAdapter

    settings = dict(
        scene_scale=10.0,
        refine_start_iter=500,
        refine_stop_iter=2000,
        refine_every=100,
        reset_every=300,
        grow_grad2d=0.00015,
        prune_opa=0.1,
        split_scale_m=0.2,
        prune_scale_m=0.2,
        exact_mipmap_lifecycle=True,
        prune_opa_late=0.05,
        prune_switch_step=100000,
        reset_opacity_cap=0.2,
    )
    settings.update(overrides)
    return DefaultStrategyAdapter(**settings)


def _population(torch, means, grad2d):
    count = len(means)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.tensor(means, dtype=torch.float32)),
            "scales": torch.nn.Parameter(torch.full((count, 3), 0.05).log()),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count)),
            "opacities": torch.nn.Parameter(torch.full((count,), 0.5).logit()),
            "colors": torch.nn.Parameter(torch.zeros(count, 3)),
        }
    )
    optimizers = {name: torch.optim.Adam([parameter], lr=1e-3) for name, parameter in params.items()}
    for parameter in params.values():
        parameter.grad = torch.zeros_like(parameter)
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    state = {
        "grad2d": torch.tensor(grad2d, dtype=torch.float32),
        "count": torch.ones(count),
        "radii": torch.zeros(count),
        "scene_scale": 10.0,
    }
    return params, optimizers, state


def _step_info(torch, centres, sky):
    count = len(centres)
    means2d = torch.tensor([centres], dtype=torch.float32, requires_grad=True)  # [1, N, 2]
    means2d.grad = torch.zeros_like(means2d)
    return {
        "means2d": means2d,
        "radii": torch.ones((1, count, 2)),
        "width": WIDTH,
        "height": HEIGHT,
        "n_cameras": 1,
        "gaussian_ids": None,
        SKY_MASK_INFO_KEY: torch.tensor(sky),
    }


@requires_torch
def test_growth_block_counts_sky_observations_and_blocks_sky_parents() -> None:
    import torch

    sky = np.zeros((HEIGHT, WIDTH), dtype=bool)
    sky[:10] = True
    block = SkyGrowthBlock()
    adapter = _adapter(sky_growth_block=block)
    # Row 0 projects into the sky band, row 1 onto the wall, row 2 off-image.
    params, optimizers, state = _population(
        torch, [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], grad2d=[0.001, 0.001, 0.001]
    )
    centres = [[5.5, 2.5], [5.5, 20.5], [-3.0, 2.5]]
    block.accumulate(params, state, _step_info(torch, centres, sky))
    block.accumulate(params, state, _step_info(torch, centres, sky))
    assert state[SKY_SEEN_KEY].tolist() == [2.0, 2.0, 2.0]
    assert state[SKY_HIT_KEY].tolist() == [2.0, 0.0, 0.0]
    assert block.last_stats == {"observed": 3, "sky_hits": 1}

    adapter._ensure_lineage(params, state)
    clone_count, split_count = adapter._grow_mipmap(params, optimizers, state)
    assert (clone_count, split_count) == (2, 0)
    assert len(params["means"]) == 5
    growth = adapter._last_growth_event
    assert growth["sky_growth_blocked_count"] == 1
    assert growth["sky_growth_blocked_total"] == 1
    assert growth["selected_parent_count"] == 2
    assert state[SKY_BLOCKED_TOTAL_KEY] == 1
    # The clones are of rows 1 and 2 (row 0 is blocked): z stays 0.
    assert torch.allclose(params["means"][3:, 2].detach(), torch.zeros(2))
    # The accumulators rode the duplicate (length follows the population).
    assert len(state[SKY_SEEN_KEY]) == 5 and len(state[SKY_HIT_KEY]) == 5
    block.reset(state)
    assert float(state[SKY_SEEN_KEY].sum()) == 0.0 and float(state[SKY_HIT_KEY].sum()) == 0.0

    # Exactly half of the observations in sky blocks; less than half does not.
    params, optimizers, state = _population(torch, [[0.0, 0.0, 0.0]] * 2, grad2d=[0.001, 0.001])
    in_sky = [[5.5, 2.5], [5.5, 2.5]]
    on_wall = [[5.5, 20.5], [5.5, 20.5]]
    block.accumulate(params, state, _step_info(torch, in_sky, sky))
    block.accumulate(params, state, _step_info(torch, on_wall, sky))
    block.accumulate(params, state, _step_info(torch, [[5.5, 2.5], [5.5, 20.5]], sky))
    # row 0: 2/3 sky -> blocked; row 1: 1/3 sky -> breeds.
    adapter._ensure_lineage(params, state)
    assert adapter._grow_mipmap(params, optimizers, state) == (1, 0)
    assert adapter._last_growth_event["sky_growth_blocked_count"] == 1

    # Without the block the same population breeds twice and no key appears.
    plain = _adapter()
    params, optimizers, state = _population(torch, [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]], grad2d=[0.001, 0.001])
    plain._ensure_lineage(params, state)
    assert plain._grow_mipmap(params, optimizers, state) == (2, 0)
    assert "sky_growth_blocked_count" not in plain._last_growth_event
    assert plain.state_dict()["sky_growth_block"] is None
    assert adapter.state_dict()["sky_growth_block"] == block.state_dict()

    # The loss must have provided the mask: fail closed otherwise.
    info = _step_info(torch, centres, sky)
    info.pop(SKY_MASK_INFO_KEY)
    with pytest.raises(RuntimeError, match="effective sky mask"):
        block.accumulate(params, state, info)


@requires_torch
def test_growth_block_is_wired_into_the_lifecycle_step() -> None:
    import torch

    sky = np.zeros((HEIGHT, WIDTH), dtype=bool)
    sky[:10] = True
    block = SkyGrowthBlock()
    adapter = _adapter(sky_growth_block=block)
    params, optimizers, state = _population(torch, [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]], grad2d=[0.0, 0.0])
    state["radii"] = None
    centres = [[5.5, 2.5], [5.5, 20.5]]
    # A non-refine step accumulates; the refine step consumes and zeroes.
    adapter._step_post_backward_mipmap(
        params=params, optimizers=optimizers, state=state, step=650, info=_step_info(torch, centres, sky)
    )
    assert state[SKY_SEEN_KEY].tolist() == [1.0, 1.0]
    assert state[SKY_HIT_KEY].tolist() == [1.0, 0.0]
    state["grad2d"] = torch.tensor([0.001, 0.001])
    state["count"] = torch.ones(2)
    adapter._step_post_backward_mipmap(
        params=params, optimizers=optimizers, state=state, step=700, info=_step_info(torch, centres, sky)
    )
    event = adapter.last_lifecycle_event
    assert event["growth_diagnostics"]["sky_growth_blocked_count"] == 1
    assert event["clone_count"] == 1
    assert len(params["means"]) == 3
    assert float(state[SKY_SEEN_KEY].sum()) == 0.0 and len(state[SKY_SEEN_KEY]) == 3
