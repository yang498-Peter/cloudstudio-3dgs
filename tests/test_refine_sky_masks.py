"""tools/refine_sky_masks.py: the photometric refinement keeps bright smooth sky, drops dark or
high-gradient pixels (branches) from the label, never adds sky, and writes a cache the verifier
accepts and binds to the same Face4 manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
from PIL import Image

from cloudstudio_3dgs.data.sky_masks import (
    build_sky_mask_manifest,
    load_sky_mask,
    load_sky_mask_manifest,
    sky_mask_path_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location("refine_sky_masks", REPO_ROOT / "tools" / "refine_sky_masks.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sky_photo_with_branch(height: int = 120, width: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Bright blue sky with a soft cloud gradient and one dark 3 px branch; the raw label says
    everything is sky (the SegFormer failure mode)."""
    rgb = np.zeros((height, width, 3), np.uint8)
    ramp = np.linspace(200, 235, width, dtype=np.float32)[None, :]
    rgb[..., 0] = np.clip(ramp - 40, 0, 255)
    rgb[..., 1] = np.clip(ramp - 10, 0, 255)
    rgb[..., 2] = np.clip(ramp, 0, 255)
    rgb[60:63, 10:150] = (40, 30, 20)
    raw = np.ones((height, width), bool)
    return rgb, raw


def test_refinement_drops_the_branch_and_keeps_the_sky():
    tool = _load_tool()
    rgb, raw = _sky_photo_with_branch()
    refined, diag = tool.refine_sky_mask(rgb, raw)
    assert refined.dtype == bool and refined.shape == raw.shape
    assert not refined[60:63, 10:150].any(), "branch pixels must leave the sky label"
    assert not refined[58:65, 10:150].any(), "the 1 px dilation plus the gradient band guards the branch edges"
    assert refined[:50].mean() > 0.98 and refined[75:].mean() > 0.98, "open sky and the cloud gradient stay sky"
    assert 0.8 < diag["kept_fraction"] < 0.97
    assert not (refined & ~raw).any(), "refined label is a subset of the raw label"


def test_refinement_copies_faces_without_enough_sky():
    tool = _load_tool()
    rgb, _ = _sky_photo_with_branch()
    raw = np.zeros(rgb.shape[:2], bool)
    raw[:5, :5] = True  # 25 px, below the reference minimum
    refined, diag = tool.refine_sky_mask(rgb, raw)
    assert np.array_equal(refined, raw) and diag["kept_fraction"] == 1.0


def test_size_mismatch_is_refused():
    tool = _load_tool()
    rgb, raw = _sky_photo_with_branch()
    with pytest.raises(ValueError):
        tool.refine_sky_mask(rgb, raw[:-1])


def _write_source_cache(tmp_path: Path) -> tuple[Path, Path]:
    face_root = tmp_path / "face4_train"
    (face_root / "faces").mkdir(parents=True)
    sky_root = tmp_path / "sky_mask_train"
    (sky_root / "faces").mkdir(parents=True)
    rgb, raw = _sky_photo_with_branch()
    faces = []
    records = []
    for image_id, face_id in (("img_a", "pitch_up_56"), ("img_b", "yaw_plus_35")):
        rgb_rel = f"faces/{image_id}_{face_id}_rgb.png"
        Image.fromarray(rgb, mode="RGB").save(face_root / rgb_rel)
        faces.append({"image_id": image_id, "face_id": face_id, "rgb_path": rgb_rel})
        mask_rel = sky_mask_path_for(image_id, face_id)
        Image.fromarray(np.where(raw, 255, 0).astype(np.uint8), mode="L").save(sky_root / mask_rel)
        valid = int(raw.size)
        records.append(
            {
                "image_id": image_id,
                "camera_id": "left",
                "face_id": face_id,
                "width": int(raw.shape[1]),
                "height": int(raw.shape[0]),
                "mask_path": mask_rel,
                "mask_sha256": hashlib.sha256((sky_root / mask_rel).read_bytes()).hexdigest(),
                "valid_pixels": valid,
                "sky_pixels": int(raw.sum()),
                "sky_fraction": float(raw.sum()) / valid,
            }
        )
    face_sha = "a" * 64
    face_manifest = {
        "face_manifest_sha256": face_sha,
        "split": "train",
        "images": [
            {"image_id": "img_a", "faces": [faces[0]]},
            {"image_id": "img_b", "faces": [faces[1]]},
        ],
    }
    (face_root / "face_manifest.json").write_text(json.dumps(face_manifest), encoding="utf-8")
    manifest = build_sky_mask_manifest(
        split="train",
        source_face_manifest_sha256=face_sha,
        source_identity={"split": "train"},
        model={"id": "m", "revision": "r", "config_sha256": "c" * 64, "weights_sha256": "d" * 64, "license_note": "nc"},
        rule={
            "decision": "p >= 0.5",
            "sky_probability_threshold": 0.5,
            "model_input": "512",
            "upsampling": "bilinear",
            "valid_mask": "face",
            "sky_fraction_denominator": "valid_pixels",
        },
        records=records,
    )
    (sky_root / "sky_mask_train.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sky_root / "sky_mask_train.json", face_root / "face_manifest.json"


def test_cli_writes_a_verifiable_subset_cache(tmp_path: Path):
    tool = _load_tool()
    source_manifest, face_manifest = _write_source_cache(tmp_path)
    out_root = tmp_path / "sky_mask_train_pr"
    assert tool.main([
        "--source-manifest", str(source_manifest),
        "--face-cache-manifest", str(face_manifest),
        "--output-root", str(out_root),
        "--workers", "1",
    ]) == 0
    manifest = load_sky_mask_manifest(out_root / "sky_mask_train.json", expected_face_manifest_sha256="a" * 64)
    assert manifest["rule"]["refinement"]["subset_of_source"] is True
    assert manifest["rule"]["refinement"]["source_sky_mask_manifest_sha256"] == json.loads(
        source_manifest.read_text(encoding="utf-8")
    )["sky_mask_manifest_sha256"]
    source = load_sky_mask_manifest(source_manifest)
    for record in manifest["masks"]:
        refined = load_sky_mask(out_root, record)
        raw = load_sky_mask(source_manifest.parent, next(r for r in source["masks"] if r["image_id"] == record["image_id"]))
        assert not (refined & ~raw).any()
        assert record["raw_sky_pixels"] == int(raw.sum())
        assert record["sky_pixels"] < record["raw_sky_pixels"]
        assert not refined[60:63, 10:150].any()
    stats = json.loads((out_root / "refine_stats.json").read_text(encoding="utf-8"))
    assert stats["faces"] == 2 and 0.8 < stats["kept_fraction_overall"] < 0.97


def test_cli_refuses_a_non_empty_output_root(tmp_path: Path):
    tool = _load_tool()
    source_manifest, face_manifest = _write_source_cache(tmp_path)
    out_root = tmp_path / "occupied"
    out_root.mkdir()
    (out_root / "x").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        tool.main([
            "--source-manifest", str(source_manifest),
            "--face-cache-manifest", str(face_manifest),
            "--output-root", str(out_root),
        ])
