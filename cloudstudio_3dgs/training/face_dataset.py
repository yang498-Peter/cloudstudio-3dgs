"""Training dataset over a prebuilt fisheye face-split cache.

``tools/build_face_cache.py`` warps every training fisheye image onto the
planned pinhole faces (``cloudstudio_3dgs.geometry.fisheye_faces``) and stores
the results as PNG/NPZ artifacts plus a signed ``face_manifest.json``.
:class:`FaceCacheDataset` replays that cache as ``TrainingSample`` items with
``camera_model="pinhole"`` so the trainer renders through gsplat's distortion
free path at full resolution.

Sample-ID contract (load-bearing for the trainer):

    sample.image_id == f"{base_image_id}::{face_id}"

``base_image_id`` is the source fisheye image ID from the dataset manifest and
never contains ``"::"`` itself, so ``exposure_id_for`` (== ``rsplit("::", 1)[0]``)
recovers it. All faces of one fisheye exposure share the physical shutter, so
exposure compensation must allocate ONE gain per base image ID: feed
``exposure_image_ids`` (unique base IDs) to the compensator and map each face
sample back through ``exposure_id_for(sample.image_id)`` when looking up its
gain. ``camera_id_by_image`` is likewise keyed by base image ID.

Seam-fusion note: the cached mask already bakes the hard acceptance test
(warped source mask AND face FoV validity AND ``face_weight`` > the manifest's
``min_face_weight``). The continuous :func:`~cloudstudio_3dgs.geometry.\
fisheye_faces.face_weight` ramp is deliberately NOT cached; consumers that
blend faces recompute it from the ``FaceSpec`` geometry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cloudstudio_3dgs.data.depth_cache import load_sparse_depth
from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.data.mono_depth import (
    resize_to_source_grid,
    verify_mono_depth_manifest,
)
from cloudstudio_3dgs.geometry.fisheye_faces import FaceSpec
from cloudstudio_3dgs.training.dataset import TrainingSample


FACE_MANIFEST_NAME = "face_manifest.json"
FACE_CACHE_SCHEMA_VERSION = 1
SAMPLE_ID_SEPARATOR = "::"

__all__ = [
    "FACE_MANIFEST_NAME",
    "FACE_CACHE_SCHEMA_VERSION",
    "SAMPLE_ID_SEPARATOR",
    "FaceCacheDataset",
    "sign_face_manifest",
    "verify_face_manifest",
]


def sign_face_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with its canonical-JSON SHA256 appended."""
    unsigned = dict(payload)
    unsigned.pop("face_manifest_sha256", None)
    signed = dict(unsigned)
    signed["face_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    return signed


def verify_face_manifest(manifest: dict[str, Any]) -> str:
    """Fail-closed structural + signature check; returns the manifest SHA256."""
    expected = str(manifest.get("face_manifest_sha256", ""))
    if not expected:
        raise ValueError("face manifest has no face_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("face_manifest_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != expected:
        raise ValueError(
            f"face manifest SHA256 mismatch: expected {expected}, computed {actual}"
        )
    if int(manifest.get("schema_version", -1)) != FACE_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported face manifest schema_version")
    if manifest.get("kind") != "fisheye_face_cache":
        raise ValueError("manifest is not a fisheye face cache manifest")
    images = manifest.get("images", [])
    image_ids = [str(record.get("image_id", "")) for record in images]
    if not images or len(image_ids) != len(set(image_ids)) or not all(image_ids):
        raise ValueError("face manifest contains invalid image IDs")
    if any(SAMPLE_ID_SEPARATOR in image_id for image_id in image_ids):
        raise ValueError(
            f"base image IDs must not contain {SAMPLE_ID_SEPARATOR!r}"
        )
    if not isinstance(manifest.get("cameras"), dict) or not manifest["cameras"]:
        raise ValueError("face manifest has no cameras section")
    return actual


def _safe_artifact(root: Path, value: str) -> Path:
    if "\\" in value:
        raise ValueError(f"artifact paths must use forward slashes: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*pure.parts)).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"artifact path escapes its root: {value!r}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FaceCacheDataset:
    """Iterate a fisheye face-split cache as pinhole ``TrainingSample`` items.

    Fail-closed: the manifest signature is verified at construction, cached
    artifacts are (by default) SHA256-verified on first access, a missing file
    raises immediately, and face entries whose cached supervision mask is
    empty are filtered out at construction (counted in
    ``filtered_empty_mask_count``).
    """

    def __init__(
        self,
        face_manifest_path: Path,
        cache_root: Path,
        *,
        verify_artifacts: bool = True,
        dataset_manifest_path: Path | None = None,
        tile_views: list[dict[str, Any]] | None = None,
        renderer_mask_manifest_path: Path | None = None,
        mono_depth_manifest_path: Path | None = None,
        mono_depth_root: Path | None = None,
        face_lidar_geometry_manifest_path: Path | None = None,
        face_lidar_geometry_root: Path | None = None,
        mesh_geometry_manifest_path: Path | None = None,
        mesh_geometry_root: Path | None = None,
    ) -> None:
        if (mono_depth_manifest_path is None) != (mono_depth_root is None):
            raise ValueError(
                "mono_depth_manifest_path and mono_depth_root must be provided together"
            )
        if (face_lidar_geometry_manifest_path is None) != (
            face_lidar_geometry_root is None
        ):
            raise ValueError(
                "face_lidar_geometry_manifest_path and root must be provided together"
            )
        if (mesh_geometry_manifest_path is None) != (mesh_geometry_root is None):
            raise ValueError(
                "mesh_geometry_manifest_path and mesh_geometry_root must be provided together"
            )
        face_manifest_path = Path(face_manifest_path)
        self.manifest = json.loads(face_manifest_path.read_text(encoding="utf-8"))
        self.face_manifest_sha256 = verify_face_manifest(self.manifest)
        self.cache_root = Path(cache_root)
        self.verify_artifacts = verify_artifacts
        self._verified_paths: set[Path] = set()
        # Sensor pixel coordinates (PPISP vignetting lives on the fisheye
        # sensor, not on the warped face) need the source camera intrinsics;
        # supplied optionally via the base dataset manifest.
        self._sensor_cameras: dict[str, dict[str, Any]] = {}
        self._sensor_coords_cache: dict[tuple[str, str], np.ndarray] = {}
        if dataset_manifest_path is not None:
            base = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
            for camera in base.get("cameras", []):
                intrinsic = camera["intrinsic"]
                params = camera["distortion"]["params"]
                self._sensor_cameras[str(camera["camera_id"])] = {
                    "K": np.array(
                        [
                            [intrinsic["fl_x"], 0.0, intrinsic["cx"]],
                            [0.0, intrinsic["fl_y"], intrinsic["cy"]],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    ),
                    "radial": np.array(
                        [params["k1"], params["k2"], params["k3"], params["k4"]],
                        dtype=np.float64,
                    ),
                    "resolution": (int(camera["width"]), int(camera["height"])),
                }
        # depth_to_range_scale is pure face geometry: cache one float32 map per
        # (camera_id, face_id) for the whole process lifetime.
        self._range_scale_cache: dict[tuple[str, str], np.ndarray] = {}
        self.renderer_mask_manifest = None
        self.renderer_mask_manifest_sha256 = None
        self._renderer_mask_by_face: dict[tuple[str, str], dict[str, Any]] = {}
        if renderer_mask_manifest_path is not None:
            # Local import avoids the module cycle: renderer_masks reuses the
            # Face4 manifest verifier from this module.
            from cloudstudio_3dgs.data.renderer_masks import (
                verify_renderer_mask_manifest,
            )

            self.renderer_mask_manifest = json.loads(
                Path(renderer_mask_manifest_path).read_text(encoding="utf-8")
            )
            self.renderer_mask_manifest_sha256 = verify_renderer_mask_manifest(
                self.renderer_mask_manifest
            )
            if (
                self.renderer_mask_manifest.get("source_face_manifest_sha256")
                != self.face_manifest_sha256
            ):
                raise ValueError(
                    "renderer mask manifest is bound to a different Face4 cache"
                )
            if self.renderer_mask_manifest.get("split") != self.manifest.get("split"):
                raise ValueError("renderer mask and Face4 manifests use different splits")
            self._renderer_mask_by_face = {
                (str(record["image_id"]), str(record["face_id"])): record
                for record in self.renderer_mask_manifest.get("masks", [])
            }
            if len(self._renderer_mask_by_face) != len(
                self.renderer_mask_manifest.get("masks", [])
            ):
                raise ValueError("renderer mask manifest contains duplicate samples")
        self.mono_depth_manifest = None
        self.mono_depth_manifest_sha256 = None
        self.mono_depth_root = (
            None if mono_depth_root is None else Path(mono_depth_root)
        )
        self._mono_by_sample: dict[str, dict[str, Any]] = {}
        if mono_depth_manifest_path is not None:
            self.mono_depth_manifest = json.loads(
                Path(mono_depth_manifest_path).read_text(encoding="utf-8")
            )
            self.mono_depth_manifest_sha256 = verify_mono_depth_manifest(
                self.mono_depth_manifest
            )
            if self.mono_depth_manifest.get("source_face_manifest_sha256") != self.face_manifest_sha256:
                raise ValueError("DA2 manifest is bound to a different Face4 cache")
            if self.mono_depth_manifest.get("split") != self.manifest.get("split"):
                raise ValueError("DA2 and Face4 manifests use different splits")
            if self.mono_depth_manifest.get("complete_face_cache") is not True:
                raise ValueError("DA2 manifest is incomplete")
            self._mono_by_sample = {
                str(record["sample_id"]): record
                for record in self.mono_depth_manifest.get("records", [])
            }
            if len(self._mono_by_sample) != len(
                self.mono_depth_manifest.get("records", [])
            ):
                raise ValueError("DA2 manifest contains duplicate sample IDs")
        self.face_lidar_geometry_manifest_sha256 = None
        self.face_lidar_geometry_root = (
            None
            if face_lidar_geometry_root is None
            else Path(face_lidar_geometry_root)
        )
        self._lidar_geometry_by_sample: dict[str, dict[str, Any]] = {}
        if face_lidar_geometry_manifest_path is not None:
            from cloudstudio_3dgs.data.face_lidar_geometry import (
                verify_face_lidar_geometry_manifest,
            )

            lidar_geometry = json.loads(
                Path(face_lidar_geometry_manifest_path).read_text(encoding="utf-8")
            )
            self.face_lidar_geometry_manifest_sha256 = (
                verify_face_lidar_geometry_manifest(lidar_geometry)
            )
            if (
                lidar_geometry.get("source_face_manifest_sha256")
                != self.face_manifest_sha256
            ):
                raise ValueError(
                    "Face4 LiDAR geometry is bound to a different RGB cache"
                )
            if lidar_geometry.get("split") != self.manifest.get("split"):
                raise ValueError("Face4 LiDAR geometry and RGB cache use different splits")
            self._lidar_geometry_by_sample = {
                str(record["sample_id"]): record
                for record in lidar_geometry["records"]
            }
            assert self.face_lidar_geometry_root is not None
            missing_geometry: list[str] = []
            for record in lidar_geometry["records"]:
                relative = record.get("path")
                if relative is None:
                    continue
                artifact = _safe_artifact(
                    self.face_lidar_geometry_root, str(relative)
                )
                if not artifact.is_file():
                    missing_geometry.append(str(record["sample_id"]))
            if missing_geometry:
                raise FileNotFoundError(
                    "Face4 LiDAR geometry has missing independent artifacts: "
                    f"{missing_geometry[:4]}"
                )
        self.mesh_geometry_manifest_sha256 = None
        self.mesh_geometry_root = (
            None if mesh_geometry_root is None else Path(mesh_geometry_root)
        )
        self._mesh_geometry_by_sample: dict[str, dict[str, Any]] = {}
        if mesh_geometry_manifest_path is not None:
            from cloudstudio_3dgs.data.mesh_geometry import (
                verify_mesh_geometry_manifest,
            )

            mesh_geometry = json.loads(
                Path(mesh_geometry_manifest_path).read_text(encoding="utf-8")
            )
            self.mesh_geometry_manifest_sha256 = verify_mesh_geometry_manifest(
                mesh_geometry
            )
            if (
                mesh_geometry.get("source_face_manifest_sha256")
                != self.face_manifest_sha256
            ):
                raise ValueError("mesh geometry is bound to a different Face4 cache")
            if mesh_geometry.get("split") != self.manifest.get("split"):
                raise ValueError("mesh geometry and Face4 manifests use different splits")
            self._mesh_geometry_by_sample = {
                str(record["sample_id"]): record
                for record in mesh_geometry.get("records", [])
            }
            if len(self._mesh_geometry_by_sample) != len(
                mesh_geometry.get("records", [])
            ):
                raise ValueError("mesh geometry manifest contains duplicate sample IDs")

        tile_by_sample: dict[str, dict[str, int]] | None = None
        if tile_views is not None:
            tile_by_sample = {}
            for raw in tile_views:
                sample_id = str(raw.get("sample_id", ""))
                if not sample_id or sample_id in tile_by_sample:
                    raise ValueError("Tile views contain missing or duplicate sample IDs")
                crop = {
                    key: int(raw[key])
                    for key in ("x", "y", "width", "height")
                }
                if min(crop.values()) < 0 or crop["width"] <= 0 or crop["height"] <= 0:
                    raise ValueError(f"Tile crop is invalid for {sample_id}")
                tile_by_sample[sample_id] = crop

        self._faces: dict[tuple[str, str], FaceSpec] = {}
        for camera_id, camera_entry in self.manifest["cameras"].items():
            for payload in camera_entry["faces"]:
                spec = FaceSpec.from_dict(payload)
                key = (str(camera_id), spec.face_id)
                if key in self._faces:
                    raise ValueError(f"duplicate face {key} in manifest")
                self._faces[key] = spec

        self.filtered_empty_mask_count = 0
        self._samples: list[
            tuple[dict[str, Any], dict[str, Any], dict[str, int] | None]
        ] = []
        accepted_tile_samples: set[str] = set()
        for image_record in self.manifest["images"]:
            camera_id = str(image_record["camera_id"])
            c2w = np.asarray(image_record["c2w"], dtype=np.float64)
            if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
                raise ValueError(
                    f"image {image_record['image_id']} has an invalid c2w"
                )
            for face_entry in image_record["faces"]:
                face_id = str(face_entry["face_id"])
                # Tile planning and Trainer sampling use the public
                # ``base::face`` identity.  DA2 cache filenames deliberately
                # retain ``base__face`` below; the two namespaces must not be
                # mixed or every real Tile view is rejected as unknown.
                manifest_sample_id = (
                    f"{image_record['image_id']}{SAMPLE_ID_SEPARATOR}{face_id}"
                )
                if tile_by_sample is not None and manifest_sample_id not in tile_by_sample:
                    continue
                if (camera_id, face_id) not in self._faces:
                    raise ValueError(
                        f"face entry {face_id!r} references an unplanned face "
                        f"for camera {camera_id!r}"
                    )
                renderer_record = self._renderer_mask_by_face.get(
                    (str(image_record["image_id"]), face_id)
                )
                if self.renderer_mask_manifest is not None:
                    if renderer_record is None:
                        raise ValueError(
                            f"renderer mask manifest does not cover {manifest_sample_id}"
                        )
                    if (
                        str(renderer_record["mask_path"])
                        != str(face_entry["mask_path"])
                        or str(renderer_record["mask_sha256"])
                        != str(face_entry["mask_sha256"])
                        or int(renderer_record["keep_pixels"])
                        != int(face_entry.get("mask_true_pixels", -1))
                    ):
                        raise ValueError(
                            f"renderer mask record differs from Face4 for {manifest_sample_id}"
                        )
                if int(face_entry.get("mask_true_pixels", -1)) == 0:
                    self.filtered_empty_mask_count += 1
                    continue
                crop = None if tile_by_sample is None else tile_by_sample[manifest_sample_id]
                if crop is not None:
                    face = self._faces[(camera_id, face_id)]
                    if (
                        crop["x"] + crop["width"] > face.width
                        or crop["y"] + crop["height"] > face.height
                    ):
                        raise ValueError(f"Tile crop exceeds Face4 bounds for {manifest_sample_id}")
                    accepted_tile_samples.add(manifest_sample_id)
                self._samples.append((image_record, face_entry, crop))
        if tile_by_sample is not None and accepted_tile_samples != set(tile_by_sample):
            missing = sorted(set(tile_by_sample) - accepted_tile_samples)
            raise ValueError(f"Tile views reference unknown Face4 samples: {missing[:4]}")
        if self.mono_depth_manifest is not None:
            selected_mono_ids = {
                f"{record['image_id']}__{entry['face_id']}"
                for record, entry, _crop in self._samples
            }
            missing_mono = sorted(selected_mono_ids - set(self._mono_by_sample))
            if missing_mono:
                raise ValueError(
                    f"DA2 manifest does not cover selected Tile views: {missing_mono[:4]}"
                )
        if self._lidar_geometry_by_sample:
            selected_ids = {
                f"{record['image_id']}{SAMPLE_ID_SEPARATOR}{entry['face_id']}"
                for record, entry, _crop in self._samples
            }
            missing_lidar = sorted(
                selected_ids - set(self._lidar_geometry_by_sample)
            )
            if missing_lidar:
                raise ValueError(
                    "Face4 LiDAR geometry does not cover selected views: "
                    f"{missing_lidar[:4]}"
                )
        if self._mesh_geometry_by_sample:
            selected_ids = {
                f"{record['image_id']}{SAMPLE_ID_SEPARATOR}{entry['face_id']}"
                for record, entry, _crop in self._samples
            }
            missing_mesh = sorted(selected_ids - set(self._mesh_geometry_by_sample))
            if missing_mesh:
                raise ValueError(
                    "mesh geometry does not cover selected Tile views: "
                    f"{missing_mesh[:4]}"
                )
        if not self._samples:
            raise ValueError("face cache contains no usable face samples")

    # ------------------------------------------------------------------ ids --

    @property
    def image_ids(self) -> list[str]:
        """All face-sample IDs (``base::face``) in manifest (split) order."""
        return [
            f"{record['image_id']}{SAMPLE_ID_SEPARATOR}{entry['face_id']}"
            for record, entry, _crop in self._samples
        ]

    @property
    def exposure_image_ids(self) -> list[str]:
        """Unique base fisheye image IDs, first-seen (split) order.

        One exposure gain per entry; every face of the same base image shares
        that gain (see the module docstring's ``::`` contract).
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for record, _entry, _crop in self._samples:
            image_id = str(record["image_id"])
            if image_id not in seen:
                seen.add(image_id)
                ordered.append(image_id)
        return ordered

    @property
    def camera_id_by_image(self) -> dict[str, str]:
        """Physical camera for each base image ID (exposure anchor groups)."""
        return {
            str(record["image_id"]): str(record["camera_id"])
            for record, _entry, _crop in self._samples
        }

    @staticmethod
    def exposure_id_for(sample_image_id: str) -> str:
        """Map a face-sample ID (``base::face``) back to its base image ID."""
        return str(sample_image_id).rsplit(SAMPLE_ID_SEPARATOR, 1)[0]

    @property
    def rig_frame_ids(self) -> tuple[str, ...]:
        """Selected Rig Frames once, first-seen (split) order.

        Pose refinement owns one correction per Rig Frame, shared by both
        cameras and by every face of both images: faces are rotations about
        the same camera center, so a world-side rig correction is exactly as
        valid for them as for the source fisheye frame.
        """
        result: list[str] = []
        seen: set[str] = set()
        for record, _entry, _crop in self._samples:
            rig_frame_id = str(record.get("rig_frame_id") or "")
            if not rig_frame_id:
                raise ValueError(f"image {record['image_id']} has no Rig Frame")
            if rig_frame_id not in seen:
                result.append(rig_frame_id)
                seen.add(rig_frame_id)
        return tuple(result)

    def indices_for_rig_frames(self, maximum_rig_frames: int) -> tuple[int, ...]:
        if maximum_rig_frames <= 0:
            raise ValueError("maximum_rig_frames must be positive")
        rig_frame_ids = self.rig_frame_ids
        if len(rig_frame_ids) <= maximum_rig_frames:
            selected = set(rig_frame_ids)
        else:
            positions = np.linspace(
                0, len(rig_frame_ids) - 1, maximum_rig_frames, dtype=np.int64
            )
            selected = {rig_frame_ids[int(index)] for index in positions}
        return tuple(
            index
            for index, (record, _entry, _crop) in enumerate(self._samples)
            if str(record.get("rig_frame_id") or "") in selected
        )

    def rig_frame_centers(self) -> dict[str, np.ndarray]:
        """Mean BASE camera center per Rig Frame, as the rotation pivot.

        Deduplicated by image ID first: the two cameras of a frame can cache
        different face counts, and per-sample averaging would drag the pivot
        toward whichever camera has more faces.
        """
        grouped: dict[str, dict[str, np.ndarray]] = {}
        for record, _entry, _crop in self._samples:
            rig_frame_id = str(record.get("rig_frame_id") or "")
            c2w = np.asarray(record.get("c2w"), dtype=np.float64)
            if not rig_frame_id or c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
                raise ValueError(f"image {record['image_id']} has no valid Rig pose")
            grouped.setdefault(rig_frame_id, {})[str(record["image_id"])] = c2w[:3, 3]
        if set(grouped) != set(self.rig_frame_ids):
            raise ValueError("face split Rig Frame centers are incomplete")
        return {
            rig_frame_id: np.mean(np.stack(list(centers.values())), axis=0)
            for rig_frame_id, centers in grouped.items()
        }

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "face_manifest_sha256": self.face_manifest_sha256,
            "face_plan": self.manifest.get("face_plan", "adaptive_full_fov"),
            "source_identity": self.manifest.get("source_identity"),
            "fov_deg": self.manifest.get("fov_deg"),
            "split": self.manifest.get("split"),
            "min_face_weight": self.manifest.get("min_face_weight"),
            "mono_depth_manifest_sha256": self.mono_depth_manifest_sha256,
            "renderer_mask_manifest_sha256": self.renderer_mask_manifest_sha256,
            "face_lidar_geometry_manifest_sha256": (
                self.face_lidar_geometry_manifest_sha256
            ),
            "tile_cropped": any(crop is not None for _, _, crop in self._samples),
        }

    @property
    def dataset_sha256(self) -> str:
        """The BASE dataset identity, carried through the face cache.

        The trainer keys the coordinate transform manifest and the train/val
        identity check on this value; faces are a resampling of the same
        capture, so the base identity is the correct answer.
        """
        source = self.manifest.get("source_identity") or {}
        value = source.get("dataset_manifest_sha256")
        if not value:
            raise ValueError("face manifest is missing the source dataset identity")
        return str(value)

    def _source_sha(self, key: str) -> str | None:
        source = self.manifest.get("source_identity") or {}
        value = source.get(key)
        return None if value is None else str(value)

    # Identity passthroughs for the run-manifest fields the trainer records.
    @property
    def mask_sha256(self) -> str | None:
        return self._source_sha("mask_manifest_sha256")

    @property
    def person_mask_sha256(self) -> str | None:
        return self._source_sha("person_mask_manifest_sha256")

    @property
    def split_sha256(self) -> str | None:
        return self._source_sha("split_manifest_sha256")

    @property
    def depth_sha256(self) -> str | None:
        return self._source_sha("depth_manifest_sha256")

    # ---------------------------------------------------------------- access --

    def __len__(self) -> int:
        return len(self._samples)

    def _verify(self, path: Path, expected: str, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        if not self.verify_artifacts or path in self._verified_paths:
            return
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected}, computed {actual}"
            )
        self._verified_paths.add(path)

    def _sensor_pixel_coords(self, camera_id: str, face: FaceSpec) -> np.ndarray | None:
        """Absolute fisheye pixel-center coordinates for every face pixel."""
        if camera_id not in self._sensor_cameras:
            return None
        key = (camera_id, str(face.face_id))
        cached = self._sensor_coords_cache.get(key)
        if cached is None:
            from cloudstudio_3dgs.data.face_warp import build_face_warp_grid

            camera = self._sensor_cameras[camera_id]
            grid = build_face_warp_grid(camera["K"], camera["radial"], face)
            # The grid stores array-index coordinates; +0.5 restores the
            # pixel-center convention PPISP expects.
            cached = np.stack(
                [grid.u + 0.5, grid.v + 0.5], axis=-1
            ).astype(np.float32)
            self._sensor_coords_cache[key] = cached
        return cached

    def _depth_to_range_scale(self, camera_id: str, face: FaceSpec) -> np.ndarray:
        key = (camera_id, face.face_id)
        cached = self._range_scale_cache.get(key)
        if cached is None:
            fx = float(face.K_face[0, 0])
            fy = float(face.K_face[1, 1])
            cx = float(face.K_face[0, 2])
            cy = float(face.K_face[1, 2])
            jj, ii = np.meshgrid(
                np.arange(face.width, dtype=np.float64) + 0.5,
                np.arange(face.height, dtype=np.float64) + 0.5,
            )
            x = (jj - cx) / fx
            y = (ii - cy) / fy
            # ||K_face^-1 [u+0.5, v+0.5, 1]||: z-depth -> Euclidean ray range.
            cached = np.sqrt(1.0 + x * x + y * y).astype(np.float32)
            self._range_scale_cache[key] = cached
        return cached

    def __getitem__(self, index: int) -> TrainingSample:
        image_record, face_entry, crop = self._samples[index]
        base_image_id = str(image_record["image_id"])
        camera_id = str(image_record["camera_id"])
        face_id = str(face_entry["face_id"])
        face = self._faces[(camera_id, face_id)]

        rgb_path = _safe_artifact(self.cache_root, str(face_entry["rgb_path"]))
        renderer_record = self._renderer_mask_by_face.get((base_image_id, face_id))
        mask_entry = face_entry if renderer_record is None else renderer_record
        mask_path = _safe_artifact(self.cache_root, str(mask_entry["mask_path"]))
        self._verify(rgb_path, str(face_entry["rgb_sha256"]), "face rgb")
        self._verify(mask_path, str(mask_entry["mask_sha256"]), "renderer mask")

        with Image.open(rgb_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as source:
            rgb_mask = np.asarray(source.convert("L"), dtype=np.uint8) > 0
        expected_shape = (face.height, face.width)
        if image.shape[:2] != expected_shape or rgb_mask.shape != expected_shape:
            raise ValueError(
                f"face artifact shape mismatch for {base_image_id}/{face_id}: "
                f"expected {expected_shape}"
            )
        if not np.any(rgb_mask):
            raise ValueError(
                f"face {base_image_id}/{face_id} has an empty supervision mask"
            )

        depth_range = None
        depth_confidence = None
        depth_mask = None
        depth_cache_path = None
        lidar_record = self._lidar_geometry_by_sample.get(
            f"{base_image_id}{SAMPLE_ID_SEPARATOR}{face_id}"
        )
        depth_entry = face_entry if lidar_record is None else lidar_record
        depth_root = (
            self.cache_root
            if lidar_record is None
            else self.face_lidar_geometry_root
        )
        if depth_entry.get("path") or depth_entry.get("depth_path"):
            relative_depth_path = depth_entry.get("path", depth_entry.get("depth_path"))
            expected_depth_sha = depth_entry.get(
                "sha256", depth_entry.get("depth_sha256")
            )
            assert depth_root is not None
            depth_cache_path = _safe_artifact(
                depth_root, str(relative_depth_path)
            )
            self._verify(
                depth_cache_path, str(expected_depth_sha), "face depth"
            )
            sparse = load_sparse_depth(depth_cache_path)
            if tuple(sparse.shape) != expected_shape:
                raise ValueError(
                    f"face depth shape mismatch for {base_image_id}/{face_id}"
                )
            depth_range, depth_confidence, depth_valid = sparse.to_dense()
            depth_mask = rgb_mask & depth_valid

        mono_depth_range = None
        mono_depth_mask = None
        mono_depth_cache_path = None
        mono_record = self._mono_by_sample.get(f"{base_image_id}__{face_id}")
        if mono_record is not None and bool(mono_record.get("alignment", {}).get("valid")):
            assert self.mono_depth_root is not None
            mono_depth_cache_path = _safe_artifact(
                self.mono_depth_root, str(mono_record["path"])
            )
            self._verify(
                mono_depth_cache_path,
                str(mono_record["sha256"]),
                "DA2 relative depth",
            )
            with np.load(mono_depth_cache_path, allow_pickle=False) as payload:
                if "relative_depth" not in payload:
                    raise ValueError("DA2 cache has no relative_depth array")
                relative = np.asarray(payload["relative_depth"], dtype=np.float32)
            relative_full = resize_to_source_grid(
                relative, (face.height, face.width)
            )
            alignment = mono_record["alignment"]
            mono_depth_range = (
                float(alignment["scale"]) * relative_full
                + float(alignment["shift"])
            ).astype(np.float32)
            mono_depth_mask = (
                rgb_mask
                & np.isfinite(mono_depth_range)
                & (mono_depth_range > 0.0)
            )

        mesh_depth_range = None
        mesh_normal_camera = None
        mesh_confidence = None
        mesh_depth_valid = None
        mesh_geometry_cache_path = None
        mesh_record = self._mesh_geometry_by_sample.get(
            f"{base_image_id}{SAMPLE_ID_SEPARATOR}{face_id}"
        )
        if mesh_record is not None:
            if crop is None or dict(mesh_record.get("crop", {})) != dict(crop):
                raise ValueError(
                    f"mesh geometry crop does not match selected Tile crop for "
                    f"{base_image_id}/{face_id}"
                )
            assert self.mesh_geometry_root is not None
            mesh_geometry_cache_path = _safe_artifact(
                self.mesh_geometry_root, str(mesh_record["path"])
            )
            self._verify(
                mesh_geometry_cache_path,
                str(mesh_record["sha256"]),
                "mesh geometry",
            )
            with np.load(mesh_geometry_cache_path, allow_pickle=False) as payload:
                required = {
                    "depth_range_m", "normal_camera", "confidence", "valid", "shape"
                }
                if not required.issubset(payload.files):
                    raise ValueError("mesh geometry cache is missing required arrays")
                mesh_depth_range = np.asarray(payload["depth_range_m"], np.float32)
                mesh_normal_camera = np.asarray(payload["normal_camera"], np.float32)
                mesh_confidence = np.asarray(payload["confidence"], np.float32)
                mesh_depth_valid = np.asarray(payload["valid"], bool)
            mesh_shape = (int(crop["height"]), int(crop["width"]))
            if mesh_depth_range.shape != mesh_shape:
                raise ValueError("mesh geometry depth shape differs from Tile crop")
            if mesh_normal_camera.shape != (*mesh_shape, 3):
                raise ValueError("mesh geometry normal shape differs from Tile crop")
            if mesh_confidence.shape != mesh_shape or mesh_depth_valid.shape != mesh_shape:
                raise ValueError("mesh geometry mask/confidence shape differs from Tile crop")

        c2w_base = np.asarray(image_record["c2w"], dtype=np.float64)
        face_to_base = np.eye(4, dtype=np.float64)
        face_to_base[:3, :3] = face.R_face
        c2w = (c2w_base @ face_to_base).astype(np.float32)

        K = face.K_face.astype(np.float32).copy()
        range_scale = self._depth_to_range_scale(camera_id, face)
        sensor_coords = self._sensor_pixel_coords(camera_id, face)
        if crop is not None:
            x = crop["x"]
            y = crop["y"]
            right = x + crop["width"]
            bottom = y + crop["height"]
            image = np.ascontiguousarray(image[y:bottom, x:right])
            rgb_mask = np.ascontiguousarray(rgb_mask[y:bottom, x:right])
            depth_range = None if depth_range is None else np.ascontiguousarray(depth_range[y:bottom, x:right])
            depth_confidence = None if depth_confidence is None else np.ascontiguousarray(depth_confidence[y:bottom, x:right])
            depth_mask = None if depth_mask is None else np.ascontiguousarray(depth_mask[y:bottom, x:right])
            mono_depth_range = None if mono_depth_range is None else np.ascontiguousarray(mono_depth_range[y:bottom, x:right])
            mono_depth_mask = None if mono_depth_mask is None else np.ascontiguousarray(mono_depth_mask[y:bottom, x:right])
            range_scale = np.ascontiguousarray(range_scale[y:bottom, x:right])
            sensor_coords = None if sensor_coords is None else np.ascontiguousarray(sensor_coords[y:bottom, x:right])
            K[0, 2] -= float(x)
            K[1, 2] -= float(y)
        mesh_depth_mask = None
        if mesh_depth_range is not None:
            assert mesh_depth_valid is not None and mesh_confidence is not None
            mesh_depth_mask = (
                rgb_mask
                & mesh_depth_valid
                & np.isfinite(mesh_depth_range)
                & (mesh_depth_range > 0.0)
                & np.isfinite(mesh_confidence)
                & (mesh_confidence > 0.0)
            )
        if not np.any(rgb_mask):
            raise ValueError(f"cropped face {base_image_id}/{face_id} has an empty mask")

        return TrainingSample(
            image_id=f"{base_image_id}{SAMPLE_ID_SEPARATOR}{face_id}",
            rig_frame_id=str(image_record["rig_frame_id"]),
            camera_id=camera_id,
            image=image,
            rgb_mask=rgb_mask,
            depth_range_m=depth_range,
            depth_confidence=depth_confidence,
            depth_mask=depth_mask,
            depth_cache_path=depth_cache_path,
            c2w=c2w,
            K=K,
            radial_coeffs=np.zeros(4, dtype=np.float32),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            camera_model="pinhole",
            depth_to_range_scale=range_scale,
            sensor_pixel_coords=sensor_coords,
            sensor_resolution=(
                self._sensor_cameras[camera_id]["resolution"]
                if camera_id in self._sensor_cameras
                else None
            ),
            mono_depth_range_m=mono_depth_range,
            mono_depth_mask=mono_depth_mask,
            mono_depth_cache_path=mono_depth_cache_path,
            mesh_depth_range_m=mesh_depth_range,
            mesh_normal_camera=mesh_normal_camera,
            mesh_confidence=mesh_confidence,
            mesh_depth_mask=mesh_depth_mask,
            mesh_geometry_cache_path=mesh_geometry_cache_path,
        )

    def camera_sample(self, index: int) -> Any:
        """Camera-only view of one sample: pose, intrinsics, size - no artifact IO.

        Mirrors ``__getitem__``'s camera math exactly (same face rotation and
        the same crop offset applied to the principal point) so an offline
        per-view render sees the training camera bit for bit, while skipping
        the image/mask/depth loading and hash verification that dominate the
        full sample's cost by orders of magnitude.
        """
        from types import SimpleNamespace

        image_record, face_entry, crop = self._samples[index]
        camera_id = str(image_record["camera_id"])
        face_id = str(face_entry["face_id"])
        face = self._faces[(camera_id, face_id)]

        c2w_base = np.asarray(image_record["c2w"], dtype=np.float64)
        face_to_base = np.eye(4, dtype=np.float64)
        face_to_base[:3, :3] = face.R_face
        c2w = (c2w_base @ face_to_base).astype(np.float32)

        K = face.K_face.astype(np.float32).copy()
        width, height = int(face.width), int(face.height)
        if crop is not None:
            K[0, 2] -= float(crop["x"])
            K[1, 2] -= float(crop["y"])
            width, height = int(crop["width"]), int(crop["height"])

        return SimpleNamespace(
            image_id=f"{image_record['image_id']}{SAMPLE_ID_SEPARATOR}{face_id}",
            rig_frame_id=str(image_record["rig_frame_id"]),
            camera_id=camera_id,
            c2w=c2w,
            K=K,
            radial_coeffs=np.zeros(4, dtype=np.float32),
            width=width,
            height=height,
            camera_model="pinhole",
        )
