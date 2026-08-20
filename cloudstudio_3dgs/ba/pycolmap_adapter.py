"""PyCOLMAP fixed-stereo Rig construction, snapshots, and staged BA execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from cloudstudio_3dgs.ba.report import stage_options


def _pycolmap():
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("Rig BA requires the optional pycolmap package") from exc
    return pycolmap


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").removeprefix("camera/")


def build_reference_model_from_manifest(dataset: dict[str, Any]) -> Any:
    """Build a known-pose COLMAP model without renaming manifest image paths."""
    pycolmap = _pycolmap()
    camera_records = dataset.get("cameras")
    image_records = dataset.get("images")
    if not isinstance(camera_records, list) or not camera_records:
        raise ValueError("dataset manifest has no camera records")
    if not isinstance(image_records, list) or not image_records:
        raise ValueError("dataset manifest has no image records")

    output = pycolmap.Reconstruction()
    camera_ids: dict[str, int] = {}
    for numeric_id, record in enumerate(
        sorted(camera_records, key=lambda item: str(item["camera_id"])), start=1
    ):
        stable_id = str(record["camera_id"])
        if stable_id in camera_ids:
            raise ValueError(f"dataset manifest repeats camera ID {stable_id}")
        intrinsic = record.get("intrinsic", {})
        model = str(record.get("distortion", {}).get("camera_model", ""))
        base_params = [
            intrinsic.get("fl_x"),
            intrinsic.get("fl_y"),
            intrinsic.get("cx"),
            intrinsic.get("cy"),
        ]
        if model == "OPENCV_FISHEYE":
            distortion = record.get("distortion", {}).get("params", {})
            params = base_params + [
                distortion.get("k1"),
                distortion.get("k2"),
                distortion.get("k3"),
                distortion.get("k4"),
            ]
        elif model == "PINHOLE":
            params = base_params
        else:
            raise ValueError(f"unsupported manifest camera model: {model}")
        parameters = np.asarray(params, dtype=np.float64)
        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
        if width <= 0 or height <= 0 or not np.all(np.isfinite(parameters)):
            raise ValueError(f"manifest camera {stable_id} has invalid calibration")
        output.add_camera_with_trivial_rig(
            pycolmap.Camera(
                model=model,
                width=width,
                height=height,
                params=parameters,
                camera_id=numeric_id,
            )
        )
        camera_ids[stable_id] = numeric_id

    normalised_names: set[str] = set()
    for numeric_id, record in enumerate(
        sorted(image_records, key=lambda item: _normalise_name(str(item["path"]))),
        start=1,
    ):
        name = _normalise_name(str(record["path"]))
        if not name or name in normalised_names:
            raise ValueError(f"dataset manifest repeats image path {name}")
        normalised_names.add(name)
        camera_id = str(record.get("camera_id", ""))
        if camera_id not in camera_ids:
            raise ValueError(f"manifest image {name} has unknown camera {camera_id}")
        if record.get("pose_convention") != "c2w_opencv":
            raise ValueError(f"manifest image {name} is not c2w_opencv")
        c2w = np.asarray(record.get("c2w"), dtype=np.float64)
        if (
            c2w.shape != (4, 4)
            or not np.all(np.isfinite(c2w))
            or not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
            or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError(f"manifest image {name} has an invalid rigid c2w pose")
        world_to_camera = np.linalg.inv(c2w)
        image = pycolmap.Image(
            name=name,
            camera_id=camera_ids[camera_id],
            image_id=numeric_id,
        )
        output.add_image_with_trivial_frame(
            image, pycolmap.Rigid3d(world_to_camera[:3])
        )
    if output.num_images() != len(image_records):
        raise ValueError("manifest reference model has an unexpected image count")
    return output


def build_training_reference_model(
    reconstruction: Any, image_names: set[str]
) -> Any:
    """Copy exactly the training images and their known poses into a clean model."""
    pycolmap = _pycolmap()
    normalised_names = {_normalise_name(name) for name in image_names}
    if not normalised_names:
        raise ValueError("training reference model has no image names")
    source_images = {
        _normalise_name(image.name): image for image in reconstruction.images.values()
    }
    missing = normalised_names - set(source_images)
    if missing:
        raise ValueError(
            f"reference model is missing training images: {sorted(missing)[:4]}"
        )
    selected = [source_images[name] for name in sorted(normalised_names)]
    if any(not image.has_pose for image in selected):
        raise ValueError("reference model contains a training image without a POS pose")
    output = pycolmap.Reconstruction()
    for camera_id in sorted({int(image.camera_id) for image in selected}):
        output.add_camera_with_trivial_rig(
            pycolmap.Camera(reconstruction.camera(camera_id).todict())
        )
    for image in selected:
        copied = pycolmap.Image(
            name=_normalise_name(image.name),
            camera_id=int(image.camera_id),
            image_id=int(image.image_id),
        )
        output.add_image_with_trivial_frame(
            copied, pycolmap.Rigid3d(image.cam_from_world().matrix())
        )
    if output.num_images() != len(normalised_names):
        raise ValueError("training reference model has an unexpected image count")
    return output


def apply_fixed_stereo_rig(
    reconstruction: Any,
    dataset: dict[str, Any],
    *,
    included_image_ids: set[str] | None = None,
) -> None:
    pycolmap = _pycolmap()
    model_images = {
        _normalise_name(image.name): image for image in reconstruction.images.values()
    }
    dataset_images = {
        str(image["image_id"]): image
        for image in dataset["images"]
        if included_image_ids is None
        or str(image["image_id"]) in included_image_ids
    }
    if included_image_ids is not None:
        unknown = included_image_ids - {
            str(image["image_id"]) for image in dataset["images"]
        }
        if unknown:
            raise ValueError(
                f"fixed Rig selection has unknown image IDs: {sorted(unknown)[:4]}"
            )
    camera_ids_by_side: dict[str, set[int]] = {"left": set(), "right": set()}
    old_poses: dict[int, Any] = {}
    image_by_id: dict[str, Any] = {}
    for image_id, image in dataset_images.items():
        name = _normalise_name(str(image["path"]))
        if name not in model_images:
            raise ValueError(f"COLMAP model is missing dataset image {name}")
        model_image = model_images[name]
        if not model_image.has_pose:
            raise ValueError(f"COLMAP image has no initial POS pose: {name}")
        camera_ids_by_side[str(image["side"])].add(int(model_image.camera_id))
        old_poses[int(model_image.image_id)] = model_image.cam_from_world()
        image_by_id[image_id] = model_image
    if any(len(camera_ids_by_side[side]) != 1 for side in ("left", "right")):
        raise ValueError("fixed stereo BA requires exactly one COLMAP camera per side")
    left_camera_id = next(iter(camera_ids_by_side["left"]))
    right_camera_id = next(iter(camera_ids_by_side["right"]))
    if left_camera_id == right_camera_id:
        raise ValueError("left and right sides cannot share one physical camera")
    left_sensor = reconstruction.camera(left_camera_id).sensor_id
    right_sensor = reconstruction.camera(right_camera_id).sensor_id

    expected_right_to_left = np.asarray(
        dataset["rig"]["expected_right_to_left"], dtype=np.float64
    )
    if expected_right_to_left.shape != (4, 4):
        raise ValueError("dataset Rig has no valid expected_right_to_left transform")
    right_from_left = np.linalg.inv(expected_right_to_left)
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(left_sensor)
    rig.add_sensor(right_sensor, pycolmap.Rigid3d(right_from_left[:3]))

    frames = []
    for image in reconstruction.images.values():
        image.reset_frame_ptr()
    for frame_id, frame_record in enumerate(
        (
            frame
            for frame in sorted(
                dataset["rig_frames"], key=lambda item: int(item["timestamp_ns"])
            )
            if {str(value) for value in frame["image_ids"]} <= set(dataset_images)
        ),
        start=1,
    ):
        left = image_by_id[str(frame_record["left_image_id"])]
        right = image_by_id[str(frame_record["right_image_id"])]
        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = rig.rig_id
        frame.rig_from_world = old_poses[int(left.image_id)]
        frame.add_data_id(pycolmap.data_t(left_sensor, int(left.image_id)))
        frame.add_data_id(pycolmap.data_t(right_sensor, int(right.image_id)))
        frame.finalize_data_ids()
        left.frame_id = frame_id
        right.frame_id = frame_id
        frames.append(frame)
    reconstruction.set_rigs_and_frames([rig], frames)
    selected_by_frame = {
        str(value)
        for frame in dataset["rig_frames"]
        if {str(value) for value in frame["image_ids"]} <= set(dataset_images)
        for value in frame["image_ids"]
    }
    if selected_by_frame != set(dataset_images):
        raise ValueError("fixed Rig selection divides or omits a stereo frame")
    if reconstruction.num_rigs() != 1 or reconstruction.num_frames() != len(frames):
        raise ValueError("PyCOLMAP did not install the fixed stereo Rig contract")


def _camera_record(camera: Any) -> dict[str, float]:
    names = [value.strip() for value in camera.params_info.split(",")]
    values = [float(value) for value in camera.params]
    mapping = dict(zip(names, values))
    aliases = {
        "fl_x": ("fx", "f"),
        "fl_y": ("fy", "f"),
        "cx": ("cx",),
        "cy": ("cy",),
        "k1": ("k1",),
        "k2": ("k2",),
        "k3": ("k3",),
        "k4": ("k4",),
    }
    output: dict[str, float] = {}
    for target, candidates in aliases.items():
        output[target] = next((mapping[name] for name in candidates if name in mapping), 0.0)
    return output


def reconstruction_snapshot(
    reconstruction: Any,
    dataset: dict[str, Any],
    *,
    model_sha256: str,
    solver_success: bool,
    included_image_ids: set[str] | None = None,
) -> dict[str, Any]:
    model_images = {
        _normalise_name(image.name): image for image in reconstruction.images.values()
    }
    image_records = {
        str(image["image_id"]): image for image in dataset["images"]
    }
    allowed_model_image_ids: set[int] | None = None
    if included_image_ids is not None:
        unknown = included_image_ids - set(image_records)
        if unknown:
            raise ValueError(
                f"BA snapshot includes unknown stable image IDs: {sorted(unknown)[:4]}"
            )
        allowed_model_image_ids = {
            int(model_images[_normalise_name(str(image_records[stable_id]["path"]))].image_id)
            for stable_id in included_image_ids
        }
    errors: list[float] = []
    for point in reconstruction.points3D.values():
        for element in point.track.elements:
            if (
                allowed_model_image_ids is not None
                and int(element.image_id) not in allowed_model_image_ids
            ):
                continue
            image = reconstruction.image(element.image_id)
            projected = image.project_point(point.xyz)
            if projected is None:
                continue
            observed = np.asarray(image.point2D(element.point2D_idx).xy, dtype=np.float64)
            errors.append(float(np.linalg.norm(np.asarray(projected) - observed)))
    frames: dict[str, Any] = {}
    camera_by_side: dict[str, Any] = {}
    for frame_record in dataset["rig_frames"]:
        stable_ids = {str(value) for value in frame_record["image_ids"]}
        if included_image_ids is not None and not stable_ids <= included_image_ids:
            continue
        left_record = image_records[str(frame_record["left_image_id"])]
        right_record = image_records[str(frame_record["right_image_id"])]
        left = model_images[_normalise_name(str(left_record["path"]))]
        right = model_images[_normalise_name(str(right_record["path"]))]
        left_from_world = np.eye(4)
        right_from_world = np.eye(4)
        left_from_world[:3] = left.cam_from_world().matrix()
        right_from_world[:3] = right.cam_from_world().matrix()
        frames[str(frame_record["rig_frame_id"])] = {
            "timestamp_ns": int(frame_record["timestamp_ns"]),
            "center_m": (
                0.5 * (left.projection_center() + right.projection_center())
            ).tolist(),
            "right_to_left": (
                left_from_world @ np.linalg.inv(right_from_world)
            ).tolist(),
        }
        camera_by_side["left"] = left.camera
        camera_by_side["right"] = right.camera
    return {
        "model_sha256": model_sha256,
        "solver_success": solver_success,
        "reprojection_errors_px": errors,
        "rig_frames": frames,
        "cameras": {
            side: _camera_record(camera) for side, camera in sorted(camera_by_side.items())
        },
    }


def run_bundle_adjustment_stage(
    reconstruction: Any,
    stage: str,
    *,
    image_ids: set[int] | None = None,
    position_priors_by_image_id: dict[int, np.ndarray] | None = None,
    position_prior_stddev_m: float = 0.05,
) -> Any:
    pycolmap = _pycolmap()
    settings = stage_options(stage)
    options = pycolmap.BundleAdjustmentOptions()
    for key, value in settings.items():
        setattr(options, key, value)
    options.refine_points3D = True
    options.print_summary = False
    config = pycolmap.BundleAdjustmentConfig()
    selected_image_ids = (
        sorted(image_ids) if image_ids is not None else reconstruction.reg_image_ids()
    )
    if not selected_image_ids:
        raise ValueError("BA stage has no selected training images")
    if not np.isfinite(position_prior_stddev_m) or position_prior_stddev_m <= 0.0:
        raise ValueError("position_prior_stddev_m must be finite and positive")
    for image_id in selected_image_ids:
        config.add_image(image_id)
    first_frame_id = min(int(reconstruction.image(image_id).frame_id) for image_id in selected_image_ids)
    config.set_constant_rig_from_world_pose(first_frame_id)
    for rig in reconstruction.rigs.values():
        for sensor_id in rig.sensor_ids():
            config.set_constant_sensor_from_rig_pose(sensor_id)
    if position_priors_by_image_id is None:
        adjuster = pycolmap.create_default_bundle_adjuster(
            options, config, reconstruction
        )
    else:
        missing = set(selected_image_ids) - set(position_priors_by_image_id)
        if missing:
            raise ValueError(
                f"POS priors are missing selected image IDs: {sorted(missing)[:4]}"
            )
        covariance = np.eye(3, dtype=np.float64) * position_prior_stddev_m**2
        priors = []
        for image_id in selected_image_ids:
            position = np.asarray(
                position_priors_by_image_id[image_id], dtype=np.float64
            )
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"POS prior for image {image_id} is not a finite xyz")
            image = reconstruction.image(image_id)
            priors.append(
                pycolmap.PosePrior(
                    corr_data_id=pycolmap.data_t(
                        image.camera.sensor_id, int(image_id)
                    ),
                    position=position,
                    position_covariance=covariance,
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                )
            )
        prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
        prior_options.prior_position_fallback_stddev = position_prior_stddev_m
        fixed_sensor_poses = []
        for rig in reconstruction.rigs.values():
            for sensor_id in rig.sensor_ids():
                if rig.is_ref_sensor(sensor_id):
                    continue
                sensor_from_rig = rig.sensor_from_rig(sensor_id)
                if sensor_from_rig is not None:
                    fixed_sensor_poses.append(
                        (
                            rig,
                            sensor_id,
                            pycolmap.Rigid3d(sensor_from_rig.matrix()),
                        )
                    )
        adjuster = pycolmap.create_pose_prior_bundle_adjuster(
            options, prior_options, config, priors, reconstruction
        )
        # COLMAP's pose-prior constructor first performs a Sim(3) alignment,
        # which also scales Rig translations. Restore the calibrated metric
        # sensor transforms before solve; the BA config keeps them constant.
        for rig, sensor_id, sensor_from_rig in fixed_sensor_poses:
            rig.set_sensor_from_rig(sensor_id, sensor_from_rig)
        summary = adjuster.solve()
        # PyCOLMAP 4.1.1's pose-prior path can still mutate calibrated Rig
        # extrinsics despite the constant-sensor configuration. Reassert the
        # immutable calibration before any candidate snapshot/publication.
        for rig, sensor_id, sensor_from_rig in fixed_sensor_poses:
            rig.set_sensor_from_rig(sensor_id, sensor_from_rig)
        return summary
    return adjuster.solve()
