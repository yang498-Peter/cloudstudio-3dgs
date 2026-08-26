"""Per-image BA: salvage the poison segment instead of dropping it.

The user's directive stands: use every photo. The triple verdict only
proves no RIG-RIGID pose exists for capture 0.68-0.92 - the fixed stereo
constraint was part of every failed method. The likeliest physics is a
left/right (or camera/trajectory) time offset during the fast close
passes, and under that hypothesis each individual image still has a
perfectly consistent pose of its own.

The triangulated model already stores every image in its own trivial frame
(add_image_with_trivial_frame); the stereo rigidity is only imposed later
by apply_fixed_stereo_rig. So per-image BA is the same machinery with the
rig step skipped and segment-aware position priors:

    audited-good frames   0.02 m  (they are already right; stay put)
    poison segment        0.25 m  (free enough to travel the ~10-20 cm a
                                   time-sync error implies, anchored enough
                                   not to leave the LiDAR frame)

Verdict comes from the same LiDAR decile audit as every pose method before
it. Nothing is published; the output candidate feeds phase-28 only if every
decile clears 4 px.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Peter\cloudstudio-3dgs-gate1")
sys.path.insert(0, str(REPO))

from cloudstudio_3dgs.ba.report import stage_options  # noqa: E402

DATASETS = Path(r"C:\Peter\3dgs-datasets")
BA_ROOT = Path(r"C:\Peter\3dgs-runs\house0305_ba")
MODEL = BA_ROOT / "sfm_model" / "sfm"
OUTPUT = BA_ROOT / "per_image_ba"
POISON = (0.68, 0.92)
STDDEV_GOOD = 0.02
STDDEV_POISON = 0.25


def main() -> int:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(MODEL))
    print(f"model: {reconstruction.num_images()} images, "
          f"{reconstruction.num_points3D():,} points")
    before_error = reconstruction.compute_mean_reprojection_error()
    print(f"before: mean reprojection {before_error:.3f}px")

    manifest = json.loads(
        (DATASETS / "house0305_manifest" / "dataset_manifest.json")
        .read_text(encoding="utf-8"))
    times = sorted(int(i["timestamp_ns"]) for i in manifest["images"])
    t0, span = times[0], times[-1] - times[0]
    by_name = {}
    for record in manifest["images"]:
        name = record["path"].replace("camera/", "").replace("\\", "/")
        fraction = (int(record["timestamp_ns"]) - t0) / span
        c2w = np.asarray(record["c2w"], dtype=np.float64)
        by_name[name] = (c2w[:3, 3], fraction)

    options = pycolmap.BundleAdjustmentOptions()
    for key, value in stage_options("stage_1").items():
        setattr(options, key, value)
    options.refine_points3D = True
    options.print_summary = False

    config = pycolmap.BundleAdjustmentConfig()
    priors = []
    poison_count = 0
    image_ids = sorted(reconstruction.reg_image_ids())
    for image_id in image_ids:
        config.add_image(image_id)
        image = reconstruction.image(image_id)
        position, fraction = by_name[image.name.replace("\\", "/")]
        poison = POISON[0] <= fraction < POISON[1]
        poison_count += int(poison)
        stddev = STDDEV_POISON if poison else STDDEV_GOOD
        priors.append(pycolmap.PosePrior(
            corr_data_id=pycolmap.data_t(image.camera.sensor_id, int(image_id)),
            position=np.asarray(position, dtype=np.float64),
            position_covariance=np.eye(3) * stddev**2,
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
        ))
    first_frame_id = min(int(reconstruction.image(i).frame_id) for i in image_ids)
    config.set_constant_rig_from_world_pose(first_frame_id)
    print(f"priors: {len(priors)} images, {poison_count} in the poison band "
          f"at {STDDEV_POISON}m, rest at {STDDEV_GOOD}m")

    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    prior_options.prior_position_fallback_stddev = STDDEV_GOOD
    adjuster = pycolmap.create_pose_prior_bundle_adjuster(
        options, prior_options, config, priors, reconstruction)
    summary = adjuster.solve()
    steps = getattr(summary, "num_successful_iterations",
                    getattr(summary, "iterations", "?"))
    print(f"solver finished (iterations: {steps})")
    after_error = reconstruction.compute_mean_reprojection_error()
    print(f"after: mean reprojection {after_error:.3f}px "
          f"({(after_error - before_error) / before_error * 100:+.1f}%)")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate = OUTPUT / "candidate_model"
    candidate.mkdir(exist_ok=True)
    reconstruction.write(str(candidate))
    print(f"candidate -> {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
