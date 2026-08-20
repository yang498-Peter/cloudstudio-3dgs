from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from cloudstudio_3dgs.data.manifest import build_manifest
from cloudstudio_3dgs.geometry.kb4 import project_kb4
from cloudstudio_3dgs.geometry.rig import distribution, rotation_error_rad


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def histogram_quantile(histogram: np.ndarray, quantile: float) -> int:
    total = int(histogram.sum())
    if total == 0:
        return 0
    target = max(1, int(np.ceil(total * quantile)))
    return int(np.searchsorted(np.cumsum(histogram), target))


def scan_las(path: Path, max_points: int) -> tuple[dict[str, Any], np.ndarray]:
    import laspy

    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    histograms = np.zeros((3, 65536), dtype=np.int64)
    black_count = 0
    sampled: list[np.ndarray] = []
    with laspy.open(path) as reader:
        total = int(reader.header.point_count)
        stride = max(1, int(np.ceil(total / max_points)))
        has_rgb = {"red", "green", "blue"} <= set(reader.header.point_format.dimension_names)
        for chunk in reader.chunk_iterator(1_000_000):
            xyz = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float64)
            xyz_min = np.minimum(xyz_min, np.min(xyz, axis=0))
            xyz_max = np.maximum(xyz_max, np.max(xyz, axis=0))
            sampled.append(xyz[::stride])
            if has_rgb:
                rgb = np.column_stack([chunk.red, chunk.green, chunk.blue]).astype(np.uint16)
                for channel in range(3):
                    histograms[channel] += np.bincount(rgb[:, channel], minlength=65536)
                black_count += int(np.all(rgb == 0, axis=1).sum())
    points = np.concatenate(sampled, axis=0)[:max_points]
    rgb_stats: dict[str, Any]
    if has_rgb:
        rgb_stats = {
            "available": True,
            "min": [histogram_quantile(hist, 0.0) for hist in histograms],
            "median": [histogram_quantile(hist, 0.5) for hist in histograms],
            "p99": [histogram_quantile(hist, 0.99) for hist in histograms],
            "max": [int(np.max(np.flatnonzero(hist))) if hist.any() else 0 for hist in histograms],
            "black_fraction": black_count / total if total else 0.0,
        }
    else:
        rgb_stats = {"available": False, "black_fraction": 0.0}
    return (
        {
            "point_count": total,
            "bounds_min": xyz_min.tolist(),
            "bounds_max": xyz_max.tolist(),
            "extent": (xyz_max - xyz_min).tolist(),
            "rgb": rgb_stats,
            "qa_sample_points": len(points),
            "qa_sample_stride": stride,
        },
        points,
    )


def camera_parameters(camera: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    return camera["intrinsic"], camera["distortion"]["params"]


def project_world_points(
    points_world: np.ndarray,
    image: dict[str, Any],
    camera: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_from_camera = np.asarray(image["c2w"], dtype=np.float64)
    points_camera = (points_world - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    intrinsic, distortion = camera_parameters(camera)
    uv, ranges, valid = project_kb4(
        points_camera,
        intrinsic,
        distortion,
        min_range_m=float(config["min_range_m"]),
        max_range_m=float(config["max_range_m"]),
        max_theta_rad=np.radians(float(config["max_theta_deg"])),
    )
    valid &= (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < int(camera["width"]))
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < int(camera["height"]))
    )
    return uv, ranges, valid


def edge_alignment(
    image_path: Path,
    uv: np.ndarray,
    ranges: np.ndarray,
    valid: np.ndarray,
    camera: dict[str, Any],
    config: dict[str, Any],
    overlay_path: Path,
) -> dict[str, Any]:
    source = Image.open(image_path).convert("RGB")
    maximum = int(config["edge_image_max_dimension"])
    scale = min(1.0, maximum / max(source.size))
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.BILINEAR,
    )
    projected = uv[valid] * scale
    projected_ranges = ranges[valid]
    if len(projected) < 8:
        raise ValueError(f"too few projected LiDAR points for edge QA: {len(projected)}")
    tree = cKDTree(projected)
    distances, indices = tree.query(projected, k=min(5, len(projected)))
    neighbor_valid = distances <= float(config["depth_edge_neighbor_px"])
    range_delta = np.abs(
        np.log(np.maximum(projected_ranges[:, None], 1e-6))
        - np.log(np.maximum(projected_ranges[indices], 1e-6))
    )
    lidar_edge = np.any(
        neighbor_valid & (range_delta >= float(config["depth_edge_log_range_delta"])), axis=1
    )
    edge_points = projected[lidar_edge]
    if not len(edge_points):
        raise ValueError("no LiDAR depth edges found for edge QA")

    gray = np.asarray(resized.convert("L"), dtype=np.float32)
    gradient = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
    positive = gradient[gradient > 0]
    if not len(positive):
        raise ValueError("image has no measurable gradients")
    threshold = float(np.percentile(positive, float(config["edge_gradient_percentile"])))
    image_edges = gradient >= threshold
    distance_map = ndimage.distance_transform_edt(~image_edges)
    x = np.clip(np.rint(edge_points[:, 0]).astype(int), 0, resized.width - 1)
    y = np.clip(np.rint(edge_points[:, 1]).astype(int), 0, resized.height - 1)
    distances_original_px = distance_map[y, x] / scale

    overlay = np.asarray(resized).copy()
    overlay[image_edges, 1] = 255
    rendered = Image.fromarray(overlay)
    draw = ImageDraw.Draw(rendered)
    for px, py in edge_points[:: max(1, len(edge_points) // 2000)]:
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(255, 32, 32))
    rendered.save(overlay_path, quality=90)
    return {
        "lidar_edge_points": int(len(edge_points)),
        "distance_px": distribution(distances_original_px.tolist()),
        "overlay": overlay_path.name,
    }


def interval_metrics(timestamps_ns: list[int]) -> dict[str, float]:
    values = np.diff(np.asarray(sorted(timestamps_ns), dtype=np.float64)) / 1_000_000.0
    return distribution(values.tolist())


def trajectory_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    images = {image["image_id"]: image for image in manifest["images"]}
    left_from_lidar = np.asarray(manifest["rig"]["left_from_lidar"], dtype=np.float64)
    poses: list[tuple[int, np.ndarray]] = []
    for frame in manifest["rig_frames"]:
        left = images[frame["left_image_id"]]
        poses.append((int(frame["timestamp_ns"]), np.asarray(left["c2w"]) @ left_from_lidar))
    speeds: list[float] = []
    angular_speeds: list[float] = []
    for (time_a, pose_a), (time_b, pose_b) in zip(poses, poses[1:]):
        seconds = (time_b - time_a) / 1_000_000_000.0
        if seconds <= 0:
            continue
        speeds.append(float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]) / seconds))
        angular_speeds.append(
            float(np.degrees(rotation_error_rad(pose_b, pose_a)) / seconds)
        )
    return {
        "speed_mps": distribution(speeds),
        "angular_speed_deg_s": distribution(angular_speeds),
    }


def add_gate(gates: list[dict[str, Any]], name: str, value: float, operator: str, threshold: float) -> None:
    passed = value >= threshold if operator == ">=" else value <= threshold
    gates.append(
        {"name": name, "value": value, "operator": operator, "threshold": threshold, "passed": passed}
    )


def evaluate_gates(metrics: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    projection = config["projection"]
    timing = config["timing"]
    trajectory = config["trajectory"]
    rig = config["rig"]
    point_cloud = config["point_cloud"]
    add_gate(gates, "projection.visible_fraction_p50", metrics["projection"]["visible_fraction"]["p50"], ">=", projection["minimum_visible_fraction_p50"])
    add_gate(gates, "projection.edge_distance_p50_px", metrics["projection"]["edge_distance_px"]["p50"], "<=", projection["maximum_edge_distance_p50_px"])
    add_gate(gates, "projection.edge_frame_success_fraction", metrics["projection"]["edge_frame_success_fraction"], ">=", projection["minimum_edge_frame_success_fraction"])
    add_gate(gates, "timing.pair_delta_max_ms", metrics["timing"]["pair_delta_ms"]["max"], "<=", timing["maximum_pair_delta_ms"])
    add_gate(gates, "timing.frame_interval_max_ms", metrics["timing"]["frame_interval_ms"]["max"], "<=", timing["maximum_frame_interval_ms"])
    add_gate(gates, "trajectory.speed_max_mps", metrics["trajectory"]["speed_mps"]["max"], "<=", trajectory["maximum_speed_mps"])
    add_gate(gates, "trajectory.angular_speed_max_deg_s", metrics["trajectory"]["angular_speed_deg_s"]["max"], "<=", trajectory["maximum_angular_speed_deg_s"])
    add_gate(gates, "rig.translation_error_p95_m", metrics["rig"]["relative_translation_error_m"]["p95"], "<=", rig["maximum_translation_error_p95_m"])
    add_gate(gates, "rig.rotation_error_p95_rad", metrics["rig"]["relative_rotation_error_rad"]["p95"], "<=", rig["maximum_rotation_error_p95_rad"])
    add_gate(gates, "rig.intrinsic_max_difference", metrics["rig"]["calibration_vs_transforms"]["max_abs_difference"], "<=", rig["maximum_intrinsic_difference"])
    add_gate(gates, "point_cloud.black_rgb_fraction", metrics["point_cloud"]["rgb"]["black_fraction"], "<=", point_cloud["maximum_black_rgb_fraction"])
    return gates


def build_qa_report(
    recording_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    overlay_dir: Path,
    *,
    allow_qa_warning: bool = False,
) -> dict[str, Any]:
    manifest = build_manifest(recording_dir, run_dir, hash_images=False, hash_point_cloud=False)
    point_cloud_path = run_dir / manifest["point_cloud"]["path"]
    point_cloud_metrics, points = scan_las(point_cloud_path, int(config["projection"]["max_points"]))
    cameras = {camera["camera_id"]: camera for camera in manifest["cameras"]}
    per_frame: list[dict[str, Any]] = []
    fractions: list[float] = []
    projections: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for image in manifest["images"]:
        camera = cameras[image["camera_id"]]
        uv, ranges, valid = project_world_points(points, image, camera, config["projection"])
        count = int(valid.sum())
        fraction = count / len(points) if len(points) else 0.0
        fractions.append(fraction)
        per_frame.append(
            {"image_id": image["image_id"], "path": image["path"], "visible_points": count, "visible_fraction": fraction}
        )
        projections[image["image_id"]] = (uv, ranges, valid)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    for side in ("left", "right"):
        side_images = [image for image in manifest["images"] if image["side"] == side]
        count = min(int(config["projection"]["edge_frames_per_camera"]), len(side_images))
        selected.extend(side_images[index] for index in np.linspace(0, len(side_images) - 1, count, dtype=int))
    edge_reports: list[dict[str, Any]] = []
    edge_distances: list[float] = []
    edge_successes = 0
    for image in selected:
        uv, ranges, valid = projections[image["image_id"]]
        relative = image["path"].removeprefix("camera/")
        overlay_path = overlay_dir / f"{image['side']}_{image['timestamp_ns']}_edges.jpg"
        try:
            result = edge_alignment(
                recording_dir / "camera" / Path(relative),
                uv,
                ranges,
                valid,
                cameras[image["camera_id"]],
                config["projection"],
                overlay_path,
            )
            edge_successes += 1
            edge_distances.append(result["distance_px"]["p50"])
        except ValueError as exc:
            result = {"error": str(exc), "overlay": None}
        result.update({"image_id": image["image_id"], "path": image["path"]})
        edge_reports.append(result)

    side_intervals = {
        side: interval_metrics([image["timestamp_ns"] for image in manifest["images"] if image["side"] == side])
        for side in ("left", "right")
    }
    frame_interval_max = max(value["max"] for value in side_intervals.values())
    pair_delta_ms = {
        key: value / 1_000_000.0
        for key, value in manifest["rig_diagnostics"]["timestamp_delta_ns"].items()
    }
    metrics = {
        "projection": {
            "visible_fraction": distribution(fractions),
            "edge_distance_px": distribution(edge_distances),
            "edge_frame_success_fraction": edge_successes / len(selected) if selected else 0.0,
            "edge_frames": edge_reports,
            "per_frame_visible_lidar_points": per_frame,
        },
        "timing": {
            "pair_delta_ms": pair_delta_ms,
            "frame_interval_by_camera_ms": side_intervals,
            "frame_interval_ms": {"p50": max(v["p50"] for v in side_intervals.values()), "p95": max(v["p95"] for v in side_intervals.values()), "max": frame_interval_max},
        },
        "trajectory": trajectory_metrics(manifest),
        "rig": manifest["rig_diagnostics"],
        "point_cloud": point_cloud_metrics,
    }
    gates = evaluate_gates(metrics, config)
    passed = all(gate["passed"] for gate in gates)
    return {
        "schema_version": 1,
        "recording_id": manifest["recording_id"],
        "coordinate_frame": manifest["coordinate_frame"],
        "manifest_sha256": manifest["manifest_sha256"],
        "config": config,
        "metrics": metrics,
        "gates": gates,
        "passed": passed,
        "override_used": bool(allow_qa_warning and not passed),
        "status": "PASS" if passed else ("WARNING_OVERRIDDEN" if allow_qa_warning else "FAIL"),
    }


def write_report(report: dict[str, Any], output_dir: Path, source_overlays: Path, *, force: bool) -> Path:
    qa_dir = output_dir / "qa"
    if qa_dir.exists() and any(qa_dir.iterdir()) and not force:
        raise FileExistsError(f"QA output is not empty: {qa_dir}; pass --force")
    qa_dir.mkdir(parents=True, exist_ok=True)
    overlays = qa_dir / "overlays"
    if overlays.exists():
        shutil.rmtree(overlays)
    shutil.move(str(source_overlays), overlays)
    report_path = qa_dir / "report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".report.", suffix=".tmp", dir=qa_dir)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, report_path)
    rows = "".join(
        f"<tr><td>{html.escape(gate['name'])}</td><td>{gate['value']:.6g}</td>"
        f"<td>{html.escape(gate['operator'])} {gate['threshold']:.6g}</td>"
        f"<td class={'pass' if gate['passed'] else 'fail'}>{'PASS' if gate['passed'] else 'FAIL'}</td></tr>"
        for gate in report["gates"]
    )
    html_payload = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>S1 3DGS QA</title>
<style>body{{font:14px Segoe UI,Arial;margin:24px;color:#1f2937}}table{{border-collapse:collapse}}td,th{{border:1px solid #d1d5db;padding:7px 10px}}.pass{{color:#087443}}.fail{{color:#b42318;font-weight:700}}pre{{white-space:pre-wrap;background:#f3f4f6;padding:12px}}</style>
<h1>S1 3DGS data QA: {html.escape(report['status'])}</h1><p>Recording: {html.escape(report['recording_id'])}</p>
<table><tr><th>Gate</th><th>Value</th><th>Threshold</th><th>Status</th></tr>{rows}</table>
<h2>Machine-readable report</h2><pre>{html.escape(payload)}</pre></html>"""
    (qa_dir / "report.html").write_text(html_payload, encoding="utf-8", newline="\n")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantitative QA on an MVP S1 recording and solve")
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/qa_default.json"))
    parser.add_argument("--allow-qa-warning", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    overlay_temp = Path(tempfile.mkdtemp(prefix="s1gs-qa-overlays-"))
    try:
        report = build_qa_report(
            args.recording.resolve(),
            args.run.resolve(),
            load_config(args.config),
            overlay_temp,
            allow_qa_warning=args.allow_qa_warning,
        )
        destination = write_report(report, args.output.resolve(), overlay_temp, force=args.force)
    finally:
        if overlay_temp.exists():
            shutil.rmtree(overlay_temp)
    print(f"QA {report['status']} -> {destination}")
    for gate in report["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['name']}: {gate['value']:.6g} {gate['operator']} {gate['threshold']:.6g}")
    return 0 if report["passed"] or args.allow_qa_warning else 2


if __name__ == "__main__":
    raise SystemExit(main())
