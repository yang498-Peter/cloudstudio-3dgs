"""Project recovered MipMap densification candidates into Face4 training views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset


def _project(points: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    positive = camera[:, 2] > 1e-4
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = K[0, 0] * camera[:, 0] / np.maximum(camera[:, 2], 1e-4) + K[0, 2]
    pixels[:, 1] = K[1, 1] * camera[:, 1] / np.maximum(camera[:, 2], 1e-4) + K[1, 2]
    return pixels, positive


def _candidate_masks(payload: dict, threshold: float, opacity_floor: float) -> tuple[np.ndarray, np.ndarray]:
    state = payload["strategy_state"]
    opacity = torch.sigmoid(payload["params"]["opacities"].flatten()).cpu().numpy()
    legacy = (state["grad2d"] / state["count"].clamp_min(1.0)).cpu().numpy()
    recovered = (
        state["_cloudstudio_mipmap_grad_sum"]
        / state["_cloudstudio_mipmap_weight_sum"].clamp_min(1e-8)
    ).cpu().numpy()
    opacity_ok = opacity > opacity_floor
    return (legacy > threshold) & opacity_ok, (recovered > threshold) & opacity_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", type=float, default=1.5e-4)
    parser.add_argument("--opacity-floor", type=float, default=0.15)
    parser.add_argument("--top-views", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tile_manifest = json.loads(Path(config["tile_inputs_manifest"]).read_text(encoding="utf-8"))
    tile = next(
        item for item in tile_manifest["tiles"]
        if int(item["tile_id"]) == int(config["mipmap_tile_id"])
    )
    dataset = FaceCacheDataset(
        face_manifest_path=Path(config["face_cache_manifest"]),
        cache_root=Path(config["face_cache_root"]),
        verify_artifacts=False,
        dataset_manifest_path=Path(config["dataset_manifest"]),
        tile_views=tile["views"],
        renderer_mask_manifest_path=Path(config["renderer_mask_manifest"]),
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    means = payload["params"]["means"].detach().cpu().numpy().astype(np.float64)
    legacy_mask, recovered_mask = _candidate_masks(payload, args.threshold, args.opacity_floor)
    relevant = legacy_mask | recovered_mask
    candidate_points = means[relevant]
    legacy_relevant = legacy_mask[relevant]
    recovered_relevant = recovered_mask[relevant]

    scores: list[dict] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        pixels, positive = _project(candidate_points, sample.c2w, sample.K)
        x = np.rint(pixels[:, 0]).astype(np.int64)
        y = np.rint(pixels[:, 1]).astype(np.int64)
        inside = (
            positive & (x >= 0) & (x < sample.width)
            & (y >= 0) & (y < sample.height)
        )
        visible = np.zeros(len(candidate_points), dtype=bool)
        selected = np.flatnonzero(inside)
        if len(selected):
            visible[selected] = sample.rgb_mask[y[selected], x[selected]]
        gray = cv2.cvtColor(sample.image, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        visible_indexes = np.flatnonzero(visible)
        local_gradient = gradient[y[visible_indexes], x[visible_indexes]] if len(visible_indexes) else np.empty(0)
        scores.append({
            "index": index,
            "sample_id": sample.image_id,
            "visible_count": int(visible.sum()),
            "recovered_only_visible_count": int((visible & recovered_relevant & ~legacy_relevant).sum()),
            "overlap_visible_count": int((visible & recovered_relevant & legacy_relevant).sum()),
            "projected_gradient_p50": float(np.median(local_gradient)) if len(local_gradient) else 0.0,
            "projected_gradient_p90": float(np.quantile(local_gradient, 0.9)) if len(local_gradient) else 0.0,
        })

    ranked = sorted(
        scores,
        key=lambda row: (row["recovered_only_visible_count"] * row["projected_gradient_p90"], row["visible_count"]),
        reverse=True,
    )[: args.top_views]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for rank, row in enumerate(ranked, start=1):
        sample = dataset[row["index"]]
        pixels, positive = _project(candidate_points, sample.c2w, sample.K)
        x = np.rint(pixels[:, 0]).astype(np.int64)
        y = np.rint(pixels[:, 1]).astype(np.int64)
        visible = (
            positive & (x >= 0) & (x < sample.width)
            & (y >= 0) & (y < sample.height)
        )
        selected = np.flatnonzero(visible)
        selected = selected[sample.rgb_mask[y[selected], x[selected]]]
        if len(selected) > args.max_points:
            selected = rng.choice(selected, size=args.max_points, replace=False)
        canvas = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
        overlap = selected[legacy_relevant[selected] & recovered_relevant[selected]]
        recovered_only = selected[recovered_relevant[selected] & ~legacy_relevant[selected]]
        legacy_only = selected[legacy_relevant[selected] & ~recovered_relevant[selected]]
        for indexes, color in ((legacy_only, (255, 80, 40)), (overlap, (0, 220, 255)), (recovered_only, (30, 30, 255))):
            for item in indexes:
                cv2.circle(canvas, (int(x[item]), int(y[item])), 1, color, -1, lineType=cv2.LINE_AA)
        label = (
            f"red=recovered-only yellow=overlap blue=legacy-only  "
            f"visible={row['visible_count']} new={row['recovered_only_visible_count']}"
        )
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1120), 34), (0, 0, 0), -1)
        cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        safe_id = sample.image_id.replace(":", "_").replace("/", "_")
        path = output_dir / f"{rank:02d}_{safe_id}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        row["overlay_path"] = str(path.resolve())

    report = {
        "schema_version": 1,
        "kind": "mipmap_gradient_candidate_face4_projection_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": int(payload["step"]),
        "threshold": args.threshold,
        "opacity_floor": args.opacity_floor,
        "gaussian_count": int(len(means)),
        "legacy_candidate_count": int(legacy_mask.sum()),
        "recovered_candidate_count": int(recovered_mask.sum()),
        "ranked_views": ranked,
    }
    report_path = output_dir / "candidate_projection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "ranked_views": ranked}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
