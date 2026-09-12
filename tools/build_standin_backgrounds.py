#!/usr/bin/env python3
"""Per-view backdrops with a frozen stand-in for everything outside one Tile.

A Tile is trained with ``final = render + (1 - alpha) * backdrop`` and today
the backdrop is the sky dome alone (tools/build_view_backgrounds.py cropped
by tools/build_tile_view_backgrounds.py). Every photo pixel whose true
surface is not in the Tile - the neighbouring Tiles' walls, the trees behind
the eave, the far interior seen through a doorway - therefore has no owner
and the Tile grows gaussians at arbitrary depth to paint it
(research/quality_recovery_v2/README.zh-CN.md, 2026-09-11 17:15 and the
door-plane check; survey 11 section 5: every block system gives a block a
stand-in for out-of-block content).

This tool builds the stand-in backdrop, B-line:

    backdrop(view) = render( dome  +  standin \\ box(Tile) )

where ``standin`` is one or more frozen checkpoints (the other Tiles' trained
checkpoints for the neighbouring house parts, a coarse whole-scene model for
trees and far ground nobody tiles) with every gaussian inside the selected
Tile's box removed, so the Tile still owns its own volume and nothing is
counted twice. Rendering happens at the exact Tile crop camera
(``FaceCacheDataset.camera_sample`` with ``tile_views``), at crop resolution,
so the library only ever serves the stored image as-is - the squash the crop
tool exists to prevent never enters. The manifest keeps the trainer schema
(``views[sample_id] = {file, height, width}``, signed) and adds a ``standin``
provenance block; the trainer's fail-closed rules are unchanged.

CPU parts (checkpoint loading, row selection, layer concatenation, manifest
writing, the plan report) are importable and unit-tested with synthetic
gaussians. The render loop needs the CUDA rasterizer: ``--plan-only`` does
everything but render.

    python tools/build_standin_backgrounds.py \\
        --config C:/Peter/3dgs-runs/house0305_sop/tile1_R1d_20k.json \\
        --dome C:/Peter/3dgs-runs/probes/sky_house0305.pt \\
        --standin-checkpoint C:/Peter/3dgs-runs/house0305_sop/tile0_R1_range0_20k/checkpoints/latest.pt \\
        --standin-checkpoint C:/Peter/3dgs-runs/house0305_sop/tile2_R1d_20k/checkpoints/latest.pt \\
        --standin-checkpoint C:/Peter/3dgs-runs/house0305_sop/tile3_R1d_cap13m_20k/checkpoints/latest.pt \\
        --standin-checkpoint C:/Peter/3dgs-runs/house0305_sop/global_coarse_B0_10k/checkpoints/latest.pt \\
        --output C:/Peter/3dgs-runs/house0305_sop/tile_backgrounds_B1/Tile_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.training.tile_inputs import verify_tile_inputs_manifest  # noqa: E402
from cloudstudio_3dgs.training.view_backgrounds import (  # noqa: E402
    write_view_background_manifest,
)

PARAMETER_NAMES = ("means", "quats", "scales", "opacities", "sh0", "shN")
# rgb = C0 * sh0 + 0.5 (the gsplat SH convention); shared with the merge tool.
SH_C0 = 0.28209479177387814
BOX_KINDS = ("training_and_export_box", "core_box")
STANDIN_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# checkpoint layers (CPU)
# ----------------------------------------------------------------------------


def checkpoint_layer(payload: dict[str, Any], *, torch: Any, name: str = "") -> dict[str, Any]:
    """Detached float32 CPU copies of the six gaussian parameters.

    Accepts the trainer's ``params`` mapping or the upstream ``splats``
    name; ``shN`` may be absent (a DC-only layer such as the sky dome) and is
    then materialised as an empty band so concatenation is uniform.
    """
    params = payload.get("params")
    if not isinstance(params, dict):
        params = payload.get("splats")
    if not isinstance(params, dict) or "means" not in params:
        raise ValueError(f"checkpoint {name or '<payload>'} has no params mapping")
    layer: dict[str, Any] = {}
    count = int(params["means"].shape[0])
    for key in PARAMETER_NAMES:
        if key == "shN" and key not in params:
            layer[key] = torch.zeros((count, 0, 3), dtype=torch.float32)
            continue
        if key not in params:
            raise ValueError(f"checkpoint {name or '<payload>'} is missing {key}")
        value = params[key].detach().to("cpu", torch.float32)
        if int(value.shape[0]) != count:
            raise ValueError(f"parameter {key} has another row count in {name or '<payload>'}")
        layer[key] = value
    if layer["opacities"].ndim != 1:
        layer["opacities"] = layer["opacities"].reshape(count)
    return layer


def median_exposure_gain(payload: dict[str, Any], *, torch: Any) -> float | None:
    """The checkpoint's own learned exposure gain (median over frames), if any."""
    log_gains = (payload.get("auxiliary_params") or {}).get("exposure_log_gains")
    if log_gains is None:
        return None
    return float(torch.exp(log_gains.detach().to("cpu", torch.float32)).median())


def bake_exposure_gain(layer: dict[str, Any], gain: float) -> dict[str, Any]:
    """Fold a scalar gain into the colours so the layer sits in the photo frame.

    Same DC formula as tools/merge_v28_tile_checkpoints.py
    (``rgb * gain`` with ``rgb = C0 * sh0 + 0.5``); the higher bands are
    scaled too, which the merge tool leaves alone - view-dependent residuals
    are small but the composite here is an appearance target, so keep them
    exact.
    """
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("exposure gain must be a positive finite scalar")
    baked = dict(layer)
    baked["sh0"] = layer["sh0"] * gain + (gain - 1.0) * 0.5 / SH_C0
    baked["shN"] = layer["shN"] * gain
    return baked


def inside_box_mask(means: np.ndarray, box: Any, *, margin_m: float = 0.0) -> np.ndarray:
    """Rows whose centre lies within ``box`` grown by ``margin_m`` on every face."""
    lower = np.asarray(box[0], dtype=np.float64) - float(margin_m)
    upper = np.asarray(box[1], dtype=np.float64) + float(margin_m)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("box must be a finite [[min xyz], [max xyz]] with positive extent")
    xyz = np.asarray(means, dtype=np.float64)
    return np.all((xyz >= lower) & (xyz <= upper), axis=1)


def anchor_distance(means: np.ndarray, anchors: np.ndarray, *, max_distance_m: float) -> np.ndarray:
    """Distance to the nearest anchor with the query bounded at ``max_distance_m``.

    Rows beyond the bound read ``inf``; the same bounded cKDTree query the
    surface-anchor prune uses (cloudstudio_3dgs/training/surface_anchor.py).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(anchors, dtype=np.float64))
    distance, _ = tree.query(
        np.asarray(means, dtype=np.float64), k=1,
        distance_upper_bound=float(max_distance_m), workers=-1,
    )
    return distance


def select_standin_rows(
    layer: dict[str, Any],
    *,
    torch: Any,
    exclude_box: Any | None,
    exclude_margin_m: float = 0.0,
    min_opacity: float = 0.0,
    anchors: np.ndarray | None = None,
    max_anchor_distance_m: float | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Drop rows the Tile owns (inside its box), dead rows, and unanchored rows.

    Returns the filtered layer and the attribution of every removed row in
    the order the rules are applied (box, then opacity, then anchor), so
    the counts add up and a rule that removes nothing is visible as zero.
    """
    means = layer["means"].numpy()
    count = int(means.shape[0])
    keep = np.ones(count, dtype=bool)
    stats = {"input_count": count, "removed_inside_box": 0, "removed_opacity": 0, "removed_anchor": 0}
    if exclude_box is not None:
        inside = inside_box_mask(means, exclude_box, margin_m=exclude_margin_m)
        stats["removed_inside_box"] = int(np.count_nonzero(inside & keep))
        keep &= ~inside
    if min_opacity > 0.0:
        opacity = torch.sigmoid(layer["opacities"]).numpy()
        dead = opacity < float(min_opacity)
        stats["removed_opacity"] = int(np.count_nonzero(dead & keep))
        keep &= ~dead
    if max_anchor_distance_m is not None:
        if anchors is None:
            raise ValueError("max_anchor_distance_m requires anchor points")
        far = ~np.isfinite(anchor_distance(means[keep], anchors, max_distance_m=max_anchor_distance_m))
        kept_indices = np.flatnonzero(keep)
        stats["removed_anchor"] = int(np.count_nonzero(far))
        keep[kept_indices[far]] = False
    stats["kept_count"] = int(np.count_nonzero(keep))
    index = torch.from_numpy(np.flatnonzero(keep))
    filtered = {key: value.index_select(0, index) for key, value in layer.items()}
    return filtered, stats


def concat_layers(layers: list[dict[str, Any]], *, torch: Any) -> dict[str, Any]:
    """Concatenate layers, zero-padding ``shN`` to the widest band present."""
    if not layers:
        raise ValueError("at least one layer is required")
    width = max(int(layer["shN"].shape[1]) for layer in layers)
    pieces: dict[str, list[Any]] = {key: [] for key in PARAMETER_NAMES}
    for layer in layers:
        for key in PARAMETER_NAMES:
            value = layer[key]
            if key == "shN" and int(value.shape[1]) < width:
                pad = torch.zeros(
                    (int(value.shape[0]), width - int(value.shape[1]), 3), dtype=value.dtype
                )
                value = torch.cat([value, pad], dim=1)
            pieces[key].append(value)
    return {key: torch.cat(values, dim=0) for key, values in pieces.items()}


# ----------------------------------------------------------------------------
# plan (CPU): which rows stand in, from which sources
# ----------------------------------------------------------------------------


def selected_tile(tile_inputs: dict[str, Any], tile_id: int) -> dict[str, Any]:
    matches = [tile for tile in tile_inputs["tiles"] if int(tile["tile_id"]) == int(tile_id)]
    if len(matches) != 1:
        raise ValueError("Tile inputs do not contain a unique selected Tile")
    return matches[0]


def build_standin(
    *,
    torch: Any,
    dome_path: Path,
    standin_paths: list[Path],
    exclude_box: Any | None,
    exclude_box_kind: str,
    exclude_margin_m: float,
    min_opacity: float,
    anchors: np.ndarray | None,
    anchor_source: str | None,
    max_anchor_distance_m: float | None,
    harmonize_exposure: bool,
    target_gain: float,
    load=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load dome + stand-ins, filter, harmonize, concatenate; return (params, provenance).

    ``load`` defaults to ``torch.load(path, map_location="cpu", weights_only=False)``
    and is injectable so the selection logic is testable without files.
    """
    if load is None:
        def load(path: Path) -> dict[str, Any]:
            return torch.load(path, map_location="cpu", weights_only=False)

    dome_payload = load(dome_path)
    dome = checkpoint_layer(dome_payload, torch=torch, name=str(dome_path))
    layers = [dome]
    sources: list[dict[str, Any]] = []
    for path in standin_paths:
        started = time.time()
        payload = load(path)
        layer = checkpoint_layer(payload, torch=torch, name=str(path))
        gain = median_exposure_gain(payload, torch=torch) if harmonize_exposure else None
        applied_gain = None
        if harmonize_exposure:
            if gain is None:
                raise ValueError(
                    f"{path} carries no exposure gains to harmonize; pass "
                    "--no-harmonize-exposure to use its colours as stored"
                )
            applied_gain = float(gain) / float(target_gain)
            layer = bake_exposure_gain(layer, applied_gain)
        filtered, stats = select_standin_rows(
            layer,
            torch=torch,
            exclude_box=exclude_box,
            exclude_margin_m=exclude_margin_m,
            min_opacity=min_opacity,
            anchors=anchors,
            max_anchor_distance_m=max_anchor_distance_m,
        )
        layers.append(filtered)
        sources.append(
            {
                "path": str(Path(path)),
                "sha256": sha256_file(path) if Path(path).is_file() else None,
                "step": int(payload.get("step", -1)),
                "exposure_gain_median": gain,
                "exposure_gain_applied": applied_gain,
                "sh_bands": int(layer["shN"].shape[1]),
                **stats,
                "seconds": round(time.time() - started, 1),
            }
        )
        del payload
    params = concat_layers(layers, torch=torch)
    provenance = {
        "schema_version": STANDIN_SCHEMA_VERSION,
        "composition": "single_pass_render_of_dome_plus_standin",
        "dome_count": int(dome["means"].shape[0]),
        "sources": sources,
        "standin_count": int(sum(source["kept_count"] for source in sources)),
        "rendered_gaussian_count": int(params["means"].shape[0]),
        "exclusion": {
            "box_kind": exclude_box_kind,
            "box": None if exclude_box is None else np.asarray(exclude_box, dtype=np.float64).tolist(),
            "margin_m": float(exclude_margin_m),
        },
        "min_opacity": float(min_opacity),
        "anchor": {
            "source": anchor_source,
            "max_distance_m": max_anchor_distance_m,
            "anchor_count": None if anchors is None else int(len(anchors)),
        },
        "exposure": {
            "harmonized": bool(harmonize_exposure),
            "target_gain": float(target_gain),
            "frame": "photo" if float(target_gain) == 1.0 else "tile_ungained",
        },
    }
    return params, provenance


# ----------------------------------------------------------------------------
# render (GPU) and manifest
# ----------------------------------------------------------------------------


def render_backdrops(
    *,
    backend: Any,
    torch: Any,
    params: dict[str, Any],
    samples: list[Any],
    output_root: Path,
    background_rgb: tuple[float, float, float],
    downsample: int = 1,
    save_threads: int = 8,
    verify_against: Path | None = None,
    progress=print,
) -> dict[str, dict[str, int | str]]:
    """Render every camera sample once and store it at its own (crop) size.

    ``backend.render(params, sample, with_range=False, background_rgb=...)``
    must return ``(rgb[H, W, 3] in [0, 1], ...)`` - the trainer's backend.
    Stored height/width are the sample's, divided by ``downsample`` only when
    asked (the stand-in carries high-frequency content; the dome-only library
    was downsampled 4x because it is low-frequency by construction).
    """
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    def encode_and_save(name: str, image: np.ndarray) -> None:
        picture = Image.fromarray(image)
        if downsample > 1:
            picture = picture.resize(
                (picture.width // downsample, picture.height // downsample), Image.BILINEAR
            )
        picture.save(output_root / name)
        if verify_against is not None:
            with Image.open(Path(verify_against) / name) as previous:
                reference = np.asarray(previous.convert("RGB"), dtype=np.int16)
            fresh = np.asarray(picture.convert("RGB"), dtype=np.int16)
            if reference.shape != fresh.shape:
                raise ValueError(f"verification shape mismatch for {name}")
            worst = int(np.abs(reference - fresh).max())
            if worst > 1:
                raise ValueError(f"verification mismatch for {name}: max |delta| = {worst}")

    views: dict[str, dict[str, int | str]] = {}
    with ThreadPoolExecutor(max_workers=save_threads) as pool:
        pending = []
        for index, sample in enumerate(samples):
            with torch.no_grad():
                rendered = backend.render(
                    params, sample, with_range=False, background_rgb=tuple(background_rgb)
                )[0]
            image = (rendered.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            if image.shape[0] != int(sample.height) or image.shape[1] != int(sample.width):
                raise ValueError(
                    f"render size {image.shape[:2]} differs from the camera sample "
                    f"{(sample.height, sample.width)} for {sample.image_id}"
                )
            name = sample.image_id.replace("::", "__") + ".png"
            if sample.image_id in views:
                raise ValueError(f"duplicate sample id {sample.image_id}")
            pending.append(pool.submit(encode_and_save, name, image))
            views[sample.image_id] = {
                "file": name,
                "height": int(sample.height) // downsample,
                "width": int(sample.width) // downsample,
            }
            if index % 200 == 0:
                progress(f"  {index + 1}/{len(samples)}")
        for task in pending:
            task.result()
    return views


def write_standin_manifest(
    path: Path,
    *,
    views: dict[str, dict[str, int | str]],
    tile_id: int,
    tile_inputs_manifest_sha256: str,
    dome_path: Path,
    dome_sha256: str,
    background_rgb: tuple[float, float, float],
    downsample: int,
    provenance: dict[str, Any],
    split: str = "train",
) -> str:
    """Signed manifest in the trainer's schema plus the ``standin`` block."""
    return write_view_background_manifest(
        path,
        views=views,
        metadata={
            "split": split,
            "dome_source": str(dome_path),
            "dome_sha256": dome_sha256,
            "background_rgb": [float(value) for value in background_rgb],
            "downsample": int(downsample),
            "tile_id": int(tile_id),
            "source_tile_inputs_manifest_sha256": tile_inputs_manifest_sha256,
            "render_resolution": "tile_crop",
            "standin": provenance,
        },
    )


def load_anchor_points(path: Path) -> np.ndarray:
    """xyz of a binary PLY point cloud (the Tile initialization convention)."""
    from tools.gaussian_health import read_ply_records

    records = read_ply_records(Path(path))
    return np.stack(
        [np.asarray(records[axis], dtype=np.float64) for axis in ("x", "y", "z")], axis=1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="Tile trainer config: dataset, renderer lock, Tile inputs and mipmap_tile_id")
    parser.add_argument("--dome", type=Path, required=True, help="sky dome checkpoint (.pt)")
    parser.add_argument("--standin-checkpoint", type=Path, action="append", default=[],
                        help="frozen checkpoint(s) standing in for out-of-Tile content; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-box-kind", choices=BOX_KINDS + ("none",), default="training_and_export_box",
                        help="which box of the selected Tile removes stand-in rows (default: training_and_export_box)")
    parser.add_argument("--exclude-margin-m", type=float, default=0.0)
    parser.add_argument("--min-opacity", type=float, default=0.05,
                        help="drop stand-in rows below this sigmoid opacity (delivery export uses 0.05)")
    parser.add_argument("--anchor-ply", type=Path,
                        help="LiDAR point cloud PLY; with --max-anchor-distance-m, stand-in rows farther than this from any point are dropped")
    parser.add_argument("--max-anchor-distance-m", type=float)
    parser.add_argument("--no-harmonize-exposure", action="store_true",
                        help="use stand-in colours as stored instead of baking each source's median exposure gain")
    parser.add_argument("--target-gain", type=float, default=1.0,
                        help="1.0 = photo frame (matches the dome); the selected Tile's own median gain would put the stand-in in its un-gained frame")
    parser.add_argument("--background", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--save-threads", type=int, default=8)
    parser.add_argument("--limit", type=int, help="render only the first N views (smoke; no manifest)")
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--plan-only", action="store_true",
                        help="CPU only: load, filter and count; write standin_plan.json; do not render")
    parser.add_argument("--plan-json", type=Path, help="where --plan-only writes (default: <output>/standin_plan.json)")
    args = parser.parse_args()

    import torch

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if raw.get("tile_inputs_manifest") is None or raw.get("mipmap_tile_id") is None:
        raise ValueError("the config must select a Tile (tile_inputs_manifest + mipmap_tile_id)")
    tile_inputs_path = Path(raw["tile_inputs_manifest"])
    tile_inputs = json.loads(tile_inputs_path.read_text(encoding="utf-8"))
    tile_inputs_sha = verify_tile_inputs_manifest(tile_inputs)
    tile = selected_tile(tile_inputs, int(raw["mipmap_tile_id"]))
    exclude_box = None if args.exclude_box_kind == "none" else tile[args.exclude_box_kind]

    anchors = None
    if args.max_anchor_distance_m is not None:
        if args.anchor_ply is None:
            raise ValueError("--max-anchor-distance-m requires --anchor-ply")
        anchors = load_anchor_points(args.anchor_ply)

    params, provenance = build_standin(
        torch=torch,
        dome_path=args.dome,
        standin_paths=list(args.standin_checkpoint),
        exclude_box=exclude_box,
        exclude_box_kind=args.exclude_box_kind,
        exclude_margin_m=args.exclude_margin_m,
        min_opacity=args.min_opacity,
        anchors=anchors,
        anchor_source=None if args.anchor_ply is None else str(args.anchor_ply),
        max_anchor_distance_m=args.max_anchor_distance_m,
        harmonize_exposure=not args.no_harmonize_exposure,
        target_gain=args.target_gain,
    )
    provenance["tile_id"] = int(tile["tile_id"])
    provenance["view_count"] = int(tile["view_count"])
    for source in provenance["sources"]:
        print(
            f"  {Path(source['path']).parent.parent.name}: {source['input_count']:,} -> "
            f"{source['kept_count']:,} (in box {source['removed_inside_box']:,}, "
            f"opacity {source['removed_opacity']:,}, anchor {source['removed_anchor']:,}; "
            f"gain {source['exposure_gain_applied']})"
        )
    print(
        f"stand-in {provenance['standin_count']:,} + dome {provenance['dome_count']:,} = "
        f"{provenance['rendered_gaussian_count']:,} gaussians for Tile_{tile['tile_id']} "
        f"({tile['view_count']} views)"
    )

    if args.plan_only:
        plan_path = args.plan_json or (args.output / "standin_plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = {
            "kind": "standin_backdrop_plan_v1",
            "config": str(args.config),
            "dome": {"path": str(args.dome), "sha256": sha256_file(args.dome)},
            "tile_inputs_manifest_sha256": tile_inputs_sha,
            "output": str(args.output),
            "background_rgb": list(args.background),
            "downsample": int(args.downsample),
            "estimated_param_bytes": int(
                sum(int(value.numel()) * 4 for value in params.values())
            ),
            **provenance,
        }
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(f"plan-only: wrote {plan_path}; nothing rendered")
        return 0

    # ------------------------------------------------------------------
    # GPU entry point: everything below needs the CUDA rasterizer.
    # ------------------------------------------------------------------
    from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset
    from tools.sharpness_metrics import _load_backend

    backend, torch_mod = _load_backend(raw, sh_degree=int(raw.get("sh_degree", 1)))
    device = raw.get("device", "cuda:0")
    backend.sh_degree = max(int(backend.sh_degree), 1 if params["shN"].shape[1] >= 3 else 0)
    gpu_params = {key: value.to(device) for key, value in params.items()}
    del params
    dataset = FaceCacheDataset(
        Path(raw["face_cache_manifest"]),
        Path(raw["face_cache_root"]),
        verify_artifacts=False,
        dataset_manifest_path=Path(raw["dataset_manifest"]),
        tile_views=tile["views"],
        renderer_mask_manifest_path=(
            None if raw.get("renderer_mask_manifest") is None else Path(raw["renderer_mask_manifest"])
        ),
    )
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    samples = [dataset.camera_sample(index) for index in range(total)]
    started = time.time()
    views = render_backdrops(
        backend=backend,
        torch=torch_mod,
        params=gpu_params,
        samples=samples,
        output_root=args.output,
        background_rgb=tuple(args.background),
        downsample=args.downsample,
        save_threads=args.save_threads,
        verify_against=args.verify_against,
    )
    print(f"rendered {len(views)} views in {(time.time() - started) / 60.0:.1f} min")
    if args.limit is not None:
        print(f"limit={args.limit}: smoke run, no manifest written")
        return 0
    expected = {str(view["sample_id"]) for view in tile["views"]}
    if set(views) != expected:
        raise ValueError(
            f"rendered views ({len(views)}) do not match the Tile's view set ({len(expected)})"
        )
    provenance["render_minutes"] = round((time.time() - started) / 60.0, 1)
    signature = write_standin_manifest(
        args.output / "background_manifest.json",
        views=views,
        tile_id=int(tile["tile_id"]),
        tile_inputs_manifest_sha256=tile_inputs_sha,
        dome_path=args.dome,
        dome_sha256=sha256_file(args.dome),
        background_rgb=tuple(args.background),
        downsample=args.downsample,
        provenance=provenance,
    )
    print(f"{len(views)} views -> {args.output / 'background_manifest.json'} ({signature[:8]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
