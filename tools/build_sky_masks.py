#!/usr/bin/env python3
"""Build per-face SegFormer-B4 (ADE20K) sky masks for a Face4 cache.

Research supervision data for the "sky is special" training arm. Per face:

1. read the face RGB PNG and its valid mask (``mask_path``) from the face
   cache;
2. resize the RGB so its SHORT side is 512 px, keeping the aspect ratio
   (Face4 faces are exactly 2:1, so 1456x2912 -> 512x1024 and 2912x1456 ->
   1024x512; PIL bilinear, antialiased). The whole resized face goes through
   the network in one pass - the same test-time protocol as the SegFormer
   ADE20K reference (mmseg ``Resize(keep_ratio)`` + ``mode='whole'``). No
   square squash (the HF processor's default 512x512 would distort 2:1
   faces) and no tiling (SegFormer has no positional embeddings, so an
   arbitrary 2:1 input is in-distribution);
3. normalise with the checkpoint's ImageNet mean/std, run the model on CPU,
   bilinearly upsample the 150-class logits from 1/4 scale to the model input
   size (the HF ``post_process_semantic_segmentation`` convention), softmax,
   and take channel 2 - "sky" by ADE20K label id, never by name (id 48 is
   "skyscraper");
4. bilinearly upsample the sky PROBABILITY map to the face resolution and
   threshold at >= 0.5. A probability >= 0.5 implies argmax == sky, so this
   is the stricter (higher precision) variant of the argmax rule; the
   argmax-only sky fraction at model scale is recorded per face for
   comparison;
5. AND with the face valid mask, write a uint8 PNG (255 = sky, 0 otherwise)
   to ``<output-root>/faces/<image_id>_<face_id>_sky.png`` and append a
   record to ``records.jsonl`` (resumable: a face is skipped when its PNG
   exists, its sha matches the record, and the record carries the current
   rule fingerprint);
6. assemble and sign ``<output-root>/sky_mask_<split>.json`` via
   :mod:`cloudstudio_3dgs.data.sky_masks`.

CPU only by default: the GPU is reserved for training. ``CUDA_VISIBLE_DEVICES``
is emptied before torch is imported unless ``--allow-cuda`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_ID = "nvidia/segformer-b4-finetuned-ade-512-512"
LICENSE_NOTE = (
    "NVIDIA Source Code License-NC; research supervision data only, "
    "weights not shipped"
)
SMOKE_FACE_PLAN = (
    "pitch_up_56",
    "pitch_up_56",
    "yaw_minus_35",
    "yaw_plus_35",
    "pitch_down_56",
    "pitch_down_56",
)


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _lower_priority() -> None:
    """Run below normal so the trainer's data loading is never starved."""
    if os.name != "nt":
        try:
            os.nice(10)
        except OSError:
            pass
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    # Without explicit types the pseudo-handle (-1) is truncated to a 32-bit
    # int and SetPriorityClass fails silently, leaving the process at Normal.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    below_normal = 0x00004000
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal):
        _log(f"warning: could not lower process priority (error {ctypes.get_last_error()})")


def _parse_select(value: str | None) -> set[tuple[str, str]] | None:
    if not value:
        return None
    selected: set[tuple[str, str]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "::" not in item:
            raise SystemExit(f"--select entries must be image_id::face_id, got {item!r}")
        image_id, face_id = item.split("::", 1)
        selected.add((image_id, face_id))
    return selected


def _smoke_selection(face_manifest: dict[str, Any]) -> set[tuple[str, str]]:
    """Six faces spread over the capture: 2 pitch_up, 2 yaw, 2 pitch_down."""
    images = face_manifest["images"]
    picks: set[tuple[str, str]] = set()
    count = len(SMOKE_FACE_PLAN)
    for index, face_id in enumerate(SMOKE_FACE_PLAN):
        image = images[(index * len(images)) // count]
        if face_id not in {str(face["face_id"]) for face in image["faces"]}:
            raise SystemExit(f"image {image['image_id']} has no face {face_id}")
        picks.add((str(image["image_id"]), face_id))
    return picks


class SkySegmenter:
    """SegFormer wrapper: RGB uint8 (H, W, 3) -> sky probability at face scale."""

    def __init__(self, model_id: str, *, short_side: int, threads: int, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialised by an earlier parallel op
        self.torch = torch
        self.F = torch.nn.functional
        self.device = device
        self.short_side = int(short_side)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        self.model.eval().to(device)
        self.mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        from cloudstudio_3dgs.data.sky_masks import ADE20K_SKY_LABEL_ID

        if id2label.get(ADE20K_SKY_LABEL_ID) != "sky":
            raise SystemExit(
                f"label id {ADE20K_SKY_LABEL_ID} of {model_id} is "
                f"{id2label.get(ADE20K_SKY_LABEL_ID)!r}, not 'sky'"
            )
        self.sky_id = ADE20K_SKY_LABEL_ID
        self.id2label = id2label
        self.identity = self._identity(model_id)

    def _identity(self, model_id: str) -> dict[str, Any]:
        commit = str(getattr(self.model.config, "_commit_hash", "") or "")
        snapshot: Path | None = None
        config_sha = weights_sha = weights_file = ""
        if commit:
            from huggingface_hub import constants

            cache = Path(constants.HF_HUB_CACHE)
            candidate = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / commit
            if candidate.is_dir():
                snapshot = candidate
        if snapshot is not None:
            config_path = snapshot / "config.json"
            if config_path.is_file():
                config_sha = _sha256_file(config_path)
            for name in ("model.safetensors", "pytorch_model.bin"):
                weights = snapshot / name
                if weights.is_file():
                    weights_file = name
                    weights_sha = _sha256_file(weights)
                    break
        if not (commit and config_sha and weights_sha):
            raise SystemExit(
                "could not pin the model identity (commit/config/weights); refusing "
                "to write masks whose provenance cannot be recorded"
            )
        import torch
        import transformers

        return {
            "id": model_id,
            "revision": commit,
            "config_sha256": config_sha,
            "weights_file": weights_file,
            "weights_sha256": weights_sha,
            "num_labels": len(self.id2label),
            "image_mean": [float(v) for v in self.processor.image_mean],
            "image_std": [float(v) for v in self.processor.image_std],
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": self.device,
            "license_note": LICENSE_NOTE,
        }

    def model_input_size(self, width: int, height: int) -> tuple[int, int]:
        scale = self.short_side / min(width, height)
        return max(1, round(width * scale)), max(1, round(height * scale))

    def sky_probability(self, rgb_image) -> tuple[Any, float]:
        """Return (sky probability at face resolution [H, W] float32, argmax sky fraction at model scale)."""
        import numpy as np
        from PIL import Image

        torch = self.torch
        width, height = rgb_image.size
        in_w, in_h = self.model_input_size(width, height)
        resized = rgb_image.resize((in_w, in_h), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        pixels = ((pixels - self.mean) / self.std).to(self.device)
        with torch.inference_mode():
            logits = self.model(pixel_values=pixels).logits
            logits = self.F.interpolate(
                logits.float(), size=(in_h, in_w), mode="bilinear", align_corners=False
            )
            probs = logits.softmax(dim=1)
            argmax_sky = float((probs.argmax(dim=1) == self.sky_id).float().mean().item())
            sky = probs[:, self.sky_id : self.sky_id + 1]
            sky_full = self.F.interpolate(
                sky, size=(height, width), mode="bilinear", align_corners=False
            )
        return sky_full[0, 0].cpu().numpy(), argmax_sky


def _rule(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "decision": (
            f"sky_probability >= {args.prob_threshold} at face resolution "
            "(implies argmax == sky at model scale)"
        ),
        "sky_probability_threshold": float(args.prob_threshold),
        "model_input": (
            f"short side resized to {args.short_side} px keeping aspect "
            "(PIL bilinear, antialiased), whole-image single pass, no tiling, "
            "no square squash"
        ),
        "logits_to_probability": (
            "150-class logits bilinearly upsampled (align_corners=False) from 1/4 "
            "scale to the model input size, then softmax; channel 2 taken by id"
        ),
        "upsampling": (
            "sky probability bilinearly upsampled (align_corners=False) from the "
            "model input size to the face size, threshold applied after upsampling"
        ),
        "valid_mask": "AND face cache mask_path != 0 (circle/FoV/face-weight/person validity)",
        "mask_encoding": "uint8 PNG, 255 = sky AND valid, 0 otherwise",
        "sky_fraction_denominator": "valid_pixels",
        "argmax_sky_fraction_model_scale": (
            "per-record diagnostic: fraction of model-scale pixels (valid mask NOT "
            "applied) whose argmax is sky; compares the argmax rule with the "
            "probability rule"
        ),
    }


def _rule_fingerprint(model_identity: dict[str, Any], rule: dict[str, Any]) -> str:
    from cloudstudio_3dgs.data.manifest import canonical_json_bytes

    body = {
        "model": {
            k: model_identity[k]
            for k in ("id", "revision", "config_sha256", "weights_sha256", "image_mean", "image_std")
        },
        "rule": rule,
    }
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _load_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[(str(record["image_id"]), str(record["face_id"]))] = record
    return records


def _reusable(record: dict[str, Any] | None, png: Path, fingerprint: str) -> bool:
    if record is None or record.get("rule_fingerprint") != fingerprint or not png.is_file():
        return False
    return _sha256_file(png) == str(record["mask_sha256"])


def _write_png_atomic(path: Path, array) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array, mode="L").save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)
    return _sha256_file(path)


def _contact_sheet(entries: list[dict[str, Any]], face_root: Path, output: Path, *, cell: int = 720) -> None:
    """Photo with a blue sky overlay per processed face, tiled 3 per row."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    entries = entries[:12]
    columns = 3
    rows = (len(entries) + columns - 1) // columns
    label_h = 36
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    for index, entry in enumerate(entries):
        with Image.open(face_root / entry["rgb_path"]) as source:
            rgb = source.convert("RGB")
            scale = cell / max(rgb.size)
            small = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.BILINEAR)
        with Image.open(entry["mask_file"]) as source:
            mask = source.convert("L").resize(small.size, Image.Resampling.NEAREST)
        photo = np.asarray(small, dtype=np.float32)
        sky = np.asarray(mask) != 0
        tint = np.array([40.0, 90.0, 255.0], dtype=np.float32)
        photo[sky] = photo[sky] * 0.45 + tint * 0.55
        tile = Image.fromarray(photo.clip(0, 255).astype(np.uint8))
        x0 = (index % columns) * cell + (cell - tile.width) // 2
        y0 = (index // columns) * (cell + label_h) + label_h
        sheet.paste(tile, (x0, y0))
        label = (
            f"{entry['face_id']} {entry['image_id'][4:12]} "
            f"sky {100 * entry['sky_fraction']:.1f}% "
            f"(argmax {100 * entry['argmax_sky_fraction_model_scale']:.1f}%)"
        )
        draw.text(((index % columns) * cell + 8, (index // columns) * (cell + label_h) + 8), label, fill=(255, 255, 255), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--face-manifest", required=True, type=Path)
    parser.add_argument("--face-cache-root", type=Path, default=None, help="defaults to the face manifest's directory")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--select", default=None, help="comma list of image_id::face_id to process (no manifest is written)")
    parser.add_argument("--smoke", action="store_true", help="process 6 spread faces (2 pitch_up, 2 yaw, 2 pitch_down); no manifest")
    parser.add_argument("--contact-sheet", type=Path, default=None, help="write a JPEG overlay grid of the faces processed in this run")
    parser.add_argument("--no-resume", action="store_true", help="recompute every face even if a matching PNG exists")
    parser.add_argument("--allow-cuda", action="store_true", help="do not blank CUDA_VISIBLE_DEVICES (default: CPU only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-below-normal-priority", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not args.allow_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if args.device != "cpu":
            raise SystemExit("--device other than cpu requires --allow-cuda")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    if not args.no_below_normal_priority:
        _lower_priority()

    import numpy as np
    from PIL import Image

    from cloudstudio_3dgs.data.sky_masks import (
        SKY_MASK_VALUE,
        build_sky_mask_manifest,
        load_sky_mask_manifest,
        sky_mask_path_for,
    )
    from cloudstudio_3dgs.training.face_dataset import verify_face_manifest

    face_manifest_path = args.face_manifest.resolve()
    face_root = (args.face_cache_root or face_manifest_path.parent).resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _log(f"pid {os.getpid()}  face manifest {face_manifest_path}  output {output_root}")

    face_manifest = json.loads(face_manifest_path.read_text(encoding="utf-8"))
    face_sha = verify_face_manifest(face_manifest)
    split = str(face_manifest.get("split", ""))
    _log(f"face manifest verified: split={split} sha={face_sha}")

    selected = _parse_select(args.select)
    if args.smoke:
        selected = (selected or set()) | _smoke_selection(face_manifest)
    partial = selected is not None

    jobs: list[dict[str, Any]] = []
    for image in face_manifest["images"]:
        for face in image["faces"]:
            key = (str(image["image_id"]), str(face["face_id"]))
            if selected is not None and key not in selected:
                continue
            jobs.append(
                {
                    "image_id": key[0],
                    "camera_id": str(image["camera_id"]),
                    "face_id": key[1],
                    "rgb_path": str(face["rgb_path"]),
                    "mask_path": str(face["mask_path"]),
                }
            )
    if selected is not None and len(jobs) != len(selected):
        raise SystemExit(f"selection has {len(selected)} keys but only {len(jobs)} matched the face manifest")
    _log(f"{len(jobs)} faces to consider ({'partial selection' if partial else 'full split'})")

    t_model = time.time()
    segmenter = SkySegmenter(args.model_id, short_side=args.short_side, threads=args.threads, device=args.device)
    rule = _rule(args)
    fingerprint = _rule_fingerprint(segmenter.identity, rule)
    _log(
        f"model ready in {time.time() - t_model:.1f}s: {segmenter.identity['id']}@{segmenter.identity['revision'][:12]} "
        f"weights={segmenter.identity['weights_file']} sha={segmenter.identity['weights_sha256'][:12]} "
        f"threads={args.threads} device={args.device} rule_fingerprint={fingerprint[:12]}"
    )

    records_path = output_root / "records.jsonl"
    existing = {} if args.no_resume else _load_records(records_path)
    processed: list[dict[str, Any]] = []
    skipped = 0
    done = 0
    face_seconds: list[float] = []
    t_loop = time.time()
    with records_path.open("a", encoding="utf-8", newline="\n") as records_stream:
        for job in jobs:
            key = (job["image_id"], job["face_id"])
            relative = sky_mask_path_for(*key)
            png = output_root / Path(*relative.split("/"))
            prior = existing.get(key)
            if _reusable(prior, png, fingerprint):
                skipped += 1
                continue
            t0 = time.time()
            with Image.open(face_root / job["rgb_path"]) as source:
                rgb = source.convert("RGB")
                width, height = rgb.size
                probability, argmax_sky = segmenter.sky_probability(rgb)
            with Image.open(face_root / job["mask_path"]) as source:
                valid = np.asarray(source.convert("L"), dtype=np.uint8) != 0
            if valid.shape != (height, width):
                raise SystemExit(f"valid mask size mismatch for {key}")
            sky = (probability >= args.prob_threshold) & valid
            array = np.where(sky, np.uint8(SKY_MASK_VALUE), np.uint8(0)).astype(np.uint8)
            sha = _write_png_atomic(png, array)
            valid_pixels = int(np.count_nonzero(valid))
            sky_pixels = int(np.count_nonzero(sky))
            record = {
                "image_id": key[0],
                "camera_id": job["camera_id"],
                "face_id": key[1],
                "width": int(width),
                "height": int(height),
                "mask_path": relative,
                "mask_sha256": sha,
                "valid_pixels": valid_pixels,
                "sky_pixels": sky_pixels,
                "sky_fraction": (sky_pixels / valid_pixels) if valid_pixels else 0.0,
                "argmax_sky_fraction_model_scale": float(argmax_sky),
                "rule_fingerprint": fingerprint,
            }
            records_stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            records_stream.flush()
            existing[key] = record
            elapsed = time.time() - t0
            face_seconds.append(elapsed)
            done += 1
            processed.append({**record, "rgb_path": job["rgb_path"], "mask_file": png})
            if done % args.log_every == 0 or done == 1 or done + skipped == len(jobs):
                remaining = len(jobs) - done - skipped
                mean = sum(face_seconds) / len(face_seconds)
                _log(
                    f"{done + skipped}/{len(jobs)} (computed {done}, reused {skipped}) "
                    f"last {key[0]}::{key[1]} sky={100 * record['sky_fraction']:.1f}% "
                    f"{elapsed:.2f}s  mean {mean:.2f}s/face  ETA {remaining * mean / 60:.1f} min"
                )
    _log(
        f"loop done in {(time.time() - t_loop) / 60:.1f} min: computed {done}, reused {skipped}"
        + (f", mean {sum(face_seconds) / len(face_seconds):.2f}s/face" if face_seconds else "")
    )

    if args.contact_sheet is not None:
        sheet_entries = processed or [
            {**existing[(j["image_id"], j["face_id"])], "rgb_path": j["rgb_path"], "mask_file": output_root / Path(*sky_mask_path_for(j["image_id"], j["face_id"]).split("/"))}
            for j in jobs
        ]
        _contact_sheet(sheet_entries, face_root, args.contact_sheet.resolve())
        _log(f"contact sheet written: {args.contact_sheet.resolve()}")
    for entry in processed:
        _log(
            f"  {entry['face_id']:14s} {entry['image_id']}  sky={100 * entry['sky_fraction']:.2f}% of valid  "
            f"argmax@model={100 * entry['argmax_sky_fraction_model_scale']:.2f}%"
        )

    if partial:
        _log("partial selection: no manifest written")
        return 0

    records: list[dict[str, Any]] = []
    for job in jobs:
        key = (job["image_id"], job["face_id"])
        record = existing.get(key)
        png = output_root / Path(*sky_mask_path_for(*key).split("/"))
        if not _reusable(record, png, fingerprint):
            raise SystemExit(f"face {key} has no verified sky mask after the loop")
        records.append({k: v for k, v in record.items() if k != "rule_fingerprint"})
    manifest = build_sky_mask_manifest(
        split=split,
        source_face_manifest_sha256=face_sha,
        source_identity=dict(face_manifest.get("source_identity", {})),
        model=segmenter.identity,
        rule=rule,
        records=records,
    )
    manifest_path = output_root / f"sky_mask_{split}.json"
    _atomic_json(manifest_path, manifest)
    reloaded = load_sky_mask_manifest(manifest_path, expected_face_manifest_sha256=face_sha)
    summary = reloaded["summary"]
    _log(
        f"manifest written: {manifest_path} sha={reloaded['sky_mask_manifest_sha256'][:12]} "
        f"faces={summary['face_count']} images={summary['image_count']} "
        f"mean_sky={100 * summary['mean_sky_fraction']:.2f}% "
        f"faces_gt_10pct={summary['faces_with_sky_gt_10pct']} faces_without_sky={summary['faces_without_sky']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
