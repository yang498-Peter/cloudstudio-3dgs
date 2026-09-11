"""ROI-only sharpness on diagnostic compare strips.

``score_compare_sharpness.py`` scores whole panels, so a compare frame that
looks at sky or at content outside the Tile drags a *region* score down even
though the region itself is fine.  This scorer crops the region-of-interest box
that ``build_diagnostic_set.py`` recorded for each Tile view (``roi_in_crops``,
array indices inside the Tile crop, i.e. directly the compare-panel pixels)
from the photo / ours / reference panels and reports Laplacian-variance ratios
on those boxes only.  Frames without an ROI box, or with fewer than
``--min-samples`` region samples inside the crop, are listed as skipped.

usage: python tools/score_compare_roi.py --selection <region>/DIAG_40/selection.json
           [--min-samples 10] [--json OUT] <arm>/compare [<arm>/compare ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def laplacian_variance(panel) -> float:
    import cv2

    return float(cv2.Laplacian(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def split_panels(strip, gap: int = 8):
    """The strip is photo | ours | reference at native crop size with ``gap`` px between."""
    width = (strip.shape[1] - 2 * gap) // 3
    return [strip[:, i * (width + gap): i * (width + gap) + width] for i in range(3)]


def roi_boxes(selection: dict) -> dict:
    return {
        entry["sample_id"]: entry["roi"]
        for entry in selection.get("roi_in_crops", [])
        if entry.get("roi")
    }


def score_compare_dir(compare_dir: Path, boxes: dict, min_samples: int, *, imread=None, var=laplacian_variance) -> dict:
    summary = json.loads((compare_dir / "compare_summary.json").read_text(encoding="utf-8"))
    if imread is None:
        import cv2

        imread = cv2.imread
    rows, skipped = [], []
    for frame in summary["frames"]:
        sample_id = frame["image_id"]
        roi = boxes.get(sample_id)
        if roi is None:
            skipped.append({"sample_id": sample_id, "reason": "no ROI box in this Tile view"})
            continue
        if int(roi.get("samples_in_crop", 0)) < min_samples:
            skipped.append({"sample_id": sample_id, "reason": f"only {roi.get('samples_in_crop', 0)} region samples in crop"})
            continue
        strip = imread(str(compare_dir / frame["file"]))
        if strip is None:
            raise FileNotFoundError(compare_dir / frame["file"])
        x0, y0, x1, y1 = (int(roi[k]) for k in ("x0", "y0", "x1", "y1"))
        if x1 - x0 < 8 or y1 - y0 < 8:
            skipped.append({"sample_id": sample_id, "reason": f"ROI box {x1 - x0}x{y1 - y0} too small"})
            continue
        panels = split_panels(strip)
        crop_w, crop_h = int(roi["crop"]["width"]), int(roi["crop"]["height"])
        if panels[0].shape[1] != crop_w or panels[0].shape[0] < crop_h:
            raise ValueError(
                f"{frame['file']}: panel {panels[0].shape[1]}x{panels[0].shape[0]} does not match the "
                f"recorded Tile crop {crop_w}x{crop_h}; ROI boxes cannot be applied"
            )
        photo, ours, ref = (var(p[y0:y1, x0:x1]) for p in panels)
        rows.append({
            "sample_id": sample_id, "file": frame["file"], "box": [x0, y0, x1, y1],
            "samples_in_crop": int(roi["samples_in_crop"]),
            "lv_photo": photo, "lv_ours": ours, "lv_ref": ref,
            "ours_over_photo": ours / photo if photo else None,
            "ref_over_photo": ref / photo if photo else None,
            "ours_over_ref": ours / ref if ref else None,
        })
    scored = [r for r in rows if r["ours_over_photo"] is not None and r["ref_over_photo"] is not None]
    result = {
        "compare_dir": str(compare_dir), "arm": compare_dir.parent.name, "n": len(scored),
        "n_skipped": len(skipped), "frames": rows, "skipped": skipped,
    }
    if scored:
        result["ours_over_photo_median"] = statistics.median(r["ours_over_photo"] for r in scored)
        result["ref_over_photo_median"] = statistics.median(r["ref_over_photo"] for r in scored)
        result["ours_over_ref_median"] = statistics.median(r["ours_over_ref"] for r in scored if r["ours_over_ref"] is not None)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True, help="diagnostic selection.json carrying roi_in_crops")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--json", type=Path, help="write all per-frame rows here")
    parser.add_argument("compare_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    boxes = roi_boxes(json.loads(args.selection.read_text(encoding="utf-8")))
    results = []
    for compare_dir in args.compare_dirs:
        result = score_compare_dir(compare_dir, boxes, args.min_samples)
        results.append(result)
        if result["n"]:
            print(
                f"{result['arm']} ROI n {result['n']} (skipped {result['n_skipped']}) "
                f"sharpness ours/photo median {result['ours_over_photo_median']:.3f}  "
                f"ref/photo median {result['ref_over_photo_median']:.3f}  "
                f"ours/ref median {result['ours_over_ref_median']:.3f}"
            )
        else:
            print(f"{result['arm']} ROI n 0 (skipped {result['n_skipped']})")
        for row in result["frames"]:
            print(
                f"  {row['sample_id']:48s} box {row['box']} samples {row['samples_in_crop']:4d}  "
                f"ours/photo {row['ours_over_photo']:.3f}  ref/photo {row['ref_over_photo']:.3f}  ours/ref {row['ours_over_ref']:.3f}"
            )
        for skip in result["skipped"]:
            print(f"  skipped {skip['sample_id']}: {skip['reason']}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
