"""Two controls for the multi-view consistency number.

1. Synthetic shift: match a patch against the SAME image displaced by a known
   +3/-2 px. A matcher that cannot recover a known shift cannot measure an
   unknown one.
2. Confidence sweep: does the measured disagreement fall as the NCC peak rises?
   A real registration error stays put; matcher noise collapses.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Peter\cloudstudio-3dgs-work")
from cloudstudio_3dgs.training.face_dataset import FaceCacheDataset

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
raw = json.loads((RUN / "tile0_G1_flatten033_t0_20k.json").read_text(encoding="utf-8"))
tiles = json.loads(Path(raw["tile_inputs_manifest"]).read_text(encoding="utf-8"))["tiles"]
tile = [t for t in tiles if int(t["tile_id"]) == int(raw.get("mipmap_tile_id", 0))][0]
dataset = FaceCacheDataset(
    Path(raw["face_cache_manifest"]),
    Path(raw["face_cache_root"]),
    verify_artifacts=False,
    dataset_manifest_path=Path(raw["dataset_manifest"]),
    renderer_mask_manifest_path=Path(raw["renderer_mask_manifest"]),
    tile_views=tile["views"],
)

PATCH, SEARCH, half = 21, 18, 10
TRUE_DX, TRUE_DY = 3, -2
rng = np.random.default_rng(1)
errors = []
for index in range(0, len(dataset), max(1, len(dataset) // 12)):
    sample = dataset[index]
    grey = cv2.cvtColor(np.asarray(sample.image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    shifted = np.roll(np.roll(grey, TRUE_DY, axis=0), TRUE_DX, axis=1)
    margin = half + SEARCH + 4
    for _ in range(40):
        x = int(rng.integers(margin, sample.width - margin))
        y = int(rng.integers(margin, sample.height - margin))
        template = grey[y - half : y + half + 1, x - half : x + half + 1]
        if template.std() < 12:
            continue
        window = shifted[y - half - SEARCH : y + half + SEARCH + 1, x - half - SEARCH : x + half + SEARCH + 1]
        result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(result)
        errors.append((abs(loc[0] - SEARCH - TRUE_DX) + abs(loc[1] - SEARCH - TRUE_DY), peak))
    if len(errors) > 300:
        break
exact = sum(1 for e, _ in errors if e == 0) / len(errors)
print(f"control 1 - known shift ({TRUE_DX:+d},{TRUE_DY:+d}) on {len(errors)} patches:")
print(f"  recovered exactly: {exact:.1%}   median |error| {np.median([e for e, _ in errors]):.1f} px   median peak {np.median([p for _, p in errors]):.2f}")

rows = json.loads((Path(__file__).with_name("multiview_consistency.json")).read_text())
print(f"\ncontrol 2 - confidence sweep over {len(rows)} cross-view matches (search +/-18 px):")
for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
    kept = [r for r in rows if r["peak"] >= threshold]
    if len(kept) < 15:
        print(f"  peak >= {threshold:.2f}: only {len(kept)} matches, skipped")
        continue
    radial = np.hypot([r["dx"] for r in kept], [r["dy"] for r in kept])
    print(f"  peak >= {threshold:.2f}: n={len(kept):4d}  median |disagreement| {np.median(radial):5.2f} px   p90 {np.percentile(radial, 90):5.2f}")
