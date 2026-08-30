"""Per-view prerendered backgrounds for training-time compositing.

A baked far-field layer never changes, so instead of rasterizing its
gaussians every step (and having to shield them from the optimizer), each
training view's backdrop is rendered once offline and composited into the
loss as ``final = render + (1 - alpha) * background_view``. The trainable
set then contains surface gaussians only, which frees the population
lifecycle from any row-identity bookkeeping.

Fail-closed: when a library is configured, every sampled view must have a
stored background - a missing view is an error, never a silent fall back to
the constant background colour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ViewBackgroundLibrary:
    """Loads per-view background images and serves them at render size."""

    def __init__(self, manifest_path: Path, root: Path, *, device: str) -> None:
        import numpy as np

        manifest_path = Path(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = payload.get("manifest_sha256")
        body = dict(payload)
        body.pop("manifest_sha256", None)
        actual = _sha256_bytes(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if recorded != actual:
            raise ValueError(
                f"view background manifest signature mismatch: {manifest_path}"
            )
        views = payload.get("views")
        if not isinstance(views, dict) or not views:
            raise ValueError(f"view background manifest lists no views: {manifest_path}")
        self.manifest_sha256 = recorded
        self.root = Path(root)
        self.device = device
        self.views = views
        self._np = np
        self._cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.views)

    def background_for(self, image_id: str, *, height: int, width: int, torch: Any):
        """Return the (height, width, 3) float background for one view."""
        entry = self.views.get(image_id)
        if entry is None:
            raise ValueError(
                f"view {image_id!r} has no prerendered background; the "
                "library is fail-closed by design"
            )
        cached = self._cache.get(image_id)
        if cached is None:
            from PIL import Image

            with Image.open(self.root / entry["file"]) as image:
                cached = self._np.asarray(image.convert("RGB"), dtype=self._np.uint8)
            self._cache[image_id] = cached
        tensor = (
            torch.as_tensor(cached, device=self.device, dtype=torch.float32) / 255.0
        )
        if tensor.shape[0] != height or tensor.shape[1] != width:
            tensor = torch.nn.functional.interpolate(
                tensor.permute(2, 0, 1).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        return tensor


def write_view_background_manifest(
    path: Path, *, views: dict[str, dict], metadata: dict
) -> str:
    """Write a signed manifest; returns the signature."""
    body = {"schema_version": 1, **metadata, "views": views}
    signature = _sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    body["manifest_sha256"] = signature
    Path(path).write_text(
        json.dumps(body, indent=1), encoding="utf-8"
    )
    return signature
