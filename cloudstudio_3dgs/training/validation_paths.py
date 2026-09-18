"""Where the held-out battery reads its views from, derived from a trainer config.

``tools/evaluate_probe_views.py`` has always scored the *validation* caches that sit next
to the training caches a config names: ``face4`` next to ``face4_val``, ``renderer_mask_train``
next to ``renderer_mask_val``, and so on. Nothing records those validation paths - they are
derived by name - so the rule lives here, once, and the SDK checks at prepare time that the
derived caches exist rather than finding out after a full training run.

The rule is a name substitution, not a directory scan: a dataset version whose training
cache is ``face4_train`` (v9) maps to ``face4_val``, the same as one whose training cache is
plain ``face4`` (v8). A naive ``"face4" -> "face4_val"`` replace turned the former into
``face4_val_train``, which is how the first SDK delivery died at the battery.
"""

from __future__ import annotations

from typing import Any

FACE_KEYS = ("face_cache_manifest", "face_cache_root")
TRAIN_VAL_KEYS = (
    "renderer_mask_manifest",
    "face_lidar_geometry_manifest",
    "face_lidar_geometry_root",
    "background_image_manifest",
    "background_image_root",
)


def validation_face_path(path: str) -> str:
    """``face4_train`` and bare ``face4`` both map to ``face4_val``; ``face4_val`` stays."""
    if "face4_val" in path:
        return path
    if "face4_train" in path:
        return path.replace("face4_train", "face4_val")
    return path.replace("face4", "face4_val")


def validation_train_path(path: str) -> str:
    return path.replace("_train", "_val")


def derive_validation_paths(raw: dict[str, Any]) -> dict[str, str]:
    """The validation-side path for every dataset key the battery reads.

    Keys absent from ``raw`` (an optional background library) are absent from the result.
    """
    out: dict[str, str] = {}
    for key in FACE_KEYS:
        value = raw.get(key)
        if value:
            out[key] = validation_face_path(str(value))
    for key in TRAIN_VAL_KEYS:
        value = raw.get(key)
        if value:
            out[key] = validation_train_path(str(value))
    return out
