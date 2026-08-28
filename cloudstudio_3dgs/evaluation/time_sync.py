"""Camera-pose interpolation helpers for timestamp-offset audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class PoseTrajectory:
    """One physical camera's strictly ordered camera-to-world trajectory."""

    timestamps_ns: np.ndarray
    translations: np.ndarray
    rotations: Rotation

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], side: str) -> "PoseTrajectory":
        records = sorted(
            (
                image
                for image in manifest.get("images", [])
                if str(image.get("side")) == side
            ),
            key=lambda image: int(image["timestamp_ns"]),
        )
        if len(records) < 2:
            raise ValueError(f"camera side {side!r} needs at least two poses")
        timestamps = np.asarray(
            [int(image["timestamp_ns"]) for image in records], dtype=np.int64
        )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"camera side {side!r} timestamps are not strictly increasing")
        matrices = np.asarray([image["c2w"] for image in records], dtype=np.float64)
        if matrices.shape != (len(records), 4, 4) or not np.all(np.isfinite(matrices)):
            raise ValueError(f"camera side {side!r} contains an invalid c2w matrix")
        return cls(
            timestamps_ns=timestamps,
            translations=matrices[:, :3, 3].copy(),
            rotations=Rotation.from_matrix(matrices[:, :3, :3]),
        )

    def interpolate(self, timestamp_ns: int, offset_ms: float = 0.0) -> np.ndarray:
        """Interpolate the pose at ``timestamp + offset`` without extrapolation."""

        offset_ns = int(round(float(offset_ms) * 1_000_000.0))
        query_ns = int(timestamp_ns) + offset_ns
        first = int(self.timestamps_ns[0])
        last = int(self.timestamps_ns[-1])
        if query_ns < first or query_ns > last:
            raise ValueError(
                f"shifted timestamp {query_ns} is outside trajectory [{first}, {last}]"
            )
        origin = first
        times_s = (self.timestamps_ns - origin).astype(np.float64) / 1_000_000_000.0
        query_s = (query_ns - origin) / 1_000_000_000.0
        translation = np.stack(
            [
                np.interp(query_s, times_s, self.translations[:, axis])
                for axis in range(3)
            ]
        )
        rotation = Slerp(times_s, self.rotations)([query_s]).as_matrix()[0]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation
        result[:3, 3] = translation
        return result


def trajectories_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, PoseTrajectory]:
    """Build the left/right camera trajectories used by a sync sweep."""

    sides = sorted({str(image.get("side")) for image in manifest.get("images", [])})
    if sides != ["left", "right"]:
        raise ValueError(f"expected left/right camera sides, got {sides}")
    return {side: PoseTrajectory.from_manifest(manifest, side) for side in sides}
