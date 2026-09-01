"""Conservative production policy for multi-view LiDAR-mesh filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOURCE_INVALID = 0
SOURCE_LIDAR_ANCHOR = 2
SOURCE_CROSS_VIEW_SUPPORTED = 3
SOURCE_UNOBSERVABLE_RETAINED = 4


@dataclass(frozen=True)
class CrossViewFilterConfig:
    min_observations: int = 2
    min_consistent: int = 1
    min_support_fraction: float = 0.5
    min_conflicts_to_reject: int = 2
    unsupported_confidence: float = 0.2
    single_support_confidence: float = 0.6

    def validate(self) -> None:
        if self.min_observations < 1 or self.min_consistent < 1:
            raise ValueError("observation thresholds must be positive")
        if self.min_conflicts_to_reject < 1:
            raise ValueError("min_conflicts_to_reject must be positive")
        for name in (
            "min_support_fraction",
            "unsupported_confidence",
            "single_support_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


def classify_cross_view_support(
    *,
    source_valid: np.ndarray,
    lidar_anchor: np.ndarray,
    observed: np.ndarray,
    consistent: np.ndarray,
    conflicts: np.ndarray,
    config: CrossViewFilterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered valid, confidence and per-pixel source type.

    Occluded-only and unobservable pixels are retained at low confidence:
    another camera cannot disprove a surface it cannot see. Rejection is
    limited to pixels observed repeatedly in front of the target mesh without
    any consistent support. Native LiDAR anchors always win.
    """

    config.validate()
    source_valid = np.asarray(source_valid, dtype=bool)
    lidar_anchor = np.asarray(lidar_anchor, dtype=bool)
    observed = np.asarray(observed, dtype=np.int16)
    consistent = np.asarray(consistent, dtype=np.int16)
    conflicts = np.asarray(conflicts, dtype=np.int16)
    if not (
        source_valid.shape
        == lidar_anchor.shape
        == observed.shape
        == consistent.shape
        == conflicts.shape
    ):
        raise ValueError("cross-view support arrays must share one shape")

    support_fraction = np.divide(
        consistent,
        observed,
        out=np.zeros_like(consistent, dtype=np.float32),
        where=observed > 0,
    )
    supported = consistent >= config.min_consistent
    supported &= (observed < config.min_observations) | (
        support_fraction >= config.min_support_fraction
    )
    contradicted = observed >= config.min_observations
    contradicted &= conflicts >= config.min_conflicts_to_reject
    contradicted &= consistent == 0

    filtered_valid = source_valid & (~contradicted | lidar_anchor)
    confidence = np.zeros(source_valid.shape, dtype=np.float32)
    source_type = np.zeros(source_valid.shape, dtype=np.uint8)

    unresolved = filtered_valid & ~supported & ~lidar_anchor
    confidence[unresolved] = float(config.unsupported_confidence)
    source_type[unresolved] = SOURCE_UNOBSERVABLE_RETAINED

    supported_only = filtered_valid & supported & ~lidar_anchor
    observation_factor = np.minimum(observed.astype(np.float32) / 3.0, 1.0)
    confidence[supported_only] = np.maximum(
        float(config.single_support_confidence),
        0.5
        + 0.5
        * support_fraction[supported_only]
        * observation_factor[supported_only],
    )
    source_type[supported_only] = SOURCE_CROSS_VIEW_SUPPORTED

    anchored = filtered_valid & lidar_anchor
    confidence[anchored] = 1.0
    source_type[anchored] = SOURCE_LIDAR_ANCHOR
    return filtered_valid, confidence, source_type
