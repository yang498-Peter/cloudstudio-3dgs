"""Explicit topology and phase policy for LiDAR-first Gaussian training.

The policy separates three experiments which older configurations represented
indirectly by moving a refinement window beyond ``max_steps``:

* ``strict_fixed`` keeps every initialized Gaussian for the entire run;
* ``opacity_prune_only`` permits one auditable opacity cull but no births;
* ``adaptive_growth`` enables the existing split/clone/relocate lifecycle.

The fixed-topology schedule is deliberately orthogonal to the topology mode.
It controls when geometry can move and when LiDAR range/normal terms become
optimization losses.  Phase A still renders and reports those terms, but their
weight is zero so they are monitors rather than forces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TOPOLOGY_MODES = ("strict_fixed", "opacity_prune_only", "adaptive_growth")


@dataclass(frozen=True)
class TopologyPolicyConfig:
    mode: str = "adaptive_growth"
    opacity_prune_step: int | None = None
    opacity_prune_threshold: float = 0.01

    def validate(self, *, max_steps: int) -> None:
        if self.mode not in TOPOLOGY_MODES:
            raise ValueError(f"topology mode must be one of {list(TOPOLOGY_MODES)}")
        if self.mode == "opacity_prune_only":
            if self.opacity_prune_step is None:
                raise ValueError("opacity_prune_only requires opacity_prune_step")
            if not 1 <= int(self.opacity_prune_step) < int(max_steps):
                raise ValueError("opacity_prune_step must be within [1, max_steps)")
            if not 0.0 < float(self.opacity_prune_threshold) < 1.0:
                raise ValueError("opacity_prune_threshold must be within (0, 1)")
        elif self.opacity_prune_step is not None:
            raise ValueError("opacity_prune_step is only valid for opacity_prune_only")

    @property
    def strategy_enabled(self) -> bool:
        return self.mode == "adaptive_growth"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixedTopologyScheduleConfig:
    enabled: bool = False
    phase_a_steps: int = 0
    phase_b_steps: int = 0
    phase_b_geometry_lr_scale: float = 1.0
    phase_c_geometry_lr_scale: float = 1.0
    phase_b_range_weight_scale: float = 1.0
    phase_c_range_weight_scale: float = 1.0
    phase_b_normal_weight_scale: float = 1.0
    phase_c_normal_weight_scale: float = 1.0
    audit_steps: tuple[int, ...] = ()

    def validate(self, *, max_steps: int, topology_mode: str) -> None:
        if not self.enabled:
            if self.phase_a_steps or self.phase_b_steps or self.audit_steps:
                raise ValueError("disabled fixed_topology_schedule must not define phases")
            return
        if topology_mode == "adaptive_growth":
            raise ValueError("fixed_topology_schedule requires a no-birth topology mode")
        if self.phase_a_steps < 0 or self.phase_b_steps < 0:
            raise ValueError("fixed-topology phase lengths must be non-negative")
        if self.phase_a_steps + self.phase_b_steps > max_steps:
            raise ValueError("fixed-topology phases exceed max_steps")
        for name, value in (
            ("phase_b_geometry_lr_scale", self.phase_b_geometry_lr_scale),
            ("phase_c_geometry_lr_scale", self.phase_c_geometry_lr_scale),
            ("phase_b_range_weight_scale", self.phase_b_range_weight_scale),
            ("phase_c_range_weight_scale", self.phase_c_range_weight_scale),
            ("phase_b_normal_weight_scale", self.phase_b_normal_weight_scale),
            ("phase_c_normal_weight_scale", self.phase_c_normal_weight_scale),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        previous = 0
        for completed_step in self.audit_steps:
            if not isinstance(completed_step, int) or isinstance(completed_step, bool):
                raise ValueError("audit_steps must contain integers")
            if completed_step <= previous or not 1 <= completed_step <= max_steps:
                raise ValueError("audit_steps must be unique, sorted, and within training")
            previous = completed_step

    def phase_for_step(self, step: int) -> dict[str, Any]:
        """Return the zero-based step's effective optimization controls."""

        if not self.enabled:
            return {
                "name": "UNSCHEDULED",
                "geometry_lr_scale": 1.0,
                "range_weight_scale": 1.0,
                "normal_weight_scale": 1.0,
                "emit_gap_analysis": False,
            }
        if step < self.phase_a_steps:
            return {
                "name": "A_APPEARANCE_WARMUP",
                "geometry_lr_scale": 0.0,
                "range_weight_scale": 0.0,
                "normal_weight_scale": 0.0,
                "emit_gap_analysis": False,
            }
        if step < self.phase_a_steps + self.phase_b_steps:
            return {
                "name": "B_LIDAR_GEOMETRY",
                "geometry_lr_scale": float(self.phase_b_geometry_lr_scale),
                "range_weight_scale": float(self.phase_b_range_weight_scale),
                "normal_weight_scale": float(self.phase_b_normal_weight_scale),
                "emit_gap_analysis": False,
            }
        return {
            "name": "C_GAP_DIAGNOSTIC",
            "geometry_lr_scale": float(self.phase_c_geometry_lr_scale),
            "range_weight_scale": float(self.phase_c_range_weight_scale),
            "normal_weight_scale": float(self.phase_c_normal_weight_scale),
            "emit_gap_analysis": True,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["audit_steps"] = list(self.audit_steps)
        return value


def topology_count_transition(
    *, mode: str, before_count: int, after_count: int, prune_due: bool
) -> None:
    """Fail closed when a topology arm performs an unauthorized transition."""

    if mode == "strict_fixed" and after_count != before_count:
        raise RuntimeError("strict_fixed topology changed Gaussian count")
    if mode == "opacity_prune_only":
        if after_count > before_count:
            raise RuntimeError("opacity_prune_only topology created Gaussians")
        if not prune_due and after_count != before_count:
            raise RuntimeError("opacity_prune_only changed count outside its prune step")
