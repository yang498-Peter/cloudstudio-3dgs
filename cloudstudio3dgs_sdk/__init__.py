"""CloudStudio 3DGS SDK: run a measured delivery recipe on a new dataset.

The research campaign that produced the recipe is not part of the contract.
A production team supplies a dataset and a profile name; the SDK plans,
preflights and runs the four stages (``prepare``, ``train``, ``deliver``,
``report``) on top of the existing ``tools/pipeline.py`` job-state machine.

Profiles are data. Adding a new recipe means adding a :class:`Profile`
object, never a new code path in :mod:`cloudstudio3dgs_sdk.project`.
"""

from __future__ import annotations

from cloudstudio3dgs_sdk.bundle import DerivedCaches, PreparedScene, load_dataset_bundle
from cloudstudio3dgs_sdk.plan import (
    DatasetSummary,
    Estimate,
    Plan,
    PlannedStep,
    TileSummary,
    build_plan,
)
from cloudstudio3dgs_sdk.profile import (
    PROFILES,
    PROFILE_B5FILL2,
    Profile,
    Provenance,
    get_profile,
)
from cloudstudio3dgs_sdk.project import Project, StageResult, StageState
from cloudstudio3dgs_sdk.requirements import Check, PreflightReport, Probes, preflight

__all__ = [
    "PROFILES",
    "PROFILE_B5FILL2",
    "Check",
    "DatasetSummary",
    "DerivedCaches",
    "Estimate",
    "Plan",
    "PlannedStep",
    "PreflightReport",
    "PreparedScene",
    "Probes",
    "Profile",
    "Project",
    "Provenance",
    "StageResult",
    "StageState",
    "TileSummary",
    "build_plan",
    "get_profile",
    "load_dataset_bundle",
    "preflight",
]

SDK_VERSION = "0.1.0"
