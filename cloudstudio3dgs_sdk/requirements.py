"""Preflight: everything that must be true before a run is allowed to start.

The failure this prevents is the expensive one - discovering at hour six that
the card is too small, the disk is too small, the sky-mask weights are not on
this host, or that this checkout cannot actually produce the artefact the
profile describes.

Every probe is injectable (:class:`Probes`), so the whole preflight runs on a
CPU host with no GPU, no network and no weights: the tests exercise the
failure paths, not the happy path of one particular machine.

The report is fail-closed. :meth:`PreflightReport.raise_for_status` raises on
any required FAIL; :class:`~cloudstudio3dgs_sdk.project.Project` calls it
before every stage that costs GPU time.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from cloudstudio3dgs_sdk.plan import GIB, Plan
from cloudstudio3dgs_sdk.profile import Profile

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


class PreflightFailed(RuntimeError):
    """A required check failed; nothing was started."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    required: bool = True
    remedy: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "required": self.required,
            "remedy": self.remedy,
        }


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[Check, ...]
    profile_sha256: str
    plan_sha256: str

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.status == FAIL and check.required)

    @property
    def warnings(self) -> tuple[Check, ...]:
        return tuple(
            check for check in self.checks if check.status == WARN or (check.status == FAIL and not check.required)
        )

    def get(self, name: str) -> Check:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(f"no check named {name!r}")

    def raise_for_status(self) -> None:
        if self.ok:
            return
        lines = [f"{check.name}: {check.detail}" + (f" -> {check.remedy}" if check.remedy else "")
                 for check in self.failures]
        raise PreflightFailed("preflight refused to start:\n  " + "\n  ".join(lines))

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ok": self.ok,
            "profile_sha256": self.profile_sha256,
            "plan_sha256": self.plan_sha256,
            "checks": [check.as_json() for check in self.checks],
        }

    def render(self) -> str:
        lines = [f"preflight {'PASS' if self.ok else 'FAIL'} (profile {self.profile_sha256[:12]})"]
        for check in self.checks:
            marker = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "skip"}[check.status]
            lines.append(f"  [{marker}] {check.name}: {check.detail}")
            if check.remedy and check.status in (FAIL, WARN):
                lines.append(f"           -> {check.remedy}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    available: bool
    name: str = ""
    total_vram_gib: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class Probes:
    """Everything preflight learns about the host. All of it injectable."""

    python_version: str
    torch_version: str | None
    gpu: GpuInfo
    free_disk_bytes: Callable[[Path], int]
    gsplat_extension_sha256: str | None = None
    gsplat_import_error: str = ""
    external_asset_locator: Callable[[Mapping[str, Any]], tuple[bool, str]] | None = None

    @staticmethod
    def detect() -> "Probes":
        torch_version: str | None = None
        gpu = GpuInfo(False, detail="torch is not importable")
        extension: str | None = None
        import_error = ""
        try:  # pragma: no cover - host dependent
            import torch  # type: ignore

            torch_version = str(torch.__version__)
            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(index)
                gpu = GpuInfo(True, properties.name, properties.total_memory / GIB)
            else:
                gpu = GpuInfo(False, detail="torch.cuda.is_available() is False")
        except Exception as error:  # pragma: no cover - host dependent
            import_error = repr(error)
        try:  # pragma: no cover - host dependent
            import hashlib

            from gsplat.cuda import _backend  # type: ignore

            extension = hashlib.sha256(Path(_backend._C.__file__).read_bytes()).hexdigest()
        except Exception as error:  # pragma: no cover - host dependent
            import_error = import_error or repr(error)
        return Probes(
            python_version=".".join(str(part) for part in sys.version_info[:2]),
            torch_version=torch_version,
            gpu=gpu,
            free_disk_bytes=_free_disk_bytes,
            gsplat_extension_sha256=extension,
            gsplat_import_error=import_error,
        )


def _free_disk_bytes(path: Path) -> int:  # pragma: no cover - host dependent
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def default_asset_locator(asset: Mapping[str, Any]) -> tuple[bool, str]:  # pragma: no cover - host dependent
    """Look for a HuggingFace-cached model directory for ``asset['model_id']``."""
    model_id = asset.get("model_id")
    if not model_id:
        return True, "no local artefact required"
    roots = [
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ.get("TRANSFORMERS_CACHE", "")) if os.environ.get("TRANSFORMERS_CACHE") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    folder = "models--" + str(model_id).replace("/", "--")
    for root in roots:
        if root is None:
            continue
        candidate = root / folder
        if candidate.is_dir():
            return True, str(candidate)
    return False, f"{folder} not found under {', '.join(str(r) for r in roots if r)}"


#: Steps whose disk grows with the number of views: the ingest caches, the sky masks, the
#: ownership masks and every rendered background. Tile runs, the coarse prior, the merge and
#: the export are sized by the caps and the tile count instead.
VIEW_SCALED_STEP_PREFIXES = ("ingest_dataset", "sky_masks", "ownership_", "backdrop_", "global_view_backgrounds")


def _subset_hint(plan: Plan, profile: Profile, *, free: int, needed: int, factor: float) -> str:
    """How many images of this capture would fit, for a capture nobody has prepared.

    A subset of ``s`` of the capture's ``I`` images is costed as: the view-scaled steps
    shrink by ``s / I``; the per-tile rest (tile runs, merge, export) shrinks with the tile
    count, which follows the point count down to the profile's reference count and no
    further. Both are linear guesses that hold the per-tile part at its full-scene size per
    tile, so the count errs small.
    """
    from cloudstudio3dgs_sdk.plan import step_is_skippable

    faces = plan.dataset.train_view_count
    images = faces / 4
    scaling = sum(
        step.estimate.disk_bytes
        for step in plan.steps
        if step.name.startswith(VIEW_SCALED_STEP_PREFIXES) and not step_is_skippable(step)
    ) * factor
    if scaling <= 0 or faces <= 0:
        return ""
    fixed = needed - scaling
    floor_share = min(1.0, int(profile.tiling["reference_tile_count"]) / max(plan.dataset.tile_count, 1))
    floor = fixed * floor_share
    if free <= floor:
        return (
            f"; even a small subset does not fit: at the reference {profile.tiling['reference_tile_count']} "
            f"tiles the tile runs and the delivery alone need {floor/GIB:.1f} GB"
        )
    # share s/I of the capture: cost = scaling*share + fixed*max(share, floor_share)
    share = min(1.0, free / needed)
    if share < floor_share:
        share = min(floor_share, (free - floor) / scaling)
    return (
        f"; or prepare a subset: about {int(share * images)} of its {int(images)} images fit "
        f"({needed/GIB/images*1024:.0f} MB per image at this scale, x {factor})"
    )


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def preflight(
    plan: Plan,
    profile: Profile,
    *,
    repo_root: Path,
    probes: Probes | None = None,
    require_gpu: bool = True,
) -> PreflightReport:
    probes = probes or Probes.detect()
    runtime = profile.runtime
    checks: list[Check] = []

    # -- python ---------------------------------------------------------
    wanted_python = str(runtime["python"])
    checks.append(
        Check(
            "python_version",
            PASS if probes.python_version == wanted_python else FAIL,
            f"running {probes.python_version}, profile pins {wanted_python}",
            remedy=f"use the {wanted_python} training venv",
        )
    )

    # -- torch ----------------------------------------------------------
    wanted_torch = str(runtime["torch"])
    if probes.torch_version is None:
        checks.append(
            Check(
                "torch_version",
                FAIL if require_gpu else SKIP,
                f"torch is not importable ({probes.gsplat_import_error or 'no detail'})",
                required=require_gpu,
                remedy=f"install {wanted_torch} into the training venv",
            )
        )
    else:
        matches = probes.torch_version == wanted_torch
        checks.append(
            Check(
                "torch_version",
                PASS if matches else WARN,
                f"torch {probes.torch_version}, profile pins {wanted_torch}",
                required=False,
                remedy="" if matches else "a different torch changes the compiled extension identity",
            )
        )

    # -- gsplat lock ----------------------------------------------------
    lock_path = Path(repo_root) / str(runtime["gsplat_lock_relpath"])
    if not lock_path.is_file():
        checks.append(
            Check("gsplat_lock", FAIL, f"missing {lock_path}", remedy="check out the pinned upstream lock")
        )
    else:
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except ValueError as error:
            lock = {}
            checks.append(Check("gsplat_lock", FAIL, f"{lock_path} is not JSON: {error}"))
        if lock:
            same = str(lock.get("version")) == str(runtime["gsplat_version"])
            checks.append(
                Check(
                    "gsplat_lock",
                    PASS if same else FAIL,
                    f"lock {lock.get('version')} commit {str(lock.get('commit'))[:12]}, "
                    f"profile pins {runtime['gsplat_version']}",
                    remedy="" if same else "rebuild gsplat from the locked commit and patch",
                )
            )
            checks.append(
                Check(
                    "gsplat_extension",
                    PASS if probes.gsplat_extension_sha256 else (FAIL if require_gpu else SKIP),
                    (
                        f"compiled extension sha256 {probes.gsplat_extension_sha256[:12]}"
                        if probes.gsplat_extension_sha256
                        else f"no compiled extension ({probes.gsplat_import_error or 'not importable'})"
                    ),
                    required=require_gpu,
                    remedy=(
                        ""
                        if probes.gsplat_extension_sha256
                        else "build the extension; note checkout_diff_sha256 depends on git's "
                             "abbreviation length, so a mismatch on a new host is worth checking "
                             "with core.abbrev=7 before suspecting the patch"
                    ),
                )
            )

    # -- GPU and VRAM ---------------------------------------------------
    largest_cap = max(plan.tile_caps.values()) if plan.tile_caps else 0
    if not probes.gpu.available:
        checks.append(
            Check(
                "gpu",
                FAIL if require_gpu else SKIP,
                probes.gpu.detail or "no CUDA device",
                required=require_gpu,
                remedy="training and the battery need a CUDA device; prepare() alone does not",
            )
        )
    else:
        min_vram = float(runtime["min_vram_gib"])
        enough = probes.gpu.total_vram_gib + 1e-6 >= min_vram
        checks.append(
            Check(
                "gpu",
                PASS if enough else FAIL,
                f"{probes.gpu.name}, {probes.gpu.total_vram_gib:.1f} GiB (profile needs {min_vram:.1f})",
                remedy="" if enough else "use a larger card or a profile with smaller caps",
            )
        )
        ceiling = int(
            float(runtime["max_gaussians_per_gib_vram"])
            * probes.gpu.total_vram_gib
            * float(runtime["vram_safety_factor"])
        )
        fits = largest_cap <= ceiling
        checks.append(
            Check(
                "vram_headroom",
                PASS if fits else FAIL,
                f"largest planned cap {largest_cap/1e6:.2f}M vs {ceiling/1e6:.2f}M this card can hold",
                remedy=(
                    ""
                    if fits
                    else "re-plan with vram_gib set so the cap rule clamps; an over-cap tile dies "
                         "mid-run in the duplicate phase, not at startup"
                ),
            )
        )

    # -- every tile can hold its initialisation ---------------------------
    # The trainer refuses a tile whose initialisation reaches its cap, at startup - which on a
    # fresh capture is after the whole prepare stage. A cap under the profile's ratio is legal
    # but throttles growth the recipe was measured with.
    ratio = float(profile.tile_rules["cap_ratio_of_initialisation"])
    half_step = int(profile.tile_rules["cap_round_to"]) / 2  # rounding alone is not throttling
    over = [t for t in plan.dataset.tiles if t.init_point_count >= plan.tile_caps.get(t.tile_id, 0)]
    tight = [
        t for t in plan.dataset.tiles
        if t not in over and plan.tile_caps.get(t.tile_id, 0) + half_step < t.init_point_count * ratio
    ]
    rows = ", ".join(
        f"{t.name} init {t.init_point_count/1e6:.2f}M cap {plan.tile_caps.get(t.tile_id, 0)/1e6:.2f}M"
        for t in (over or tight)
    )
    if over:
        from cloudstudio3dgs_sdk.plan import tile_count_for

        card = probes.gpu.total_vram_gib if probes.gpu.available else None
        wanted = tile_count_for(profile, plan.dataset.lidar_point_count, vram_gib=card)
        checks.append(
            Check(
                "tile_capacity",
                FAIL,
                f"initialisation at or over the cap, the trainer refuses these: {rows}",
                remedy=(
                    f"cut the cloud into more tiles ({wanted} for {plan.dataset.lidar_point_count/1e6:.1f}M "
                    f"points under this profile) - a fresh prepare does this itself"
                    if wanted > plan.dataset.tile_count
                    else "cut the cloud into more tiles"
                ),
            )
        )
    elif tight:
        checks.append(
            Check(
                "tile_capacity",
                WARN,
                f"cap below {ratio}x the initialisation (growth throttled): {rows}",
                required=False,
            )
        )
    else:
        checks.append(Check("tile_capacity", PASS, f"every tile's cap is at least {ratio}x its initialisation"))

    # -- disk -----------------------------------------------------------
    # Only the steps a run would still execute count: a resumed run has already paid
    # for the outputs that exist, and asking for the whole plan's disk again refused
    # the first SDK delivery at the deliver stage after seven hours of training.
    factor = float(runtime["disk_safety_factor"])
    needed = int(plan.pending().disk_bytes * factor)
    whole = int(plan.total().disk_bytes * factor)
    try:
        free = int(probes.free_disk_bytes(Path(plan.work_root)))
    except OSError as error:
        checks.append(Check("disk_headroom", FAIL, f"cannot stat {plan.work_root}: {error}"))
    else:
        enough = free >= needed
        scope = (
            f"the steps still to run need {needed/GIB:.1f} GB"
            + (f" (whole plan {whole/GIB:.1f} GB)" if whole != needed else "")
        )
        remedy = "" if enough else "free space or point --work at a larger volume"
        if not enough and plan.dataset.estimated:
            remedy += _subset_hint(plan, profile, free=free, needed=needed, factor=factor)
        checks.append(
            Check(
                "disk_headroom",
                PASS if enough else FAIL,
                f"{free/GIB:.1f} GB free at {plan.work_root}, {scope} (estimate x {factor})",
                remedy=remedy,
            )
        )

    # -- external assets -------------------------------------------------
    locate = probes.external_asset_locator or default_asset_locator
    for asset in profile.external_assets:
        if asset.get("ships_in_delivery"):
            checks.append(
                Check(
                    f"asset_{asset['id']}",
                    FAIL,
                    "profile declares an external asset as shipping in the delivery",
                    remedy="no third-party weights may be redistributed",
                )
            )
            continue
        if not asset.get("model_id"):
            continue
        found, detail = locate(asset)
        checks.append(
            Check(
                f"asset_{asset['id']}",
                PASS if found else FAIL,
                f"{asset['model_id']} ({asset['license']}): {detail}. {asset['license_note']}",
                remedy="" if found else f"fetch {asset['model_id']} revision {asset.get('revision', '?')} onto this host",
            )
        )

    # -- this checkout can produce what the profile describes -------------
    blocking = plan.blocking_steps()
    checks.append(
        Check(
            "checkout_supports_profile",
            PASS if not blocking else FAIL,
            "every planned step has a tool in this checkout"
            if not blocking
            else "; ".join(f"{step.name}: {step.blocking}" for step in blocking),
            remedy="" if not blocking else "port the missing tool support before running this profile",
        )
    )

    # -- tools exist ------------------------------------------------------
    missing_tools = sorted(
        {
            Path(step.command[1]).name
            for step in plan.steps
            if len(step.command) > 1 and not Path(step.command[1]).is_file()
        }
    )
    checks.append(
        Check(
            "tools_present",
            PASS if not missing_tools else FAIL,
            "all planned tools exist" if not missing_tools else f"missing: {', '.join(missing_tools)}",
            remedy="" if not missing_tools else "run from a complete checkout",
        )
    )

    # -- knobs nobody measured -------------------------------------------
    unmeasured = profile.unmeasured_knobs()
    checks.append(
        Check(
            "profile_confidence",
            PASS if not unmeasured else WARN,
            "every knob has a measured justification"
            if not unmeasured
            else f"{len(unmeasured)} knob group(s) are inferred or unmeasured: {', '.join(unmeasured)}",
            required=False,
            remedy="" if unmeasured else "",
        )
    )

    # -- where the dataset numbers came from ------------------------------
    # A warning, not a failure: preflight answers "can this host run it", and
    # an estimated summary is a perfectly good question to ask that about. The
    # refusal lives on the run path, in Project._run_stage.
    checks.append(
        Check(
            "dataset_summary_source",
            WARN if plan.dataset.estimated else PASS,
            "ESTIMATED from the capture: tile boxes, per-tile view counts and initialisation "
            "counts are derivations, so every cost built on them is one too"
            if plan.dataset.estimated
            else "measured by prepare()",
            required=False,
            remedy="run prepare() before reading these numbers as a commitment"
            if plan.dataset.estimated
            else "",
        )
    )

    # -- plan warnings ----------------------------------------------------
    for index, warning in enumerate(plan.warnings):
        checks.append(Check(f"plan_warning_{index}", WARN, warning, required=False))

    return PreflightReport(tuple(checks), profile.profile_sha256, plan.plan_sha256)
