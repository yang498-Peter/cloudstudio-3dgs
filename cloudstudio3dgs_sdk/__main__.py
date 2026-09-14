"""Command line entry point.

    python -m cloudstudio3dgs_sdk run --dataset <path> --work <path> \
        --profile b5fill2 [--dry-run] [--stages prepare,train,deliver,report]

``--dry-run`` prints the full execution plan - every step, its argv, its time
and disk estimate and the confidence behind each estimate - and runs nothing.
Two helper commands exist because both answer questions people ask before
they commit a machine for a day: ``preflight`` prints the host report, and
``profile`` prints one recipe with its provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from cloudstudio3dgs_sdk.plan import STAGES, DatasetSummary
from cloudstudio3dgs_sdk.profile import PROFILES, get_profile
from cloudstudio3dgs_sdk.project import Project, StageRefused
from cloudstudio3dgs_sdk.requirements import PreflightFailed

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2


def parse_stages(value: str) -> tuple[str, ...]:
    """``prepare,train`` -> ('prepare', 'train'), in canonical stage order."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--stages needs at least one stage")
    unknown = [name for name in names if name not in STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {', '.join(unknown)}; known: {', '.join(STAGES)}"
        )
    seen: list[str] = []
    for stage in STAGES:
        if stage in names and stage not in seen:
            seen.append(stage)
    return tuple(seen)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cloudstudio3dgs_sdk",
        description=__doc__.split("\n")[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan, preflight and execute the stages for one dataset")
    run.add_argument("--dataset", required=True, type=Path, help="customer dataset root")
    run.add_argument("--work", required=True, type=Path, help="work root; all SDK output lands here")
    run.add_argument(
        "--profile",
        default="b5fill2",
        choices=sorted(PROFILES),
        help="frozen recipe to run (default: b5fill2)",
    )
    run.add_argument("--dry-run", action="store_true", help="print the execution plan and exit")
    run.add_argument(
        "--stages",
        type=parse_stages,
        default=STAGES,
        metavar="prepare,train,deliver,report",
        help="subset of stages to run (default: all four, in order)",
    )
    run.add_argument("--force", action="store_true", help="re-run stages whose state says COMPLETE")
    run.add_argument("--scene-tag", default=None, help="override the scene tag (default: from prepare)")
    run.add_argument("--delivery-tag", default=None, help="delivery tag (default: the profile name)")
    run.add_argument("--python", type=Path, default=None, help="training interpreter (default: this one)")
    run.add_argument("--repo-root", type=Path, default=None, help="checkout holding tools/ (default: this one)")
    run.add_argument(
        "--vram-gib",
        type=float,
        default=None,
        help="clamp the per-tile caps to what a card this size can hold, at plan time",
    )
    run.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="dataset summary JSON to plan against before prepare() has run",
    )
    run.add_argument(
        "--prior-checkpoint",
        action="append",
        default=[],
        metavar="TILE=PATH",
        help=(
            "a previous generation's checkpoint for one tile; supplying one per tile removes the "
            "seed generation, because the stand-in backdrops can render these instead"
        ),
    )
    run.add_argument("--plan-json", type=Path, default=None, help="also write the plan as JSON here")

    pre = sub.add_parser("preflight", help="host report only; runs nothing")
    pre.add_argument("--dataset", required=True, type=Path)
    pre.add_argument("--work", required=True, type=Path)
    pre.add_argument("--profile", default="b5fill2", choices=sorted(PROFILES))
    pre.add_argument("--summary", type=Path, default=None)
    pre.add_argument("--vram-gib", type=float, default=None)
    pre.add_argument("--repo-root", type=Path, default=None)
    pre.add_argument("--python", type=Path, default=None)
    pre.add_argument("--no-gpu", action="store_true", help="do not require a CUDA device (prepare-only host)")

    show = sub.add_parser("profile", help="print a profile, its provenance and its open questions")
    show.add_argument("name", nargs="?", default=None, choices=[*sorted(PROFILES), None])
    show.add_argument("--json", action="store_true", help="print the profile as canonical JSON")
    return parser


def parse_prior_checkpoints(values: Sequence[str]) -> dict[int, str]:
    """``3=D:/runs/tile3/latest.pt`` -> {3: 'D:/runs/tile3/latest.pt'}."""
    out: dict[int, str] = {}
    for value in values:
        tile, sep, path = value.partition("=")
        if not sep or not path.strip():
            raise ValueError(f"--prior-checkpoint expects TILE=PATH, got {value!r}")
        try:
            tile_id = int(tile)
        except ValueError:
            raise ValueError(f"--prior-checkpoint tile id must be an integer, got {tile!r}") from None
        if tile_id in out:
            raise ValueError(f"--prior-checkpoint given twice for tile {tile_id}")
        out[tile_id] = path
    return out


def _project(args: argparse.Namespace, stream) -> Project:
    summary = None
    if getattr(args, "summary", None):
        summary = DatasetSummary.from_json(json.loads(Path(args.summary).read_text(encoding="utf-8")))
    return Project(
        args.dataset,
        args.work,
        get_profile(args.profile),
        repo_root=getattr(args, "repo_root", None),
        python=getattr(args, "python", None),
        scene_tag=getattr(args, "scene_tag", None),
        delivery_tag=getattr(args, "delivery_tag", None),
        dataset=summary,
        vram_gib=getattr(args, "vram_gib", None),
        prior_tile_checkpoints=parse_prior_checkpoints(getattr(args, "prior_checkpoint", [])),
        stream=stream,
    )


def main(argv: Sequence[str] | None = None, *, stream=None) -> int:
    stream = stream or sys.stdout
    # A Windows console defaults to cp1252 and the profile's provenance text carries
    # characters it cannot encode, which crashed `profile` mid-listing. Print UTF-8
    # regardless of what the console claims, and never let an encoding fault lose output.
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None and getattr(stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "profile":
        names = [args.name] if args.name else sorted(PROFILES)
        for name in names:
            profile = get_profile(name)
            if args.json:
                print(json.dumps(profile.as_dict(), indent=1, ensure_ascii=False, sort_keys=True), file=stream)
                continue
            print(f"{profile.name}@{profile.version}  sha256 {profile.profile_sha256}", file=stream)
            print(f"  {profile.summary}", file=stream)
            print("  provenance:", file=stream)
            for key in sorted(profile.provenance):
                why = profile.provenance[key]
                print(f"    [{why.confidence:<12}] {key}: {why.claim}", file=stream)
                print(f"                     source: {why.source}", file=stream)
            print("  open questions:", file=stream)
            for item in profile.open_questions:
                print(f"    - {item['id']}: {item['what']}", file=stream)
                print(f"      expressed as: {item['expressed_as']} ({item['status']})", file=stream)
        return EXIT_OK

    try:
        project = _project(args, stream)
        if args.command == "preflight":
            report = project.preflight(require_gpu=not args.no_gpu)
            print(report.render(), file=stream)
            return EXIT_OK if report.ok else EXIT_REFUSED
        if args.command == "run":
            if args.dry_run:
                plan = project.plan(stages=args.stages)
                print(plan.render(), file=stream)
                if args.plan_json:
                    Path(args.plan_json).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.plan_json).write_text(
                        json.dumps(plan.as_json(), indent=1, ensure_ascii=False), encoding="utf-8"
                    )
                    print(f"plan written to {args.plan_json}", file=stream)
                return EXIT_OK
            results = project.run_all(stages=args.stages, force=args.force)
            for result in results:
                print(f"{result.stage}: {result.action} {result.reason}".rstrip(), file=stream)
            return EXIT_OK if all(result.ok for result in results) else EXIT_REFUSED
    except (StageRefused, PreflightFailed) as error:
        print(f"cloudstudio3dgs_sdk: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except (OSError, ValueError, KeyError) as error:
        print(f"cloudstudio3dgs_sdk: {error}", file=sys.stderr)
        return EXIT_USAGE
    parser.error(f"unknown command {args.command}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
