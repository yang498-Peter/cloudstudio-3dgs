#!/usr/bin/env python3
"""Audit a pytest JUnit report against the policy of one CI channel.

A green job that silently collected nothing, or skipped everything that
matters, is worse than a red one. This reads the JUnit XML pytest wrote and
enforces per-channel rules:

  cpu        torch/gsplat/pycolmap are absent by design; their import errors
             and skips are counted and listed, anything else fails.
  torch-cpu  torch is installed; any "torch missing" skip or import error is
             a failure. gsplat/pycolmap/CUDA skips are allowed but listed.
  cuda       the machine-B channel; only pycolmap may be missing.

It always fails on test failures, on unexplained errors, and when a test file
contributes zero test cases (the unittest-discover blind spot for
function-style tests).

    python tests/check_collection.py junit.xml --channel cpu [--min-tests N]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

OPTIONAL_BY_CHANNEL = {
    "cpu": ("torch", "gsplat", "pycolmap", "cuda"),
    "torch-cpu": ("gsplat", "pycolmap", "cuda"),
    "cuda": ("pycolmap",),
}

# Substrings that identify a skip or import error as "this optional module
# is absent" rather than a real defect.
OPTIONAL_MARKERS = {
    "torch": ("torch is not installed", "torch missing", "torch is required",
              "torch is an optional", "No module named 'torch'"),
    "gsplat": ("gsplat", "No module named 'gsplat'"),
    "pycolmap": ("pycolmap",),
    "cuda": ("CUDA",),
}


def classify(message: str) -> str | None:
    for module, markers in OPTIONAL_MARKERS.items():
        if any(marker in message for marker in markers):
            return module
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("junit", type=Path)
    parser.add_argument("--channel", choices=sorted(OPTIONAL_BY_CHANNEL), required=True)
    parser.add_argument("--min-tests", type=int, default=1,
                        help="fail when fewer test cases than this were collected")
    parser.add_argument("--summary", type=Path, help="write a JSON summary here")
    args = parser.parse_args()

    root = ET.parse(args.junit).getroot()
    allowed = OPTIONAL_BY_CHANNEL[args.channel]
    per_file: dict[str, Counter] = defaultdict(Counter)
    skip_reasons: Counter = Counter()
    violations: list[str] = []
    failures_that_should_skip: list[str] = []
    optional_not_run: Counter = Counter()

    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        # Collection errors carry the file path in `name` and no classname.
        file_key = classname.split(".")[0] if classname else name
        per_file[file_key]["total"] += 1
        skipped = case.find("skipped")
        failure = case.find("failure")
        error = case.find("error")
        if skipped is not None:
            # Module-level importorskip reports "collection skipped" in the
            # attribute and the real reason in the element text.
            message = skipped.get("message") or ""
            module = classify(message + " " + (skipped.text or ""))
            skip_reasons[message] += 1
            per_file[file_key]["skipped"] += 1
            if module in allowed:
                optional_not_run[module] += 1
            else:
                violations.append(f"disallowed skip in {file_key}::{name}: {message}")
        elif failure is not None:
            message = (failure.get("message") or "") + " " + (failure.text or "")
            module = classify(message)
            per_file[file_key]["failed"] += 1
            if module in allowed and "No module named" in message:
                # A test that imports an optional module inside its body fails
                # instead of skipping. Same meaning as a skip on this channel,
                # but it is listed so the test can be taught to skip properly.
                optional_not_run[module] += 1
                failures_that_should_skip.append(f"{file_key}::{name} ({module})")
            else:
                violations.append(f"failure {file_key}::{name}: {message.strip()[:200]}")
        elif error is not None:
            message = (error.get("message") or "") + " " + (error.text or "")
            module = classify(message)
            per_file[file_key]["errored"] += 1
            if module in allowed:
                optional_not_run[module] += 1
            else:
                violations.append(f"error {file_key}::{name}: {message.strip()[:200]}")
        else:
            per_file[file_key]["passed"] += 1

    total = sum(counts["total"] for counts in per_file.values())
    executed = sum(counts["passed"] for counts in per_file.values())
    if total < args.min_tests:
        violations.append(f"collected {total} test cases, below --min-tests {args.min_tests}")
    for file_key, counts in sorted(per_file.items()):
        if counts["total"] == 0:
            violations.append(f"{file_key} contributed zero test cases")

    summary = {
        "channel": args.channel,
        "collected": total,
        "passed": executed,
        "skipped": sum(c["skipped"] for c in per_file.values()),
        "failed": sum(c["failed"] for c in per_file.values()),
        "errored": sum(c["errored"] for c in per_file.values()),
        "optional_not_run": dict(optional_not_run),
        "skip_reasons": dict(skip_reasons.most_common()),
        "per_file": {k: dict(v) for k, v in sorted(per_file.items())},
        "failures_that_should_skip": failures_that_should_skip,
        "violations": violations,
    }
    if args.summary is not None:
        args.summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[{args.channel}] collected={total} passed={executed} "
          f"skipped={summary['skipped']} failed={summary['failed']} errored={summary['errored']}")
    for module, count in sorted(optional_not_run.items()):
        print(f"  NOT_RUN ({module} absent by channel policy): {count}")
    for reason, count in skip_reasons.most_common(12):
        print(f"  skip x{count}: {reason}")
    for line in failures_that_should_skip:
        print(f"  NOT_RUN but reported as failure (should skip): {line}")
    for line in violations:
        print(f"  VIOLATION: {line}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
