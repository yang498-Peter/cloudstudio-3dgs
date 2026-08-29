"""Offset selection must survive a flat, noisy sweep.

Every offset renders the same rig frames, so comparing against zero is paired
per frame. Taking the argmax of the mean alone adopts whichever offset noise
happens to favour; on a flat sweep that is close to a coin toss, and a spurious
non-zero winner would demand re-deriving the whole pipeline from new
timestamps. Selection therefore requires a real paired margin.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_camera_time_sync.py"
_spec = importlib.util.spec_from_file_location("_time_sync_tool", _TOOL)
_module = importlib.util.module_from_spec(_spec)
sys.modules["_time_sync_tool"] = _module
_spec.loader.exec_module(_module)
_select_offset = _module._select_offset


def _sweep(per_offset):
    """Build a results block; per_offset maps offset -> list of frame PSNRs."""
    results = []
    for offset, values in per_offset.items():
        frames = [
            {"image_id": f"img_{i}", "psnr_db": v, "ssim": 0.5}
            for i, v in enumerate(values)
        ]
        results.append(
            {
                "offset_ms": float(offset),
                "frame_count": len(frames),
                "psnr_db_mean": sum(values) / len(values),
                "ssim_mean": 0.5,
                "frames": frames,
            }
        )
    baseline = next(r for r in results if r["offset_ms"] == 0.0)
    return results, baseline


def test_noisy_flat_sweep_keeps_zero():
    """A tiny mean gain swamped by per-frame spread must not be adopted."""
    zero = [14.0 + (i % 7) * 0.1 for i in range(40)]
    ten = [v + (0.01 if i % 2 else -0.008) for i, v in enumerate(zero)]
    results, baseline = _sweep({0: zero, 10: ten})
    best, selection = _select_offset(results, baseline, 0.05)
    assert best["offset_ms"] == 0.0
    assert "does not clear alpha" in selection["reason"]


def test_consistent_real_offset_is_adopted():
    """A uniform per-frame improvement is exactly what the audit should catch."""
    zero = [14.0 + (i % 7) * 0.1 for i in range(40)]
    ten = [v + 0.5 for v in zero]
    results, baseline = _sweep({0: zero, 10: ten})
    best, selection = _select_offset(results, baseline, 0.05)
    assert best["offset_ms"] == 10.0
    assert "improves PSNR" in selection["reason"]


def test_argmax_alone_would_have_picked_the_noise():
    """Pin the difference from the previous rule, so it cannot regress."""
    zero = [14.0 + (i % 7) * 0.1 for i in range(40)]
    ten = [v + (0.01 if i % 2 else -0.008) for i, v in enumerate(zero)]
    results, baseline = _sweep({0: zero, 10: ten})
    argmax = max(results, key=lambda r: r["psnr_db_mean"])
    assert argmax["offset_ms"] == 10.0
    best, _ = _select_offset(results, baseline, 0.05)
    assert best["offset_ms"] == 0.0


def test_worse_offset_is_never_adopted():
    zero = [14.0 + (i % 7) * 0.1 for i in range(40)]
    minus = [v - 0.5 for v in zero]
    results, baseline = _sweep({0: zero, -10: minus})
    best, _ = _select_offset(results, baseline, 0.05)
    assert best["offset_ms"] == 0.0


def test_selection_records_every_candidate():
    zero = [14.0 + (i % 5) * 0.1 for i in range(30)]
    results, baseline = _sweep(
        {0: zero, -10: [v - 0.4 for v in zero], 10: [v + 0.01 for v in zero]}
    )
    _, selection = _select_offset(results, baseline, 0.05)
    assert {c["offset_ms"] for c in selection["candidates"]} == {-10.0, 10.0}
    assert all("p_value" in c for c in selection["candidates"])
    assert selection["rule"] == "paired_t_test_against_zero"
