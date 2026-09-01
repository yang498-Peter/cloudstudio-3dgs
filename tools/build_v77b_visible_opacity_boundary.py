"""Build V77b: apply opacity sparsity only to current-view visible Gaussians."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloudstudio_3dgs.data.manifest import canonical_json_bytes
from cloudstudio_3dgs.pipeline.mipmap_gate import sign_gate, verify_gate
from cloudstudio_3dgs.training.trainer import TrainerConfig

BASE = ROOT / "outputs" / "snow-20260224-full-20260825"
SOURCE = BASE / "v77a_detail_alpha" / "tile0_boundary1202"
TARGET = BASE / "v77b_visible_opacity" / "tile0_boundary1202"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("config_manifest_sha256", None)
    result["config_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    config = _read(SOURCE / "trainer.config.json")
    gate = _read(SOURCE / "training_gate.json")
    verify_gate(gate)
    resume_checkpoint = Path(config["resume_checkpoint"])
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)

    config.update(
        {
            "run_id": "snow-tile0-v77b-visible-opacity-boundary1202",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
        }
    )
    regularization = dict(config["geometry_regularization"])
    regularization["opacity_sparsity_scope"] = "visible_current_view"
    config["geometry_regularization"] = regularization
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "visible_opacity_step1200_boundary_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate.setdefault("evidence", {})["v77b_visible_opacity_boundary"] = {
        "source_checkpoint_sha256": _sha256_file(resume_checkpoint),
        "source_completed_steps": 1102,
        "post_lifecycle_controlled_stop_completed_steps": 1202,
        "change": {"opacity_sparsity_scope": "visible_current_view"},
        "controlled_variable": (
            "repeat V77a from the same V76a checkpoint; only stop applying "
            "opacity sparsity gradients to Gaussians absent from the current view"
        ),
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": "v77b_visible_opacity_detail_boundary",
            "run_id": config["run_id"],
        }
    )
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V77b: same step-1102 source, visible-only opacity sparsity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
