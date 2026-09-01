"""Build V83a: sky-clean gradient accumulation, then one mesh-guided detail cycle."""

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
SOURCE = BASE / "v81b_deferred_reset_settle" / "tile0_pre3100"
TARGET = BASE / "v83a_sky_clean_mesh_guided_detail" / "tile0_pre3300"
SKY_ROOT = BASE / "independent_sky_background_v23l" / "train"
SKY_MANIFEST = SKY_ROOT / "sky_evidence_manifest.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    checkpoint = SOURCE / "training_ewa" / "checkpoints" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 3099:
        raise ValueError("V83a requires the exact V81b step-3099 checkpoint")
    sky_manifest = _read(SKY_MANIFEST)
    sky_sha = str(sky_manifest.get("sky_evidence_manifest_sha256", ""))
    if len(sky_sha) != 64:
        raise ValueError("V83a surface sky exclusion manifest is unsigned")

    config.update(
        {
            "run_id": "snow-tile0-v83a-sky-clean-mesh-guided-detail-pre3300",
            "output_dir": str(TARGET / "training_ewa"),
            "mipmap_pipeline_gate": str(TARGET / "training_gate.json"),
            "resume_checkpoint": str(checkpoint),
            "controlled_stop_after_steps": 3299,
            "checkpoint_every": 3299,
            # Do not duplicate the controlled-stop checkpoint; latest.pt is
            # already the signed continuation artifact.
            "checkpoint_keep_every": 12480,
            "surface_sky_exclusion_manifest": str(SKY_MANIFEST),
            "surface_sky_exclusion_root": str(SKY_ROOT),
            "surface_sky_exclusion_dilation_px": 8,
        }
    )
    strategy = dict(config["default_strategy"])
    strategy["discard_accumulated_gradient_steps"] = [3100]
    config["default_strategy"] = strategy
    config = _sign_config(config)

    gate.pop("gate_manifest_sha256", None)
    gate.update(
        {
            "status": "ADAPTIVE_GROWTH_REVIEW_READY",
            "training_allowed": True,
            "next_required_stage": "mesh_guided_detail_joint_evaluation",
        }
    )
    gate["bindings"] = dict(gate["bindings"])
    gate["bindings"]["adaptive_growth_config_manifest_sha256"] = config[
        "config_manifest_sha256"
    ]
    gate["bindings"]["surface_sky_exclusion_manifest_sha256"] = sky_sha
    gate.setdefault("evidence", {})["v83a_mesh_guided_detail_cycle"] = {
        "source_checkpoint_sha256": _sha256_file(checkpoint),
        "source_completed_steps": 3099,
        "discarded_pre_mask_gradient_step": 3100,
        "surface_sky_exclusion_manifest_sha256": sky_sha,
        "surface_sky_exclusion_dilation_px": 8,
        "detail_lifecycle_step": 3200,
        "controlled_stop_completed_steps": 3299,
        "excluded_next_opacity_cull_step": 3300,
        "purpose": (
            "remove high-confidence sky from surface RGB gradients, discard stale "
            "pre-mask statistics, and measure one mesh-guided detail split cycle"
        ),
    }
    gate["adaptive_growth"] = dict(gate["adaptive_growth"])
    gate["adaptive_growth"].update(
        {
            "profile": "v83a_sky_clean_mesh_guided_detail_pre3300",
            "run_id": config["run_id"],
            "controlled_stop_after_steps": 3299,
            "resume_checkpoint_sha256": _sha256_file(checkpoint),
            "resume_source_trainer_config_sha256": str(
                payload.get("identity", {}).get("trainer_config_sha256", "")
            ),
        }
    )
    gate = sign_gate(gate)

    _write(TARGET / "trainer.config.json", config)
    _write(TARGET / "training_gate.json", gate)
    TrainerConfig.from_dict(config).validate()
    verify_gate(gate)
    print(TARGET / "trainer.config.json")
    print(TARGET / "training_gate.json")
    print("V83a: discard stale step-3100 gradients, split at 3200, stop before cull 3300")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
