"""Resume a finished 7k arm from its 7,000 checkpoint to 7,199 (reset+299):
the 7,000 snapshot sits at reset+100, the worst phase of the opacity cycle,
so its battery under-reads. Usage: stage_resume_recovery.py <arm> [stop]
Creates tile0_<arm>_rec.json and run_<arm>_rec.cmd.
"""
import json
import sys
from pathlib import Path

RUN = Path(r"C:\Peter\3dgs-runs\house0305_sop")
SC = Path(__file__).resolve().parent
arm = sys.argv[1]
stop = int(sys.argv[2]) if len(sys.argv) > 2 else 7199
src = json.loads((RUN / f"tile0_{arm}.json").read_text(encoding="utf-8"))
new = f"{arm}_rec"
cfg = dict(src)
cfg["run_id"] = f"{src['run_id']}-rec"
cfg["output_dir"] = str(RUN / f"tile0_{new}")
cfg["resume_checkpoint"] = str(RUN / f"tile0_{arm}" / "checkpoints" / "step_00007000.pt")
cfg["controlled_stop_after_steps"] = stop
cfg["checkpoint_every"] = stop
cfg["checkpoint_keep_every"] = 0
(RUN / f"tile0_{new}.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
cmd = (SC / f"run_{arm}.cmd").read_text(encoding="ascii")
cmd = cmd.replace(f"tile0_{arm}", f"tile0_{new}")
(SC / f"run_{new}.cmd").write_text(cmd, encoding="ascii")
print("staged", new, "resume", cfg["resume_checkpoint"], "stop", stop)
