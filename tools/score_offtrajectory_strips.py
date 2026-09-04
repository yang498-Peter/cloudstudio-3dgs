"""Re-score the saved off-trajectory strips (ours | reference) with metrics that
tolerate the 3.5 cm alignment residual: PSNR at 1/4 and 1/8 resolution, and a
sharpness ratio (Laplacian variance ours / reference, alignment-free)."""
import sys, json, math, numpy as np, cv2, statistics
from pathlib import Path
RUN = Path("C:/Peter/3dgs-runs/house0305_sop")
arms = {"F6": RUN/"tile0_F6_v9_allphoto_vis6_20k/offtraj_final", "G1": RUN/"tile0_G1_flatten033_t0_20k/offtraj", "G2": RUN/"tile0_G2_align0_t0_20k/offtraj", "G3": RUN/"tile0_G3_grow5e5_cap12m_t0_20k/offtraj"}
def psnr(a, b): return 10*math.log10(1/max(float(np.mean((a.astype(np.float32)/255-b.astype(np.float32)/255)**2)),1e-10))
def lapvar(g): return float(cv2.Laplacian(g, cv2.CV_64F).var())
res = {}
for arm, d in arms.items():
    rows = []
    for f in sorted(d.glob("offtraj_*.png")):
        im = cv2.imread(str(f)); w = (im.shape[1]-8)//2; ours, ref = im[:, :w], im[:, w+8:w+8+w]
        o4, r4 = cv2.resize(ours, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA), cv2.resize(ref, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        o8, r8 = cv2.resize(ours, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA), cv2.resize(ref, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)
        go, gr = cv2.cvtColor(ours, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        rows.append(dict(file=f.name, psnr_full=round(psnr(ours, ref),2), psnr_q=round(psnr(o4, r4),2), psnr_e=round(psnr(o8, r8),2), sharp_ratio=round(lapvar(go)/max(lapvar(gr),1e-6),3)))
    res[arm] = rows
    print(arm, "n", len(rows), "median psnr full %.2f  1/4 %.2f  1/8 %.2f  sharpness ours/ref %.3f" % tuple(statistics.median([r[k] for r in rows]) for k in ("psnr_full","psnr_q","psnr_e","sharp_ratio")))
# paired wins vs F6 at 1/4 res
for arm in ("G1","G2","G3"):
    f6 = {r["file"]: r for r in res["F6"]}
    wins = sum(1 for r in res[arm] if r["file"] in f6 and r["psnr_q"] > f6[r["file"]]["psnr_q"]); n = sum(1 for r in res[arm] if r["file"] in f6)
    print(arm, "wins vs F6 at 1/4 res", wins, "/", n)
json.dump(res, open(RUN/"offtraj_robust_scores.json", "w"), indent=1)
