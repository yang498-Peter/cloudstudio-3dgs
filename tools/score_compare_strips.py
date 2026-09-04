"""On the three-way strips (photo | ours | reference): brightness of each panel,
PSNR of ours and reference against the photo at 1/4 res, and the same after a
per-channel affine match (a*x+b fitted to the photo) to remove exposure."""
import math, numpy as np, cv2, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
def psnr(a, b): return 10*math.log10(1/max(float(np.mean((a-b)**2)),1e-10))
def affine_to(x, y):
    out = np.zeros_like(x)
    for c in range(3):
        a, b = np.polyfit(x[..., c].ravel(), y[..., c].ravel(), 1); out[..., c] = np.clip(a*x[..., c]+b, 0, 1)
    return out
rows = []
for f in sorted(d.glob("compare_*.png")):
    im = cv2.imread(str(f)).astype(np.float32)/255; w = (im.shape[1]-16)//3
    p, o, r = im[:, :w], im[:, w+8:2*w+8], im[:, 2*w+16:3*w+16]
    p4, o4, r4 = [cv2.resize(x, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA) for x in (p, o, r)]
    rows.append(dict(f=f.name[8:22], mean_photo=round(float(p4.mean()),3), mean_ours=round(float(o4.mean()),3), mean_ref=round(float(r4.mean()),3),
                     ours_vs_photo=round(psnr(o4,p4),2), ref_vs_photo=round(psnr(r4,p4),2), ours_aff=round(psnr(affine_to(o4,p4),p4),2), ref_aff=round(psnr(affine_to(r4,p4),p4),2)))
for x in rows: print(x)
print("median:", {k: round(statistics.median([x[k] for x in rows]),3) for k in rows[0] if k!="f"})
