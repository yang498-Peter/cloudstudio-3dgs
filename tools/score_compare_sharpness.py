"""Sharpness on the on-trajectory strips (photo | ours | reference): Laplacian
variance of each panel, reported as ours/photo and reference/photo."""
import numpy as np, cv2, statistics, sys
from pathlib import Path
def lv(x): return float(cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
for d in sys.argv[1:]:
    rows=[]
    for f in sorted(Path(d).glob("compare_*.png")):
        im=cv2.imread(str(f)); w=(im.shape[1]-16)//3; p,o,r=im[:, :w], im[:, w+8:2*w+8], im[:, 2*w+16:3*w+16]
        rows.append((lv(o)/lv(p), lv(r)/lv(p)))
    print(Path(d).parent.name, "n", len(rows), "sharpness ours/photo median %.3f  ref/photo median %.3f" % (statistics.median([a for a,_ in rows]), statistics.median([b for _,b in rows])))
