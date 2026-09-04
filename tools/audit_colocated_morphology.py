"""Co-located morphology: voxelize ours and the aligned reference at 0.5 m and
compare, voxel by voxel where both are present, count / size / opacity /
orientation. Orientation = angle between each gaussian's shortest axis and the
local LiDAR surface normal (0 deg = a disk lying on the surface)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "C:/Peter/cloudstudio-3dgs-work/tools"); sys.path.insert(0, "C:/Peter/cloudstudio-3dgs-work")
from inspect_gaussian_ply import _read_ply
from pathlib import Path
from scipy.spatial import cKDTree
RUN = "C:/Peter/3dgs-runs/house0305_sop"
V = 0.5
def quat_to_R(q):  # w x y z
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    R = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
    return R
def load_ours():
    p = torch.load(f"{RUN}/delivery_f6/merged.pt", map_location="cpu", weights_only=False)["params"]
    return p["means"].numpy(), torch.sigmoid(p["opacities"].float()).numpy(), np.exp(p["scales"].numpy()), p["quats"].numpy()
def load_ref():
    v, _ = _read_ply(Path("C:/baidunetdiskdownload/house/USAgs.ply"))
    A = np.array(json.load(open("C:/Peter/3dgs-runs/probes/usa_gs_alignment.json"))["transform"])
    xyz = np.stack([v["x"], v["y"], v["z"]], 1) @ A[:3, :3].T + A[:3, 3]
    opa = 1/(1+np.exp(-np.asarray(v["opacity"])))
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    q = A[:3, :3] @ np.eye(3)  # rotation of frame is ~identity (0.05 deg); ignore for orientation stats
    return xyz.astype(np.float32), opa.astype(np.float32), sc.astype(np.float32), np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
# LiDAR surface + normals from the tile geometry (k30 PCA normals) if available, else PCA on the fly
lid = np.concatenate([np.stack([v["x"], v["y"], v["z"]], 1)[::4] for v in (_read_ply(Path(f"{RUN}/tile_inputs_v9/Tile_{t}/initialization_full_lidar.ply"))[0] for t in range(4))]).astype(np.float32)
tree = cKDTree(lid)
sub = lid[::20]; _, nb = tree.query(sub, k=16, workers=8)
P = lid[nb] - sub[:, None]; C = np.einsum("nki,nkj->nij", P, P); w, vecs = np.linalg.eigh(C); normals = vecs[:, :, 0]
ntree = cKDTree(sub)
def stats(name, xyz, opa, sc, q):
    live = opa > 0.1; xyz, opa, sc, q = xyz[live], opa[live], sc[live], q[live]
    d, j = ntree.query(xyz, workers=8)
    R = quat_to_R(q); short_axis = np.take_along_axis(R, np.argmin(sc, 1)[:, None, None].repeat(3, 1), 2)[:, :, 0]
    cosang = np.abs(np.einsum("ni,ni->n", short_axis, normals[j])); ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    key = np.floor(xyz / V).astype(np.int64); kk = key[:, 0] * 1000003 + key[:, 1] * 1009 + key[:, 2]
    order = np.argsort(kk); kk, xyz, opa, sc, ang, d = kk[order], xyz[order], opa[order], sc[order], ang[order], d[order]
    uniq, start, cnt = np.unique(kk, return_index=True, return_counts=True)
    out = {}
    for u, s, c in zip(uniq, start, cnt):
        sl = slice(s, s + c)
        out[int(u)] = dict(n=int(c), long=float(np.median(sc[sl].max(1))), short=float(np.median(sc[sl].min(1))), ratio=float(np.median(sc[sl].max(1)/sc[sl].min(1))),
                           opa=float(np.median(opa[sl])), ang=float(np.median(ang[sl])), near=float((d[sl] < 0.1).mean()))
    print(name, "live", len(xyz), "voxels", len(out), "global: long %.2fmm short %.3fmm ratio %.1f opa %.3f ang %.1fdeg within10cm %.3f" % (np.median(sc.max(1))*1000, np.median(sc.min(1))*1000, np.median(sc.max(1)/sc.min(1)), np.median(opa), np.median(ang), (d < 0.1).mean()))
    return out
ours = stats("ours_F6", *load_ours()); ref = stats("reference", *load_ref())
common = [k for k in ours if k in ref and ours[k]["n"] >= 30 and ref[k]["n"] >= 30]
print("paired voxels", len(common))
def med(f): return float(np.median([f(ours[k], ref[k]) for k in common]))
print("count ratio ref/ours median %.2f" % med(lambda o, r: r["n"]/o["n"]))
print("long axis ours %.2fmm ref %.2fmm | short ours %.3f ref %.3f | ratio ours %.1f ref %.1f" % (med(lambda o, r: o["long"])*1000, med(lambda o, r: r["long"])*1000, med(lambda o, r: o["short"])*1000, med(lambda o, r: r["short"])*1000, med(lambda o, r: o["ratio"]), med(lambda o, r: r["ratio"])))
print("opacity ours %.3f ref %.3f | short-axis-vs-normal ours %.1fdeg ref %.1fdeg | within10cm ours %.3f ref %.3f" % (med(lambda o, r: o["opa"]), med(lambda o, r: r["opa"]), med(lambda o, r: o["ang"]), med(lambda o, r: r["ang"]), med(lambda o, r: o["near"]), med(lambda o, r: r["near"])))
# where does the reference put MORE gaussians than us? split by count-ratio quartile and report our angle/size there
rat = np.array([ref[k]["n"]/ours[k]["n"] for k in common]); hi = rat > np.percentile(rat, 75)
print("voxels where ref has >%.1fx our count (top quartile): ours long %.2fmm ref %.2fmm, ours ang %.1f ref %.1f, ours opa %.3f ref %.3f" % (np.percentile(rat, 75), np.median([ours[k]["long"] for k, h in zip(common, hi) if h])*1000, np.median([ref[k]["long"] for k, h in zip(common, hi) if h])*1000, np.median([ours[k]["ang"] for k, h in zip(common, hi) if h]), np.median([ref[k]["ang"] for k, h in zip(common, hi) if h]), np.median([ours[k]["opa"] for k, h in zip(common, hi) if h]), np.median([ref[k]["opa"] for k, h in zip(common, hi) if h])))
only_ref = sum(1 for k in ref if k not in ours and ref[k]["n"] >= 30); only_ours = sum(1 for k in ours if k not in ref and ours[k]["n"] >= 30)
print("voxels only reference populated", only_ref, "| only ours", only_ours)
