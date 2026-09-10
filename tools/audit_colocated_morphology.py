"""Co-located morphology: voxelize ours and the aligned reference at 0.5 m and
compare, voxel by voxel where both are present, count / size / opacity /
orientation. Orientation = angle between each gaussian's shortest axis and the
local LiDAR surface normal (0 deg = a disk lying on the surface).

    python tools/audit_colocated_morphology.py --checkpoint RUN/delivery_f6/merged.pt \\
        --reference-ply USAgs.ply --reference-alignment usa_gs_alignment.json \\
        --tile-inputs-root RUN/tile_inputs_v9 --label ours_F6

The LiDAR surface comes from the tile initialization clouds, either derived
from --tile-inputs-root (Tile_<n>/initialization_full_lidar.ply for --tiles)
or listed explicitly with --lidar-ply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIDAR_PLY_NAME = "initialization_full_lidar.ply"


def quat_to_R(q):  # w x y z
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    R = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
    return R


def load_ours(checkpoint: Path):
    import torch

    p = torch.load(checkpoint, map_location="cpu", weights_only=False)["params"]
    return p["means"].numpy(), torch.sigmoid(p["opacities"].float()).numpy(), np.exp(p["scales"].numpy()), p["quats"].numpy()


def load_ref(reference_ply: Path, reference_alignment: Path):
    from tools.inspect_gaussian_ply import _read_ply

    v, _ = _read_ply(reference_ply)
    A = np.array(json.loads(reference_alignment.read_text(encoding="utf-8"))["transform"])
    xyz = np.stack([v["x"], v["y"], v["z"]], 1) @ A[:3, :3].T + A[:3, 3]
    opa = 1/(1+np.exp(-np.asarray(v["opacity"])))
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    # The alignment rotation is ~identity (0.05 deg), so the raw quaternions
    # are used for the orientation statistics without re-rotating them.
    return xyz.astype(np.float32), opa.astype(np.float32), sc.astype(np.float32), np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)


def lidar_ply_paths(args: argparse.Namespace) -> list[Path]:
    if args.lidar_ply:
        return list(args.lidar_ply)
    return [args.tile_inputs_root / f"Tile_{t}" / LIDAR_PLY_NAME for t in args.tiles]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="our checkpoint (.pt), e.g. a merged delivery")
    parser.add_argument("--reference-ply", type=Path, required=True, help="reference delivery PLY")
    parser.add_argument(
        "--reference-alignment", type=Path, required=True,
        help="JSON carrying the rigid 'transform' that brings the reference into our frame",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tile-inputs-root", type=Path, help=f"root holding Tile_<n>/{LIDAR_PLY_NAME}")
    source.add_argument("--lidar-ply", type=Path, action="append", help="explicit LiDAR PLY (repeatable)")
    parser.add_argument("--tiles", type=int, nargs="+", default=[0, 1, 2, 3], help="tiles under --tile-inputs-root")
    parser.add_argument("--label", default="ours", help="name printed for our checkpoint's row")
    parser.add_argument("--voxel", type=float, default=0.5, help="voxel edge in metres")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lidar_paths = lidar_ply_paths(args)
    for label, path in (
        ("--checkpoint", args.checkpoint),
        ("--reference-ply", args.reference_ply),
        ("--reference-alignment", args.reference_alignment),
        *(("LiDAR PLY", path) for path in lidar_paths),
    ):
        if not path.exists():
            print(f"audit_colocated_morphology: {label} not found: {path}", file=sys.stderr)
            return 2

    from tools.inspect_gaussian_ply import _read_ply

    V = args.voxel
    # LiDAR surface + normals from the tile geometry (k30 PCA normals) if available, else PCA on the fly
    lid = np.concatenate([np.stack([v["x"], v["y"], v["z"]], 1)[::4] for v in (_read_ply(path)[0] for path in lidar_paths)]).astype(np.float32)
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

    ours = stats(args.label, *load_ours(args.checkpoint)); ref = stats("reference", *load_ref(args.reference_ply, args.reference_alignment))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
