# DIAG-40 ROI-only sharpness (2026-09-11)

Laplacian variance of the region-of-interest box only (`tools/score_compare_roi.py`, boxes from `DIAG_40/selection.json` `roi_in_crops`, min 100 region samples, box >= 64 px), rendered with `build_three_way_compare.py --sample-ids` on the pinned ROI views (`roi_compare_ids.json`: indoor 49, outdoor 45; `roi_compare_ids_u1.json`: the 5 ROI views inside the U1 five-view set). All 40-view arms: H=3000, cap 1.34x init. Rows are per arm; ours/ref is the paired per-frame ratio (same view, same box).

| region | arm | view set | n | ours/photo | ref/photo | ours/ref median | ours/ref Q1 | Q3 | frames ours>=ref |
|---|---|---|---|---|---|---|---|---|---|
| indoor | G0_c134 | ROI-all | 49 | 0.628 | 1.301 | **0.501** | 0.367 | 0.754 | 5 |
| indoor | G0_c134 | U1-ROI(5) | 5 | 0.837 | 2.053 | **0.500** | 0.333 | 0.591 | 0 |
| indoor | G1_c134 | ROI-all | 49 | 0.770 | 1.301 | **0.493** | 0.441 | 0.680 | 7 |
| indoor | G1_c134 | U1-ROI(5) | 5 | 0.970 | 2.053 | **0.491** | 0.428 | 0.514 | 0 |
| indoor | R1_c134 | ROI-all | 49 | 0.737 | 1.301 | **0.566** | 0.405 | 0.809 | 8 |
| indoor | R1_c134 | U1-ROI(5) | 5 | 1.249 | 2.053 | **0.609** | 0.383 | 0.781 | 0 |
| outdoor | G0_c134 | ROI-all | 45 | 0.749 | 1.290 | **0.605** | 0.545 | 0.676 | 0 |
| outdoor | G0_c134 | U1-ROI(5) | 5 | 0.495 | 0.799 | **0.615** | 0.580 | 0.646 | 0 |
| outdoor | G1_c134 | ROI-all | 45 | 0.739 | 1.290 | **0.601** | 0.537 | 0.648 | 0 |
| outdoor | G1_c134 | U1-ROI(5) | 5 | 0.491 | 0.799 | **0.613** | 0.570 | 0.639 | 0 |
| outdoor | R1_c134 | ROI-all | 45 | 0.742 | 1.290 | **0.602** | 0.540 | 0.665 | 0 |
| outdoor | R1_c134 | U1-ROI(5) | 5 | 0.490 | 0.799 | **0.611** | 0.575 | 0.641 | 0 |

Paired whole-panel check (40-view checkpoint rendered on the U1 five-view arm's own six compare frames, `compare_on_U1_5/`): indoor 0.316 vs the five-view arm's 0.968 on the same frames (ref 0.919); outdoor 0.265 vs 0.557 (ref 0.876). The five-view arms' checkpoints were already deleted, so their ROI-only rows are not available; the indoor five-view arm's two ROI frames from its own compare set scored ours/ref 1.015 / 0.931.

Source files (local, not committed): `diag_v2/<region>/runs/<arm>/compare_roi_compare_ids{,_u1}/`, `.roi.json`, `.roi.txt`.

## Brightness-matched (`--match-brightness`): each ROI crop's luma scaled to the photo's mean before the Laplacian

Raw Laplacian variance scales with the square of a global gain. Our canonical renders are brighter than the photos (whole-panel luma ratio 1.16 for indoor R1) and the reference is much brighter than the photo inside the dark door ROI (raw ref/photo 2.2 vs matched 0.73), so the raw table above carries a brightness term in both numerator and denominator. This table removes it and is the one to judge by.

| region | arm | view set | n | ours/photo | ref/photo | ours/ref median | ours/ref Q1 | Q3 | frames ours>=ref |
|---|---|---|---|---|---|---|---|---|---|
| indoor | G0_c134 | ROI-all | 49 | 0.452 | 0.508 | **0.863** | 0.716 | 1.041 | 15 |
| indoor | G0_c134 | U1-ROI(5) | 5 | 0.498 | 0.526 | **0.863** | 0.742 | 0.958 | 0 |
| indoor | G1_c134 | ROI-all | 49 | 0.529 | 0.508 | **0.990** | 0.771 | 1.152 | 20 |
| indoor | G1_c134 | U1-ROI(5) | 5 | 0.478 | 0.526 | **0.910** | 0.789 | 0.998 | 1 |
| indoor | R1_c134 | ROI-all | 49 | 0.477 | 0.508 | **0.881** | 0.777 | 1.199 | 18 |
| indoor | R1_c134 | U1-ROI(5) | 5 | 0.636 | 0.526 | **0.813** | 0.777 | 1.199 | 2 |
| outdoor | G0_c134 | ROI-all | 45 | 0.852 | 1.389 | **0.597** | 0.555 | 0.673 | 0 |
| outdoor | G0_c134 | U1-ROI(5) | 5 | 0.382 | 0.685 | **0.551** | 0.545 | 0.578 | 0 |
| outdoor | G1_c134 | ROI-all | 45 | 0.843 | 1.389 | **0.597** | 0.550 | 0.649 | 0 |
| outdoor | G1_c134 | U1-ROI(5) | 5 | 0.378 | 0.685 | **0.552** | 0.536 | 0.574 | 0 |
| outdoor | R1_c134 | ROI-all | 45 | 0.837 | 1.389 | **0.595** | 0.548 | 0.666 | 0 |
| outdoor | R1_c134 | U1-ROI(5) | 5 | 0.378 | 0.685 | **0.549** | 0.539 | 0.574 | 0 |

## Does the brightness term touch the full-delivery numbers?

Re-scored the delivery compare strips the same way (whole panel, luma matched to the photo): G9 (SH1) 0.211 raw -> 0.219 matched, R1d 0.238 -> 0.232, reference 0.574 -> 0.534; luma ratios 0.95-0.96 (ours, merged with baked per-tile gains) and 1.01 (reference). The delivery-level gap (ours ~0.23 vs reference ~0.53 of the photo) stands; the brightness confound is specific to single-tile canonical checkpoints (no exposure gain applied at render time) and to dark ROIs such as the door leaf.

## X1 and D-line arms (brightness-matched, ROI-all, appended 19:05)

| region | arm | n | ours/photo | ref/photo | ours/ref median | Q1 | Q3 |
|---|---|---|---|---|---|---|---|
| indoor | D1_c134 | 49 | 0.526 | 0.508 | **0.906** | 0.816 | 1.074 |
| indoor | D1_c134_v6f | 49 | 0.505 | 0.508 | **0.825** | 0.685 | 1.065 |
| indoor | R1_c134_v6f | 49 | 0.473 | 0.508 | **0.825** | 0.704 | 1.008 |
| indoor | X1_c134 | 49 | 0.510 | 0.508 | **0.858** | 0.713 | 1.079 |
| outdoor | D1_c134 | 45 | 0.856 | 1.389 | **0.592** | 0.551 | 0.652 |
| outdoor | D1_c134_v6f | 45 | 0.859 | 1.389 | **0.597** | 0.546 | 0.658 |
| outdoor | R1_c134_v6f | 45 | 0.839 | 1.389 | **0.598** | 0.551 | 0.663 |
| outdoor | X1_c134 | 45 | 0.874 | 1.389 | **0.612** | 0.560 | 0.683 |

Reference rows (same metric): indoor R1 0.881 / G0 0.863 / G1 0.990; outdoor R1 0.595 / G0 0.597 / G1 0.597. Indoor spread across the eight near-identical D/R arms is 0.825-0.906, i.e. the ROI-all noise band indoors is about +-5%; outdoors 0.592-0.598 (+-1%).

## Full Tile_1 20k arms on the 49 indoor ROI views (brightness-matched, appended 23:52)

| arm | n | ours/photo | ref/photo | ours/ref median |
|---|---|---|---|---|
| tile1_R1d_20k | 49 | 0.219 | 0.508 | **0.366** |
| tile1_S1_anchor_20k (hard prune probe) | 49 | 0.224 | 0.508 | **0.390** |

The door ROI itself is unchanged by the hard prune (+6%, inside the indoor +-5% band) while the sky views and walls elsewhere are destroyed (README row "S1 上界探针结果"). K1 / S2 rows follow when they land.
| tile1_S2_growthgate_20k (growth gate only) | 49 | 0.217 | 0.508 | **0.384** |
| tile1_K1_sky_20k (sky alpha, weak mask) | 49 | 0.226 | 0.508 | **0.453** |
| tile1_K2_sky_20k (sky alpha, erosion 4 / guard 6) | 49 | 0.271 | 0.508 | **0.465** |
| tile1_R1d_rerun_20k (identical config rerun) | 49 | 0.174 | 0.508 | **0.315** |

Rerun spread at 20k scale (single pair): ROI-all 0.315-0.366 (+-7%), whole-panel 0.076-0.082 (+-4%), off-trajectory PSNR 15.16-16.14 (+-0.5 dB). K1/K2 ROI (0.453/0.465) and K2 whole-panel (0.107) lie outside that spread; their PSNR gains do not.
| tile1_F3_nopitchup_20k (551 pitch_up faces held out) | 44 | 0.176 | 0.508 | **0.385** |
| tile1_K3_sky_20k (full sky region, no guard) | 49 | 0.250 | 0.508 | **0.472** |
| tile1_O1_ownership_20k (cached tile ownership) | 49 | 0.255 | 0.508 | **0.509** |
| tile1_O2_ownership_sky_20k (K2 + ownership) | 49 | 0.302 | 0.508 | **0.543** |
| tile1_B1_standin_20k (stand-in backdrop) | 49 | 0.126 | 0.508 | **0.225** (fair 0.225) |
| tile1_B1_standin_K2sky_20k = B2 (sky alpha + stand-in) | 49 | 0.193 | 0.508 | **0.393** (fair 0.395) |
| tile1_R1d_20k under the stand-in backdrop (fair) | 49 | 0.219 | 0.508 | **0.367** |
