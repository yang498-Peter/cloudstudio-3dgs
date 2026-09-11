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
