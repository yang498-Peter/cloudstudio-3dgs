# Exposure policy audit (WP01 §4.2)

Facts only. Line numbers are against the research worktree at
`C:\Peter\cloudstudio-3dgs-work` on 2026-09-11 (HEAD `3eef728`). No exposure
behaviour was changed; `RenderSpec.exposure_policy` records what is found here.

## 1. What training learns and where it is applied

| Fact | Location |
| --- | --- |
| One scalar log-gain per **training source image** (face samples `base::face` share the base image's gain), clamped to `[-ln 2, ln 2]`, grouped per physical camera for the mean anchor. | `cloudstudio_3dgs/training/exposure.py:53-95` (`ExposureCompensator.__init__`), `:105-112` (`gain()`), `trainer.py:3806-3814` (`group_by_image=trainset.camera_id_by_image`) |
| Defaults: `zero_mean_projection=False`, `mean_anchor_weight=0.0`, `regularization_weight=1e-2`. Both target configs set only `enabled: true, learning_rate: 0.005`, so the gains have no zero-mean projection and no per-camera anchor (schedule audit warning E001). | `exposure.py:20-27`; `run_configs/house0305_tiles/v9/tile0_R1_range0_20k.json` `exposure_compensation`; `C:\Peter\3dgs-runs\house0305_sop\delivery_eval.json` `exposure_compensation` |
| The gain is applied **only inside `_render_supervision_loss`**: after background compositing, `rendered = rendered * rgb_gain`, then L1 / gradient / LiDAR-RGB / SSIM losses read the gained render. With `decoupled_ssim` the SSIM structure term reads the raw render and receives the gain as `luminance_gain`. | `trainer.py:2795` (def), `:2869-2873` (multiply), `:2917-2928` (decoupled SSIM) |
| The training loop passes `rgb_gain = exposure.gain(sample.image_id.split("::")[0])` for every training sample; the gain optimizer steps and `project_zero_mean()` runs after each step (a no-op with the defaults). | `trainer.py:4234-4236`, `:4504-4507` |
| The learned gains are persisted in every checkpoint under `auxiliary_params["exposure_log_gains"]`. | `trainer.py:3816`, `cloudstudio_3dgs/training/checkpoint.py:217-220` |
| `exposure.report()` (mean / p50 / p95 / saturation) is written to `training_state` and the run manifest; it is a report, not an applied transform. | `trainer.py:5052-5054`, `:5139` |

## 2. Do the trainer's own validation renders apply the learned gains?

**No.** Every validation path calls `backend.render` with only `with_range`
and `background_rgb`; no gain is passed and `_render_supervision_loss` is not
on that path.

| Path | Location | Render call |
| --- | --- | --- |
| Validation dataset | `trainer.py:3482-3495` | `valset = S1TrainingDataset(..., split="val")` — the **raw fisheye** images, not faces (`face_split.validation = "raw_fisheye"`, `trainer.py:2107`) |
| Golden views (periodic) | `trainer.py:4902-4912` → `golden_eval.py:264-284` → `golden_eval.py:102-139` | `_evaluate_views`: `render_options = {"with_range": ...}` plus `background_rgb`; `backend.render(params, sample, **render_options)` at line 137-139. No gain argument exists on the function. |
| Full validation (periodic) | `trainer.py:4944-4952` → `golden_eval.py:297-320` → same `_evaluate_views` | as above |
| Final evaluation artifacts | `trainer.py:5100-5106` → `_save_evaluation_artifacts` `trainer.py:3240-3252` | `render_options = {"with_range": has_range}` plus `background_rgb`; no gain |

The module docstring states the intent explicitly: "validation always renders
at gain 1.0 so metrics stay honest" (`exposure.py:7-9`). The code matches.

Consequence: the trainer's PSNR/SSIM history is measured in the **model
frame** (gain 1.0) on the raw fisheye through 3DGUT (protocol B), while the
photometric loss is minimised in the **per-image gained frame** on Face4
pinhole faces (protocol A).

## 3. Does `merge_v28_tile_checkpoints.py --harmonize-exposure` bake per-tile gains into `merged.pt` colours?

**Yes, one scalar per tile, DC band only.**

| Fact | Location |
| --- | --- |
| Flag `--harmonize-exposure` (store_true, default off). | `tools/merge_v28_tile_checkpoints.py:103-109` |
| For each tile: `tile_gain = median(exp(auxiliary_params["exposure_log_gains"]))` — the **median over that tile's training images**, not a per-image value. Raises if the checkpoint carries no gains. | `:152-159` |
| Bake: `sh0' = sh0 * g + (g - 1) * 0.5 / C0`, i.e. DC colour `rgb' = rgb * g` (since `rgb = sh0 * C0 + 0.5`). Higher SH bands (`shN`) are **not** scaled. | `:160-166` |
| Without the flag `tile_gain = None` and colours are untouched. | `:143-144` |
| Recorded per tile as `records[*].exposure_gain_applied` and globally as `exposure_harmonized`; the whole report is embedded in the checkpoint as `checkpoint["merge"]`, so a merged checkpoint carries the evidence. | `:209`, `:237`, `:261-266` |

Real deliveries on this machine (`merge_report.json`):

| Delivery | `exposure_harmonized` | `exposure_gain_applied` per tile 0..3 |
| --- | --- | --- |
| `delivery_g9` | true | 0.8985, 0.9430, 0.9375, 0.9634 |
| `delivery_f6` | true | 0.9156, 0.9772, 0.9696, 0.9652 |

So G9 `merged.pt` colours sit in the photograph's frame per tile (each tile
darkened by 4-10 % relative to its own model frame). A single-tile
checkpoint (e.g. `tile0_R1_range0_20k/best.pt`) rendered at gain 1.0 sits in
its model frame. The two are not colour-comparable without the gain, and
`shN` bands are un-gained in both.

## 4. Do the evaluators apply any gain?

**No.** Before this change none of `tools/sharpness_metrics.py`,
`tools/evaluate_probe_views.py`, `tools/build_three_way_compare.py`,
`tools/build_offtrajectory_compare.py`, `tools/roundtrip_checkpoint_ply.py`
read `auxiliary_params` or multiplied a render (`grep -n exposure` over the
five files: no hits). They call `backend.render(..., background_rgb=...)`
exactly as the trainer's validation does, i.e. gain 1.0 — matching the
trainer's validation convention (§2), not its loss (§1).

## 5. Resolved policy exposed in `RenderSpec.exposure_policy`

`cloudstudio_3dgs/training/render_spec.py::RenderSpec._resolve_exposure`:

| Situation | `policy` | status |
| --- | --- | --- |
| `exposure_compensation.enabled` false and no PPISP | `none` | propagated |
| Enabled; checkpoint metadata given and `merge.exposure_harmonized` is true | `per_tile_gain` (with `baked_tile_gains` from `merge.records`) | propagated — the gains are already in the colours; the evaluator renders them at 1.0 |
| Enabled; checkpoint metadata given, no harmonized merge (single-tile checkpoint, or merge without the flag) | `canonical` (gain 1.0, the trainer-validation convention; learned per-image gains exist in the checkpoint and are not applied) | propagated |
| Enabled; checkpoint metadata **not** inspected (a reference PLY, or a caller passing `params` only) | `canonical` | **unpropagated** — a merged checkpoint with baked gains cannot be told from a single-tile one |
| `per_image_gain` | reserved for an evaluator that applies the learned per-image gains; **no evaluator does this today** | — |

The `training` sub-record carries `enabled`, `learning_rate`,
`zero_mean_projection`, `mean_anchor_weight`, `ppisp` so a spec from a run with
an anchor differs in fingerprint from one without.

## 6. Open items (not changed here)

- Whether protocol-A scores should be taken in the gained frame (apply each
  view's learned gain, `per_image_gain`) is a protocol decision; the
  `--honour-render-mode` flag does not touch exposure.
- A merged checkpoint bakes only the DC band; if `shN` energy is significant
  the per-tile harmonisation is incomplete by construction (`:160-166`).
