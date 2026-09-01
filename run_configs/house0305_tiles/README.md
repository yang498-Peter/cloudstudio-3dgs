# house0305 tile-run configs (machine-B actual state)

These are the files that actually ran, committed so the repository matches the
machine. Absolute paths inside them are machine-B paths.

- `tile0_vendorclean.json` - the clean vendor lifecycle boundary probe:
  pre-optimizer order, immediate cull 0.10->0.05, reset clamp 0.2/300, world
  split 0.2 m, NO CloudStudio cull protections, corrected per-tile backdrops,
  epoch permutation, footprint-weighted gradient at 1.5e-4, stop at 1500.
- `tile0_armE.json` - the accumulated arm-E base the stager derives tiles from.
- `tile0_full_earlystop.json` - the early-stop production config this probe
  replaced; kept for the record, not as a recommendation.
- `stage_full_run.py` - derives per-tile full configs from the arm base
  (per-tile init paths, per-tile cropped backdrops, controlled stops).
- `drive_tiles.cmd` - serial three-tile driver; treats a
  ControlledTrainingInterruption exit as planned, not failure.
