@echo off
rem Smoke training: gs2 scene, 3DGUT (raw fisheye + circular masks), MCMC.
rem 8GB VRAM budget: data_factor 4 (728px), cap-max 1M gaussians, 10k steps.
rem Usage: train\run_smoke_gs2.cmd  (from repo root, or anywhere)

set S1_KEEP_FISHEYE=1
set DATA_DIR=G:\3dgs-datasets\gs2_keyframes
set RESULT_DIR=G:\3dgs-results\gs2_smoke

cd /d G:\cloudstudio-3dgs\external\gsplat\examples
python simple_trainer.py mcmc ^
  --disable_viewer ^
  --data_factor 4 ^
  --with_ut --with_eval3d ^
  --strategy.cap-max 1000000 ^
  --max_steps 10000 ^
  --eval_steps 5000 10000 ^
  --save_steps 10000 ^
  --save_ply --ply_steps 10000 ^
  --data_dir %DATA_DIR% ^
  --result_dir %RESULT_DIR%
