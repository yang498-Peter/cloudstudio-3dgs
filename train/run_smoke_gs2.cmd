@echo off
rem Smoke training: gs2 scene, 3DGUT (raw fisheye + circular masks), MCMC.
rem Dataset expected at <repo>\data\gs2_keyframes (shipped in the handoff zip).
rem VRAM budget ~8GB: data_factor 4 (728px), cap-max 1M gaussians, 10k steps.

setlocal
set ROOT=%~dp0..
set S1_KEEP_FISHEYE=1
set DATA_DIR=%ROOT%\data\gs2_keyframes
set RESULT_DIR=%ROOT%\results\gs2_smoke

if not exist "%DATA_DIR%\sparse\0\cameras.txt" (echo ERROR: dataset not found at %DATA_DIR% & exit /b 1)

cd /d "%ROOT%\external\gsplat\examples"
rem The selective Windows build omits the base 3DGS covariance op used only
rem by MCMC position-noise injection; relocation/densification remain enabled.
python simple_trainer.py mcmc ^
  --disable_viewer ^
  --data_factor 4 ^
  --with_ut --with_eval3d ^
  --strategy.cap-max 1000000 ^
  --strategy.noise-injection-stop-iter 0 ^
  --max_steps 10000 ^
  --eval_steps 5000 10000 ^
  --save_steps 10000 ^
  --save_ply --ply_steps 10000 ^
  --data_dir "%DATA_DIR%" ^
  --result_dir "%RESULT_DIR%"
