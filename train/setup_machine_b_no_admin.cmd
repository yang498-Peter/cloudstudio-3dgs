@echo off
rem One-shot gsplat bring-up for a no-admin Windows machine (machine B).
rem Prerequisites (no administrator rights needed):
rem   1. Python 3.12 x64 on PATH; VS2022 BuildTools with C++ x64 workload.
rem   2. .venv-train:  python -m venv .venv-train
rem      .venv-train\Scripts\pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
rem      .venv-train\Scripts\pip install ninja numpy jaxtyping nvtx rich pillow scipy
rem   3. CUDA 12.8 compiler env via micromamba (no admin installer):
rem      micromamba create -p .cuda128 -c nvidia/label/cuda-12.8.1 -c conda-forge ^
rem        cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 ^
rem        libcusparse-dev libcublas-dev libcusolver-dev libcurand-dev cuda-profiler-api
rem      copy .cuda128\Library\lib\*.lib .cuda128\Library\lib\x64\
rem      (torch cpp_extension only searches lib\x64)
rem   4. Clean gsplat checkout at the locked commit:
rem      git clone https://github.com/nerfstudio-project/gsplat external\gsplat-clean
rem      git -C external\gsplat-clean checkout --detach <commit from upstream\cloudstudio_trainer.lock.json>
rem      git -C external\gsplat-clean submodule update --init --depth 1 gsplat/cuda/csrc/third_party/glm
rem
rem The extension is JIT-built on first import (into .torch-ext), keeping the
rem gsplat worktree byte-clean so verify_gsplat_runtime's clean_vcs_commit
rem check passes. Do NOT set any GSPLAT BUILD_* selection variable: a partial
rem build (the first machine used BUILD_3DGUT=1) silently drops the 3DGS
rem kernel group and with it quat_scale_to_covar_preci_fwd - full MCMC dies.
setlocal
call "%~dp0env_machine_b.cmd" || exit /b 1

where nvcc >nul 2>&1 || (echo ERROR: nvcc missing - see prerequisites & exit /b 1)
python -c "import torch; assert torch.cuda.is_available(), 'torch sees no CUDA GPU'; print('torch', torch.__version__)" || exit /b 1

cd /d "%CS3DGS_ROOT%\external\gsplat-clean"
set BUILD_NO_CUDA=1
pip install -e . --no-build-isolation --no-deps --disable-pip-version-check
if errorlevel 1 (echo GSPLAT_PIP_INSTALL_FAILED & exit /b 1)
set BUILD_NO_CUDA=

echo === JIT-compiling full gsplat kernel set (sm_120, ~1-2 h first time) ===
python -c "from gsplat.cuda._backend import _C; import torch; assert _C is not None; assert hasattr(torch.ops.gsplat, 'quat_scale_to_covar_preci_fwd'), 'full-MCMC op missing - partial build?'; print('gsplat CUDA ready, full MCMC ops registered')"
if errorlevel 1 (echo GSPLAT_JIT_FAILED & exit /b 1)
echo SETUP_MACHINE_B_DONE
