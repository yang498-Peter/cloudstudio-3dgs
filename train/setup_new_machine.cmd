@echo off
rem One-shot gsplat build + deps for a fresh Windows CUDA machine.
rem Run from anywhere; paths are relative to this repo.
rem
rem PREREQUISITES (install manually first, details in train\ENV.md):
rem   1. NVIDIA driver + CUDA Toolkit 12.8+  (nvcc in PATH)
rem   2. VS2022 Build Tools with "Desktop development with C++" (x64)
rem   3. Python 3.12 x64 in PATH
rem   4. PyTorch cu128:
rem        pip install torch --index-url https://download.pytorch.org/whl/cu128
rem
rem IMPORTANT: keep the repo on an ASCII-only path (no Chinese characters).

setlocal
set ROOT=%~dp0..

where nvcc >nul 2>&1
if errorlevel 1 (echo ERROR: nvcc not in PATH - install CUDA Toolkit 12.8+ & exit /b 1)
python -c "import torch; assert torch.cuda.is_available(), 'torch sees no CUDA GPU'; print('torch', torch.__version__)"
if errorlevel 1 (echo ERROR: PyTorch cu128 not working - see prerequisites & exit /b 1)

set VSPATH=
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if "%VSPATH%"=="" (echo ERROR: MSVC Build Tools not found & exit /b 1)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul

rem --- the three Windows build fixes discovered on the first machine ---
set VSLANG=1033
set DISTUTILS_USE_SDK=1
set MSSdk=1
rem Compile only for this GPU arch (RTX 5070 = Blackwell sm_120). This is the
rem difference between a ~10 min build and a >60 min all-arch build.
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=%NUMBER_OF_PROCESSORS%

cd /d "%ROOT%\external\gsplat"
echo === building gsplat (CUDA, sm_120 only) ===
pip install -e . --no-build-isolation --disable-pip-version-check
if errorlevel 1 (echo GSPLAT_BUILD_FAILED & exit /b 1)

echo === installing curated example deps (do NOT use examples\requirements.txt: it pins torch==2.9.1) ===
pip install viser nerfview pycolmap torchmetrics tyro pyyaml tensorboard imageio[ffmpeg] opencv-python-headless scipy scikit-learn matplotlib tqdm piexif splines tensorly laspy Pillow --disable-pip-version-check
if errorlevel 1 (echo DEPS_FAILED & exit /b 1)
pip install torchvision --index-url https://download.pytorch.org/whl/cu128 --no-deps --disable-pip-version-check
if errorlevel 1 (echo TORCHVISION_FAILED & exit /b 1)

python -c "import gsplat; print('gsplat', gsplat.__version__, 'import OK')"
echo SETUP_DONE - next: train\run_smoke_gs2.cmd
