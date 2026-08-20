@echo off
rem Canonical training-environment variables for the no-admin machine
rem (RTX 5070 Ti 16GB, CUDA via micromamba, torch 2.11.0+cu128 in .venv-train).
rem EVERY entry point (build, verify, training) must call this file: torch's
rem JIT extension cache keys on these flags, and any divergence between two
rem shells silently triggers a multi-hour full rebuild of gsplat.
rem
rem NOT setlocal: intended to be `call`ed so the variables persist.

set CS3DGS_ROOT=C:\Peter\cloudstudio-3dgs
set CUDA_HOME=%CS3DGS_ROOT%\.cuda128\Library
set CUDA_PATH=%CUDA_HOME%
set PATH=%CUDA_HOME%\bin;%CS3DGS_ROOT%\.venv-train\Scripts;%PATH%

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if "%VSPATH%"=="" (echo ERROR: MSVC Build Tools not found & exit /b 1)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul

rem Windows build fixes carried over from the first machine:
set VSLANG=1033
set DISTUTILS_USE_SDK=1
set MSSdk=1
set PYTHONPATH=%CS3DGS_ROOT%\train\build_compat;%PYTHONPATH%
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=%NUMBER_OF_PROCESSORS%

rem MSVC has no __builtin_clzll (host code in the 3DGS batch rasterizer);
rem map it via a force-included micro-header instead of patching the locked
rem source tree. See train\build_compat\msvc_clzll.h.
set NVCC_FLAGS=-Xcompiler /FI%CS3DGS_ROOT%\train\build_compat\msvc_clzll.h

rem Pin the JIT build dir: MSIX container virtualization redirects AppData
rem inconsistently between shells, splitting the build across two caches.
set TORCH_EXTENSIONS_DIR=%CS3DGS_ROOT%\.torch-ext

rem Host cl.exe compiles need libcu++'s nv/target header, which the conda
rem cccl package keeps under include\targets\x64 (build.py only forwards it
rem to nvcc).
set INCLUDE=%INCLUDE%;%CUDA_HOME%\include\targets\x64

rem Two never-called world-space BATCH rasterizer entry points do not link on
rem Windows: nvcc's host pass and direct cl disagree on one bool argument in
rem the MSVC back-reference mangling table (upstream has no Windows CI for
rem those files), and torch does not export radix_sort_pairs_impl<int64,4>.
rem The trainer only uses single-view rasterization() + 3DGUT + MCMC ops from
rem other translation units; runtime acceptance verifies every op we use.
set LINK=/FORCE:UNRESOLVED
