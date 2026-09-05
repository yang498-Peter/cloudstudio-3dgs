@echo off
rem CUDA test channel for the machine-B host. GitHub-hosted runners have no
rem GPU, so this is the only place the real rasterizer tests execute; the
rem policy check is the same one CI applies to the other two channels.
setlocal
set HERE=%~dp0
set REPO=%HERE%..
if "%CS3DGS_ENV%"=="" set CS3DGS_ENV=%REPO%\train\env_machine_b.cmd
if not exist "%CS3DGS_ENV%" (
  echo missing environment script %CS3DGS_ENV%; set CS3DGS_ENV to the machine env
  exit /b 2
)
call "%CS3DGS_ENV%"
cd /d "%REPO%"
set PYTHONPATH=%REPO%;%PYTHONPATH%
set PYTHONIOENCODING=utf-8
if "%OUT%"=="" set OUT=%REPO%\research\quality_recovery_v1\ci
if not exist "%OUT%" mkdir "%OUT%"
python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors --junitxml "%OUT%\junit-cuda.xml"
python tests\check_collection.py "%OUT%\junit-cuda.xml" --channel cuda --min-tests 700 --summary "%OUT%\collection-cuda.json"
exit /b %ERRORLEVEL%
