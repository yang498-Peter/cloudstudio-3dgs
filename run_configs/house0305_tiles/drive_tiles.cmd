@echo off
rem Unattended three-tile driver. Each tile runs its signed 20-epoch schedule but
rem stops at a measured budget, because dead mass rises monotonically with
rem training and the photometric optimum is far earlier than 20 epochs.
rem A controlled stop exits non-zero BY DESIGN, so it must not be read as failure.
setlocal
call C:\Peter\cloudstudio-3dgs-gate1\train\env_machine_b.cmd
cd /d C:\Peter\cloudstudio-3dgs-gate1
set RUN=C:\Peter\3dgs-runs\house0305_sop
set PY=C:\Peter\cloudstudio-3dgs\.venv-train\Scripts\python.exe
set STAGER=C:\Users\Remot\AppData\Local\Temp\claude\C--Peter-CLOUDSUTDIO-WINDOWS\06ee1c7b-43bb-45f7-be7d-ce6fa8511c66\scratchpad\stage_full_run.py
set STATUS=%RUN%\tile_driver_status.txt

rem Arm base is a parameter, not a constant: hardcoding it silently overwrote
rem the staged configs with a different lifecycle once already.
if "%ARM%"=="" set ARM=armVC
echo [start] driver begun, arm=%ARM% > "%STATUS%"

call :run_tile 0
if errorlevel 1 exit /b 1
call :run_tile 1
if errorlevel 1 exit /b 1
call :run_tile 2
if errorlevel 1 exit /b 1

echo [complete] all three tiles trained >> "%STATUS%"
exit /b 0

:run_tile
set T=%1
echo [stage] tile %T% >> "%STATUS%"
%PY% "%STAGER%" %ARM% %T% >> "%STATUS%" 2>&1
if errorlevel 1 (
  echo [FAIL] staging tile %T% >> "%STATUS%"
  exit /b 1
)
if exist "%RUN%\tile%T%_full" rmdir /s /q "%RUN%\tile%T%_full"
echo [train] tile %T% started >> "%STATUS%"
%PY% tools/train_gsplat.py --config "%RUN%\tile%T%_full.json" > "%RUN%\tile%T%_full.log" 2> "%RUN%\tile%T%_full.log.err"
if not errorlevel 1 goto :tile_ok
rem non-zero: a planned controlled stop is success, anything else is failure
findstr /C:"ControlledTrainingInterruption" "%RUN%\tile%T%_full.log.err" >nul
if errorlevel 1 (
  echo [FAIL] tile %T% training exited nonzero >> "%STATUS%"
  exit /b 1
)
echo [done] tile %T% stopped at planned budget >> "%STATUS%"
exit /b 0
:tile_ok
echo [done] tile %T% finished >> "%STATUS%"
exit /b 0
