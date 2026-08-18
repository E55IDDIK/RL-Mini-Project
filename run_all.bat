@echo off

setlocal enabledelayedexpansion

cd /d "%~dp0"

set "STATUS=0"

REM --- pick an interpreter: plain `python`, else the `py -3` launcher --------
set "PY="
python --version >nul 2>&1
if !ERRORLEVEL! EQU 0 set "PY=python"
if not defined PY (
    py -3 --version >nul 2>&1
    if !ERRORLEVEL! EQU 0 set "PY=py -3"
)
if not defined PY (
    echo ERROR: no python interpreter found on PATH.
    exit /b 1
)
for /f "delims=" %%V in ('%PY% --version 2^>^&1') do echo Using interpreter: %%V

if not exist "results"     md "results"
if not exist "figures"     md "figures"
if not exist "checkpoints" md "checkpoints"
if not exist "logs"        md "logs"

REM --- fail loudly if the checkpoints are missing ----------------------------
REM Without this the pipeline runs to completion but writes EMPTY csv files and
REM blank figures, exiting 0 - a silent failure that is worse than a crash.
set "NCKPT=0"
for /f %%A in ('dir /b /s "checkpoints\*_step*.pt" 2^>nul ^| find /c /v ""') do set "NCKPT=%%A"
if "!NCKPT!"=="0" (
    echo ERROR: no checkpoints found in checkpoints\.
    echo        Expected files like checkpoints\gnn_dqn_seed42_step20000.pt
    echo        Train first, or restore the checkpoints shipped with this repo.
    exit /b 1
)
echo Found !NCKPT! checkpoint^(s^).

echo.
echo =========================================
echo 1. Evaluating all training checkpoints...
echo =========================================
%PY% -m src.eval_all_checkpoints
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: step 1 ^(eval_all_checkpoints^) failed.
    exit /b !ERRORLEVEL!
)

echo.
echo =========================================
echo 2. Evaluating final models ^& baselines...
echo =========================================
%PY% -m src.eval
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: step 2 ^(eval^) failed.
    exit /b !ERRORLEVEL!
)

echo.
echo =========================================
echo 3. Generating figures...
echo =========================================
%PY% -m src.plot
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: step 3 ^(plot^) failed.
    exit /b !ERRORLEVEL!
)

REM --- verify the artefacts actually exist and are non-trivial ---------------
echo.
echo =========================================
echo Verifying outputs
echo =========================================
call :check_csv "results\learning_curve_data.csv"
call :check_csv "results\eval_comparison.csv"
call :check_file "figures\learning_curve_return.pdf"
call :check_file "figures\learning_curve_metrics.pdf"
call :check_file "figures\benchmark_comparison.pdf"

if not "!STATUS!"=="0" (
    echo.
    echo FAILED: some outputs were not produced.
    exit /b 1
)

echo.
echo Done. All results and figures regenerated.
echo   results\  -^> csv tables
echo   figures\  -^> vector pdf figures
exit /b 0

REM ==========================================================================
REM Subroutines
REM ==========================================================================

:check_csv
REM A header-only csv is the silent-failure case, so require >= 1 data row.
if not exist "%~1" (
    echo   MISSING: %~1
    set "STATUS=1"
    goto :eof
)
set "LINES=0"
for /f %%A in ('find /c /v "" ^< "%~1"') do set "LINES=%%A"
set /a ROWS=!LINES!-1
if !ROWS! LSS 1 (
    echo   NO DATA ROWS: %~1
    set "STATUS=1"
) else (
    echo   OK  %~1  ^(!ROWS! rows^)
)
goto :eof

:check_file
if not exist "%~1" (
    echo   MISSING: %~1
    set "STATUS=1"
    goto :eof
)
set "SIZE=0"
for %%A in ("%~1") do set "SIZE=%%~zA"
if !SIZE! LSS 1 (
    echo   EMPTY: %~1
    set "STATUS=1"
) else (
    echo   OK  %~1
)
goto :eof