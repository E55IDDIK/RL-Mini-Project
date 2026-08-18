@echo off
setlocal enabledelayedexpansion

echo =========================================
echo 1. Evaluating all training checkpoints...
echo =========================================
python -m src.eval_all_checkpoints
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo =========================================
echo 2. Evaluating final models ^& baselines...
echo =========================================
python -m src.eval
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo =========================================
echo 3. Generating figures...
echo =========================================
python -m src.plot
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo Done! All figures and tables have been regenerated.
echo Check the 'figures\' and 'results\' directories.
