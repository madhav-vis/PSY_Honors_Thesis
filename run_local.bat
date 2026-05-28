@echo off
REM Pipeline outputs go to local disk (not OneDrive).
set PSY197B_RUNS_DIR=C:\PSY197B\runs
set PSY197B_RESULTS_DIR=C:\PSY197B\results
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist "%PSY197B_RUNS_DIR%" mkdir "%PSY197B_RUNS_DIR%"
if not exist "%PSY197B_RESULTS_DIR%" mkdir "%PSY197B_RESULTS_DIR%"

cd /d "%~dp0"

if "%1"=="" (
  echo Usage: run_local.bat ^<main.py step^> [more args]
  echo Example: run_local.bat eeg
  echo          run_local.bat fuse
  echo          run_local.bat checks
  exit /b 1
)

.venv\Scripts\python.exe src\main.py %*
