@echo off
set PSY197B_RUNS_DIR=C:\PSY197B\runs
set PSY197B_RESULTS_DIR=C:\PSY197B\results
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist "%PSY197B_RUNS_DIR%" mkdir "%PSY197B_RUNS_DIR%"
if not exist "%PSY197B_RESULTS_DIR%" mkdir "%PSY197B_RESULTS_DIR%"
cd /d "%~dp0"
.venv\Scripts\python.exe src\vision\train_models.py %*
