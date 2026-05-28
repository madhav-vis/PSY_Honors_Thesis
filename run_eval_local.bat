@echo off
set PSY197B_RUNS_DIR=C:\PSY197B\runs
set PSY197B_RESULTS_DIR=C:\PSY197B\results
REM 4 workers + batch 64 keeps GPU busy; drop to 2 if page-file errors return
set PSY_DATALOADER_WORKERS=4
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist "%PSY197B_RUNS_DIR%" mkdir "%PSY197B_RUNS_DIR%"
if not exist "%PSY197B_RESULTS_DIR%" mkdir "%PSY197B_RESULTS_DIR%"
cd /d "%~dp0"
.venv\Scripts\python.exe src\evaluate.py --vision-only --n-repeats 3 --resnet-batch-size 64 %*
