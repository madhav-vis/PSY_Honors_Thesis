@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe src\evaluate.py --deploy --deploy-model clip_head %*
pause
