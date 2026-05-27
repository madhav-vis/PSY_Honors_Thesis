@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
.venv\Scripts\python.exe -m streamlit run src/vision/stream_annotator.py --server.port 8502
