@echo off
REM Smart-Care Edge Emergency Node - laptop launcher (Windows)
REM Runs the node against the local webcam and serves the dashboard on :8000.
setlocal
cd /d "%~dp0\..\.."
set PYTHONIOENCODING=utf-8

set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo Starting Smart-Care Edge node (webcam)...
echo Open the dashboard at http://localhost:8000/
"%PY%" demo_combined\edge_emergency\edge_node.py ^
  --camera 0 --short-side 320 --det-every 2 --recog-every 5 --threads 4 --port 8000 %*

endlocal
