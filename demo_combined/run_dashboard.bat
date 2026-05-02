@echo off
REM Launch the Medical Action Recognition Streamlit Dashboard
cd /d "%~dp0.."
echo Starting Medical HAR Dashboard...
echo.
echo   Dashboard: http://localhost:8501
echo   Press Ctrl+C to stop
echo.
.venv\Scripts\python.exe -m streamlit run demo_combined/dashboard_app.py --server.headless true --server.port 8501
