@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Run setup_and_run.bat first.
    pause
    exit /b 1
)

echo Starting EchoNotes AI at http://localhost:8501
"venv\Scripts\python.exe" -m streamlit run app\ui.py --server.port 8501 --server.address localhost --server.maxUploadSize 2048

endlocal
