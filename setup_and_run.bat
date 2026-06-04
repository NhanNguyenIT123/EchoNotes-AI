@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo EchoNotes AI - Setup and Run
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10+ first, then run this file again.
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Upgrading pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

echo [3/4] Installing project dependencies...
"venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [4/4] Starting EchoNotes AI at http://localhost:8501
echo Keep this window open while using the app.
echo ============================================================
"venv\Scripts\python.exe" -m streamlit run app\ui.py --server.port 8501 --server.address localhost --server.maxUploadSize 2048

endlocal
