@echo off
setlocal

cd /d "%~dp0"

set ECHONOTES_DATABASE_FALLBACK=sqlite
set ECHONOTES_VECTOR_BACKEND=auto

echo ============================================================
echo EchoNotes AI - Premium Multimodal Workstation Launcher
echo ============================================================

echo [*] Releasing old EchoNotes ports if they are already running...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo     Stopping old backend process PID %%a
    taskkill /PID %%a /F >nul 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo     Stopping old frontend process PID %%a
    taskkill /PID %%a /F >nul 2>nul
)
ping 127.0.0.1 -n 3 >nul

echo [*] Starting local PostgreSQL project database if Docker is available...
where docker >nul 2>nul
if not errorlevel 1 (
    docker compose up -d postgres >nul 2>nul
    if errorlevel 1 (
        echo [WARN] Could not start PostgreSQL automatically. Project Library will show database offline.
    ) else (
        echo [OK] PostgreSQL container requested.
    )
) else (
    echo [WARN] Docker was not found on PATH. Project Library database will be offline.
)

:: 1. Verify python virtual environment
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found.
    echo Please run setup_and_run.bat first to initialize dependencies.
    pause
    exit /b 1
)

:: 2. Install backend dependencies (FastAPI, Uvicorn, Multipart) if not installed
echo [*] Syncing backend dependencies...
"venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Failed to sync Python dependencies automatically. 
    echo Please run: venv\Scripts\pip.exe install -r requirements.txt manually.
)
if /i "%ECHONOTES_INSTALL_PROD_DEPS%"=="1" (
    echo [*] Installing optional production dependencies...
    "venv\Scripts\pip.exe" install -r requirements-prod.txt
)

:: 3. Verify node modules in frontend
if not exist "frontend\node_modules" (
    echo [!] Node modules not found for React frontend.
    echo [*] Running 'npm install' inside frontend directory...
    
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] 'npm' was not found on PATH.
        echo Please install Node.js with npm from https://nodejs.org/
        echo after installing Node.js, run this batch script again.
        pause
        exit /b 1
    )
    
    cd frontend
    call npm install
    cd ..
    
    if errorlevel 1 (
        echo [ERROR] Failed to run 'npm install'. Please configure npm manually.
        pause
        exit /b 1
    )
)

:: 4. Start FastAPI backend (Port 8000)
echo [*] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "EchoNotes API Backend" cmd /k "venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000 --host 127.0.0.1 1> backend.out.log 2> backend.err.log"

:: 5. Start React Dev Server (Port 5173)
echo [*] Starting React Cinematic Workspace on http://localhost:5173 ...
start "EchoNotes Workstation Frontend" cmd /k "npm.cmd run dev --prefix frontend -- --host 127.0.0.1 --port 5173 --strictPort 1> frontend.out.log 2> frontend.err.log"

echo [*] Waiting for services to boot...
ping 127.0.0.1 -n 6 >nul

echo [*] Checking backend...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/status -TimeoutSec 5; Write-Host '[OK] Backend responded:' $r.StatusCode } catch { Write-Host '[WARN] Backend not ready yet. Check backend.err.log'; }"

echo [*] Checking frontend...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173 -TimeoutSec 5; Write-Host '[OK] Frontend responded:' $r.StatusCode } catch { Write-Host '[WARN] Frontend not ready yet. Check frontend.err.log'; }"

echo ============================================================
echo [SUCCESS] Both services are initializing in background windows.
echo.
echo URL Info:
echo - API Backend: http://127.0.0.1:8000/docs (Swagger/OpenAPI docs)
echo - Cinematic Dashboard: http://localhost:5173
echo.
echo Note: Keep this window open. Press Ctrl+C or close the background windows to stop.
echo ============================================================

pause
endlocal
