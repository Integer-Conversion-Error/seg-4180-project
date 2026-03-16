@echo off
echo Stopping any existing ParkOpticon Web UI instances...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing PID %%a on port 8000
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo Starting ParkOpticon Web UI...
cd /d "%~dp0"
.\venv\Scripts\python.exe -c "from web_ui.app import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"
