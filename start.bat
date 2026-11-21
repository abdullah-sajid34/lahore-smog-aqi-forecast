@echo off
cd /d "%~dp0"

if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt -q

start "Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && uvicorn backend.main:app --port 8000"

cd frontend
if not exist node_modules call npm install
start "Frontend" cmd /k "npm run dev"

timeout /t 10 >nul
start http://localhost:5173