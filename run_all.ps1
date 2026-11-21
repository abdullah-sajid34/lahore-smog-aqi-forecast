# run_all.ps1 -- Windows one-click: venv++tests+pipeline+servers
# Usage: .\run_all.ps1
# Requires: Python 3.10+ and Node.js 18+ on PATH.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

# Step 1: Create virtual environment if missing
Write-Host '[1/5] Checking virtual environment...'
\$Venv = Join-Path \$Root '.venv'
if (-not (Test-Path \$Venv)) {
    python -m venv \$Venv
}

# Step 2: Activate and install dependencies
Write-Host '[2/5] Installing Python dependencies...'
\$Activate = Join-Path \$Venv 'Scripts\Activate.ps1'
. \$Activate
pip install -r (Join-Path \$Root 'requirements.txt') --quiet

# Step 3: Run tests
Write-Host '[3/5] Running pytest...'
pytest -q
if (\$LASTEXITCODE -ne 0) {
    Write-Host 'Tests FAILED -- fix before running pipeline.' -ForegroundColor Red
    exit 1
}
Write-Host '  Tests passed.' -ForegroundColor Green

# Step 4: Run the ML pipeline
Write-Host '[4/5] Running ML pipeline (~3-5 min)...'
python (Join-Path \$Root 'src\run_experiments.py')
if (\$LASTEXITCODE -ne 0) {
    Write-Host 'Pipeline failed.' -ForegroundColor Red
    exit 1
}
Write-Host '  Pipeline complete.' -ForegroundColor Green

# Step 5: Launch backend and frontend in separate windows
Write-Host '[5/5] Launching services...'
\$BackendCmd = "cd '\$Root'; .venv\Scripts\Activate.ps1; python -m uvicorn backend.main:app"
Start-Process powershell -ArgumentList '-NoExit', '-Command', \$BackendCmd

\$FrontendCmd = "cd '\$Root\frontend'; npm run dev"
Start-Process powershell -ArgumentList '-NoExit', '-Command', \$FrontendCmd

Write-Host ''
Write-Host 'All done!' -ForegroundColor Cyan
The backend will start on http://127.0.0.1:8000
The frontend will start on http://127.0.0.1:5173