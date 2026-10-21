# Start the backend (FastAPI on :8000) and frontend (Vite HTTPS on :5173).
# Open https://localhost:5173 on this PC, or https://<this-PC's-IP>:5173 on your phone
# (same Wi-Fi). Accept the self-signed certificate warning once.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# winget-installed tools may not be on PATH in shells opened before installation
$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
$env:PYTHONUTF8 = "1"

$backend = Start-Process -PassThru -NoNewWindow -WorkingDirectory "$root\backend" `
    -FilePath "$root\backend\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"

try {
    Set-Location "$root\frontend"
    npm run dev
} finally {
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
}
