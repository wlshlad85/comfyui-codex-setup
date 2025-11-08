$ErrorActionPreference = "Stop"

# Load env
$envPath = Join-Path (Get-Location) "comfyui.env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match "^\s*#") { return }
        if ($_ -match "^\s*$") { return }
        $kv = $_.Split("=",2)
        if ($kv.Length -eq 2) { [System.Environment]::SetEnvironmentVariable($kv[0], $kv[1]) }
    }
}

$py = $env:PYTHON_EXE
if ([string]::IsNullOrWhiteSpace($py)) {
    $py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
}
# Resolve to absolute path
$py = (Resolve-Path $py).Path
if (-not (Test-Path $py)) {
    throw "Python not found at $py. Run scripts/setup_windows.ps1 first."
}

# Build args
$port = $env:COMFY_PORT
$listenHost = $env:COMFY_HOST
$low = $env:COMFY_LOW_VRAM
$cpu = $env:COMFY_FORCE_CPU
$dml = $env:COMFY_USE_DIRECTML

$flags = @("--port", $port, "--listen", $listenHost)
if ($low -eq "true") { $flags += @("--lowvram") }
if ($cpu -eq "true") { $flags += @("--cpu") }
if ($dml -eq "true") { $flags += @("--directml") }

Write-Host "Launching ComfyUI at http://$($listenHost):$port ..."
pushd ComfyUI
& $py main.py @flags
popd
