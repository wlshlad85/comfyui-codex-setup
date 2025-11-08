param(
    [switch]$ForceReinstall=$false
)

$ErrorActionPreference = "Stop"

# Read env file
$envPath = Join-Path (Get-Location) "comfyui.env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match "^\s*#") { return }
        if ($_ -match "^\s*$") { return }
        $kv = $_.Split("=",2)
        if ($kv.Length -eq 2) { [System.Environment]::SetEnvironmentVariable($kv[0], $kv[1]) }
    }
}

# Ensure git & python are available
function Assert-Cmd($cmd) {
    $null = Get-Command $cmd -ErrorAction SilentlyContinue
    if (-not $?) { throw "Required command not found: $cmd" }
}
Assert-Cmd git
Assert-Cmd python

# Create .venv
if ($ForceReinstall -and (Test-Path ".venv")) { Remove-Item -Recurse -Force ".venv" }
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
$venvPy = Join-Path ".venv" "Scripts/python.exe"
& $venvPy -m pip install --upgrade pip wheel setuptools

# Clone ComfyUI (if not present)
if (-not (Test-Path "ComfyUI")) {
    git clone https://github.com/comfyanonymous/ComfyUI.git
}

# Install ComfyUI requirements
pushd ComfyUI
& ..\$venvPy -m pip install -r requirements.txt
popd

Write-Host "Setup complete. Next: run ./scripts/run_comfyui.ps1"
