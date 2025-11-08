param(
    [string]$ListPath = "nodes.txt"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path "ComfyUI")) { throw "ComfyUI folder not found. Run setup script first." }
$nodesDir = "ComfyUI\custom_nodes"
New-Item -ItemType Directory -Force -Path $nodesDir | Out-Null

Get-Content $ListPath | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $name = ($line.Split("/")[-1])
    $dest = Join-Path $nodesDir $name
    if (Test-Path $dest) {
        Write-Host "Skipping (already exists): $name"
    } else {
        git clone $line $dest
    }
}

Write-Host "Done. Installing node requirements..."
Get-ChildItem -Path $nodesDir -Directory | ForEach-Object {
    $req = Join-Path $_.FullName "requirements.txt"
    if (Test-Path $req) {
        & .\.venv\Scripts\python.exe -m pip install -r $req
    }
}
