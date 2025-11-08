You are Claude Code. Start ComfyUI server using the repo scripts and env toggles.

**Steps:**

1. Read `./comfyui.env` to determine host/port and flags.
2. If Windows, run: `./scripts/run_comfyui.ps1`
3. If Linux/WSL, run: `bash scripts/run_comfyui.sh`
4. After launch, verify the process is bound to host:port and print the URL.
5. If the port is busy, suggest an alternate port and how to update `comfyui.env`.
