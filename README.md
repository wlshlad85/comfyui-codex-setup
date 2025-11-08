# comfyui-codex-setup

Turn-key repository to **stand up ComfyUI** on Windows 11 or WSL/Linux and orchestrate repeatable tasks via **Claude Code (Codex)** CLI.

> Works offline once models are present. Scripts create an isolated Python virtual environment, clone ComfyUI, install curated custom nodes, and provide **Codex task runners** to automate graph ops (download models, validate env, launch server, batch-render).

---

## Quickstart (Windows 11 PowerShell)

```pwsh
# 1) Clone your new repo
git clone <YOUR_FORK_URL> comfyui-codex-setup
cd comfyui-codex-setup

# 2) One-shot setup (Python 3.10–3.12 recommended)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./scripts/setup_windows.ps1

# 3) Optional: install popular custom nodes
./scripts/install_custom_nodes.ps1

# 4) Launch ComfyUI
./scripts/run_comfyui.ps1
```

ComfyUI will start at http://127.0.0.1:8188 by default.

## Quickstart (WSL / Linux)

```bash
git clone <YOUR_FORK_URL> comfyui-codex-setup
cd comfyui-codex-setup
bash scripts/setup_linux.sh          # create venv + install deps
bash scripts/install_custom_nodes.sh # optional nodes
bash scripts/run_comfyui.sh
```

## Repo Layout

```
comfyui-codex-setup/
├─ scripts/                # setup + run helpers
├─ codex/                  # Claude Code task prompts + recipes
├─ nodes.txt               # curated list of recommended custom nodes
├─ models/PLACEHOLDERS.md  # where to put models (do NOT commit models)
├─ comfyui.env             # port/paths toggles
└─ README.md
```

## Claude Code (Codex) Integration

Requires Claude Code CLI (`claude --version` should work).

Prompts in `codex/` encapsulate common tasks. Examples:

```pwsh
# Validate environment
claude run -p codex/01_env_check.md

# Bootstrap models & folders (non-destructive)
claude run -p codex/02_bootstrap_models.md

# Start ComfyUI (reads ./comfyui.env)
claude run -p codex/03_start_server.md
```

These prompts are plain text; edit them freely to fit your workflow.

## Where to Put Models

See [models/PLACEHOLDERS.md](models/PLACEHOLDERS.md) for exact directory names that ComfyUI expects.

Typical locations (under the ComfyUI root):

```
ComfyUI/models/checkpoints
ComfyUI/models/vae
ComfyUI/models/clip
ComfyUI/models/clip_vision
ComfyUI/models/unet
ComfyUI/models/upscale_models
ComfyUI/models/loras
ComfyUI/models/embeddings
ComfyUI/custom_nodes/<node_name>
```

Large files should not be committed. Consider Git LFS or keep them local.

## Recommended Custom Nodes

The file `nodes.txt` lists well-maintained nodes (render managers, downloaders, SDXL utilities, upscalers, etc.). The install scripts will clone them into `ComfyUI/custom_nodes/`.

## GPU / CUDA Notes

- **On Windows**: ensure the NVIDIA driver + CUDA runtime support your PyTorch build.
- If you see CUDA errors, try the DirectML backend or CPU fallback (`--cpu`) to verify environment.
- Use `comfyui.env` to toggle low-VRAM flags and server port.

## Troubleshooting

- **Blank UI / 404**: confirm the process is running; port not blocked; correct venv is active.
- **VRAM OOM**: enable low-VRAM flags in `comfyui.env`, use smaller checkpoints, disable high-res fix.
- **Custom node import errors**: re-run install_custom_nodes scripts; update submodules; check Python version.

## License

MIT for this scaffold. ComfyUI and each custom node retain their respective licenses. Verify model licenses before use.
