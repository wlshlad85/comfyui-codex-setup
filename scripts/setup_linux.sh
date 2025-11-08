#!/usr/bin/env bash
set -euo pipefail

# Create venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Clone ComfyUI
if [ ! -d "ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

# Install deps
cd ComfyUI
python -m pip install -r requirements.txt
cd ..

echo "Setup complete. Next: bash scripts/run_comfyui.sh"
