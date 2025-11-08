#!/usr/bin/env bash
set -euo pipefail

[ -f "./comfyui.env" ] && export $(grep -v '^#' comfyui.env | xargs -d '\n' -I {} echo {} )

PY=${PYTHON_EXE:-"./.venv/bin/python"}
[ -x "$PY" ] || { echo "Python not found at $PY. Run scripts/setup_linux.sh first."; exit 1; }

PORT=${COMFY_PORT:-8188}
HOST=${COMFY_HOST:-127.0.0.1}
LOW=${COMFY_LOW_VRAM:-false}
CPU=${COMFY_FORCE_CPU:-false}
DML=${COMFY_USE_DIRECTML:-false}

FLAGS=( --port "$PORT" --listen "$HOST" )
[ "$LOW" = "true" ] && FLAGS+=( --lowvram )
[ "$CPU" = "true" ] && FLAGS+=( --cpu )
[ "$DML" = "true" ] && FLAGS+=( --directml )

pushd ComfyUI >/dev/null
"$PY" main.py "${FLAGS[@]}"
popd >/dev/null
