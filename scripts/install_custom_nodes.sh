#!/usr/bin/env bash
set -euo pipefail

LIST=${1:-nodes.txt}
[ -d "ComfyUI" ] || { echo "ComfyUI not found. Run setup script first."; exit 1; }

NODES_DIR="ComfyUI/custom_nodes"
mkdir -p "$NODES_DIR"

while IFS= read -r line; do
    line="$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')"
    [ -z "$line" ] && continue
    [[ "$line" =~ ^# ]] && continue

    name="$(basename "$line")"
    dest="$NODES_DIR/$name"

    if [ -d "$dest" ]; then
        echo "Skipping (exists): $name"
    else
        git clone "$line" "$dest"
    fi

    if [ -f "$dest/requirements.txt" ]; then
        . .venv/bin/activate
        python -m pip install -r "$dest/requirements.txt"
    fi
done < "$LIST"

echo "Custom nodes installed."
