#!/usr/bin/env bash
# launch.sh -- open a lecture notebook in marimo.
#   bash launch.sh 0b    # opens notebooks/0b_execution_model.py
#   bash launch.sh       # opens the home/index notebook
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nbdir="$root/notebooks"
if [[ -z "${1:-}" ]]; then
  target="$nbdir/home.py"
else
  target="$(ls "$nbdir/$1"*.py 2>/dev/null | head -n1 || true)"
  [[ -z "$target" ]] && { echo "No notebook matches '$1' in $nbdir" >&2; exit 1; }
fi
echo "marimo edit $target"
marimo edit "$target"
