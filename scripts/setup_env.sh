#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  Environment setup — JupyterLab and HPC
# ════════════════════════════════════════════════════════════════════
# Usage:
#   bash scripts/setup_env.sh           # core deps only
#   bash scripts/setup_env.sh --rl      # core + reinforcement-learning stack
#   bash scripts/setup_env.sh --rl --cpu-torch   # force CPU-only torch wheel
#
# Creates a local venv at ./venv and installs requirements into it.
# On an HPC login node you may first need:  module load python/3.10
set -euo pipefail

WITH_RL=0
CPU_TORCH=0
for arg in "$@"; do
  case "$arg" in
    --rl)        WITH_RL=1 ;;
    --cpu-torch) CPU_TORCH=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# Resolve repo root (this script lives in scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
echo "▸ Using interpreter: $("$PY" --version 2>&1) ($("$PY" -c 'import sys;print(sys.executable)'))"

if [ ! -d venv ]; then
  echo "▸ Creating venv at ./venv"
  "$PY" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip wheel

echo "▸ Installing core requirements"
pip install -r requirements.txt

if [ "$WITH_RL" -eq 1 ]; then
  if [ "$CPU_TORCH" -eq 1 ]; then
    echo "▸ Installing CPU-only torch first"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  echo "▸ Installing reinforcement-learning stack"
  pip install -r requirements-rl.txt
fi

echo ""
echo "✓ Environment ready. Activate with:  source venv/bin/activate"
echo "  Next:  python scripts/verify_setup.py"
