#!/usr/bin/env bash
# RL-2b league eval monitor (2nd parallel job). Submit: sbatch scripts/hpc/eval_league.sbatch
# Build-only smoke for now; full main-vs-anchors win-rate eval lands in RL-3.
#SBATCH --job-name=f1-rl2b-eval
#SBATCH --output=logs/rl2b_eval_%j.out
#SBATCH --error=logs/rl2b_eval_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# #SBATCH --partition=<your_cpu_partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy

export CUDA_VISIBLE_DEVICES=""
echo "▸ RL-2b eval stub (build-only; full eval is RL-3)"
python -m src.rl.train_league --iters 0 --workers 1
echo "✓ eval stub ok"
