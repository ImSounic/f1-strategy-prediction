#!/usr/bin/env bash
# RL-2b AlphaStar league training. Submit: sbatch scripts/hpc/train_league.sbatch
#SBATCH --job-name=f1-rl2b
#SBATCH --output=logs/rl2b_%j.out
#SBATCH --error=logs/rl2b_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# #SBATCH --partition=<your_cpu_partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/rl_league/league

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy

export CUDA_VISIBLE_DEVICES=""          # CPU-only; small MLP policies
ITERS="${ITERS:-300}"
WORKERS="${WORKERS:-14}"

echo "▸ RL-2b league: iters=$ITERS workers=$WORKERS (6 learners — expect ~6x RL-2a per-iter)"
srun python -m src.rl.train_league --iters "$ITERS" --workers "$WORKERS" --checkpoint-every 20
echo "✓ done"
