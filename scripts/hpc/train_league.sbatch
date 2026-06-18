#!/usr/bin/env bash
# RL-2b AlphaStar league training. Submit: sbatch scripts/hpc/train_league.sbatch
#SBATCH --job-name=f1-rl2b
#SBATCH --output=logs/rl2b_%j.out
#SBATCH --error=logs/rl2b_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# #SBATCH --partition=<your_cpu_partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/rl_league/league

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy

export CUDA_VISIBLE_DEVICES=""          # CPU-only; small MLP policies
ITERS="${ITERS:-300}"                   # ~250s/iter -> 300 iters ~21h (fits 24h wall)
WORKERS="${WORKERS:-14}"
# To resume from a prior checkpoint's learner weights:
#   RESTORE=models/rl_league/league sbatch scripts/hpc/train_league.sbatch
RESTORE="${RESTORE:-}"
RESTORE_ARG=""
if [ -n "$RESTORE" ]; then RESTORE_ARG="--restore $RESTORE"; fi

echo "▸ RL-2b league: iters=$ITERS workers=$WORKERS restore='${RESTORE}'"
srun python -m src.rl.train_league --iters "$ITERS" --workers "$WORKERS" \
     --checkpoint-every 20 $RESTORE_ARG
echo "✓ done"
