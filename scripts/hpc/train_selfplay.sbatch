#!/usr/bin/env bash
# RL-2a self-play training (single circuit). Submit: sbatch scripts/hpc/train_selfplay.sbatch
#SBATCH --job-name=f1-rl2a
#SBATCH --output=logs/rl2a_%j.out
#SBATCH --error=logs/rl2a_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
# #SBATCH --partition=<your_cpu_partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/rl_league/selfplay

# module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy

export CUDA_VISIBLE_DEVICES=""          # CPU-only; policy is a small MLP
ITERS="${ITERS:-200}"
WORKERS="${WORKERS:-14}"                 # ~cpus-per-task - 2 for Ray rollout workers

echo "▸ RL-2a training: iters=$ITERS workers=$WORKERS"
srun python -m src.rl.train_selfplay --iters "$ITERS" --workers "$WORKERS" --checkpoint-every 20
echo "✓ done"
