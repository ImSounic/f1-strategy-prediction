#!/usr/bin/env bash
# RL-3a beat-baselines eval. Submit: sbatch scripts/hpc/eval_rl.sbatch
#SBATCH --job-name=f1-rl3a
#SBATCH --output=logs/rl3a_%j.out
#SBATCH --error=logs/rl3a_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
# #SBATCH --partition=<your_cpu_partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/rl_eval

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f1-strategy

export CUDA_VISIBLE_DEVICES=""
RACES="${RACES:-200}"
CIRCUIT="${CIRCUIT:-bahrain}"

echo "▸ RL-3a eval: races=$RACES circuit=$CIRCUIT"
python -m src.rl.eval.run_beat_baselines --races "$RACES" --circuit "$CIRCUIT"
echo "✓ done"
