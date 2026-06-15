#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  SLURM array job — train the 24 PPO pit-stop agents
# ════════════════════════════════════════════════════════════════════
# One circuit per array task. The "%2" caps concurrency at 2 running
# tasks at a time (your HPC limit). 24 circuits / 2 ≈ 12 waves.
#
# Submit:   sbatch scripts/hpc/train_rl.sbatch
# Watch:    squeue --me
# Logs:     logs/rl_train_<arrayjobid>_<taskid>.out
#
# RL here is CPU-bound (tiny MLP policy; env stepping dominates), so we
# request CPU cores, not a GPU. Add a --gres=gpu:1 line only if you
# later switch to a heavy/recurrent policy that actually uses the GPU.
#
# ── SLURM directives (adjust partition/account to your cluster) ──────
#SBATCH --job-name=f1-rl
#SBATCH --output=logs/rl_train_%A_%a.out
#SBATCH --error=logs/rl_train_%A_%a.err
#SBATCH --array=0-23%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# #SBATCH --partition=<your_cpu_partition>
# #SBATCH --account=<your_account>

set -euo pipefail

# Circuits — index matches SLURM_ARRAY_TASK_ID (0..23).
CIRCUITS=(
  albert_park bahrain baku barcelona cota hungaroring imola interlagos
  jeddah las_vegas lusail mexico miami monaco montreal monza
  shanghai silverstone singapore spa spielberg suzuka yas_marina zandvoort
)

CIRCUIT="${CIRCUITS[$SLURM_ARRAY_TASK_ID]}"
SEASON="${SEASON:-2025}"
TIMESTEPS="${TIMESTEPS:-500000}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load your cluster's Python module if needed, then the project venv.
# module load python/3.10
source venv/bin/activate

echo "▸ Task $SLURM_ARRAY_TASK_ID → circuit=$CIRCUIT season=$SEASON timesteps=$TIMESTEPS"
echo "▸ Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK:-?}"

# Keep torch on CPU and pin thread count to the allocated cores.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

srun python -m src.rl.train \
  --circuit "$CIRCUIT" \
  --season  "$SEASON" \
  --timesteps "$TIMESTEPS"

echo "✓ Done: $CIRCUIT"
