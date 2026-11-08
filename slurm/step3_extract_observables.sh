#!/bin/bash
#SBATCH --job-name=Study2_Step3_Obs
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --mem=96G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=24:00:00
#SBATCH --array=0-80%8
#SBATCH --output=slurm_logs/step3_obs_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

module load gcc/13.2.0
module load python/3.11.7

export MPLBACKEND=Agg
source env_astro/bin/activate

SIM_DIR="${SIM_DIR:-/scratch/cozma/study2/outputs_unbiased}"
N_WORKERS="${SLURM_NTASKS:-18}"

echo "Extracting exact bound labels + Clean & Gaia DR3 Noisy observables for IC index $SLURM_ARRAY_TASK_ID ($N_WORKERS workers)..."
python3 scripts/extract_observables.py \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean" \
    --obs_dir "/scratch/cozma/study2/obs_clean" \
    --obs_noisy_dir "/scratch/cozma/study2/obs_gaia_noisy" \
    --n_rotations 5 \
    --norm_mode zscore \
    --abs_mag_g 4.5 \
    --ic_index "$SLURM_ARRAY_TASK_ID" \
    --num_workers "$N_WORKERS"

echo "Completed observable extraction for task $SLURM_ARRAY_TASK_ID."
if [ $((SLURM_ARRAY_TASK_ID % 10)) -eq 0 ] || [ "$SLURM_ARRAY_TASK_ID" -eq 80 ]; then
    python3 scripts/live_monitor.py --once || true
fi
