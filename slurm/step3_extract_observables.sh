#!/bin/bash
#SBATCH --job-name=Study2_Step3_Obs
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --mem=450G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=06:00:00
#SBATCH --array=0-80
#SBATCH --output=slurm_logs/step3_obs_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

module load gcc/13.2.0
module load python/3.11.7
module load py-numpy/1.26.4
module load py-scipy/1.13.0
module load py-matplotlib/3.3.4
module load py-h5py/3.11.0

export MPLBACKEND=Agg
source env_astro/bin/activate

SIM_DIR="${SIM_DIR:-/home/cozma/swift_sims/outputs_unbiased}"

echo "Extracting exact bound labels + Clean & Gaia DR3 Noisy observables for IC index $SLURM_ARRAY_TASK_ID..."
python3 scripts/extract_observables.py \
    --sim_dir "$SIM_DIR" \
    --labels_dir "GNN-jed/labels_clean" \
    --obs_dir "GNN-jed/obs_clean" \
    --obs_noisy_dir "GNN-jed/obs_gaia_noisy" \
    --n_rotations 5 \
    --norm_mode zscore \
    --abs_mag_g 4.5 \
    --ic_index "$SLURM_ARRAY_TASK_ID" \
    --num_workers "$SLURM_NTASKS"

echo "Completed observable extraction for task $SLURM_ARRAY_TASK_ID."
