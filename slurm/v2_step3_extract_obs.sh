#!/bin/bash
#SBATCH --job-name=Study2V2_Step3_ExtractObs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=06:00:00
#SBATCH --array=0-80%8
#SBATCH --output=slurm_logs/v2_step3_extract_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SIM_DIR="${SIM_DIR:-/scratch/cozma/study2/outputs_unbiased}"
OBS_V2_DIR="${OBS_V2_DIR:-/scratch/cozma/study2_v2/obs_unbiased}"
G_MAX="${G_MAX:-20.7}"
N_ROTATIONS="${N_ROTATIONS:-5}"

mkdir -p slurm_logs "$OBS_V2_DIR"

module load gcc/13.2.0
module load python/3.11.7
export MPLBACKEND=Agg
source env_astro/bin/activate

echo "=================================================================="
echo "Study 2 V2 Step 3: Gaia DR3 Observable Extraction (IC_index=$SLURM_ARRAY_TASK_ID)"
echo "Read-only SWIFT input : $SIM_DIR"
echo "Isolated V2 output    : $OBS_V2_DIR"
echo "Magnitude cut         : G <= $G_MAX | Rotations: $N_ROTATIONS"
echo "=================================================================="

python3 scripts/v2_extract_observables.py \
    --sim_dir "$SIM_DIR" \
    --out_dir "$OBS_V2_DIR" \
    --n_rotations "$N_ROTATIONS" \
    --g_max "$G_MAX" \
    --contaminant_min 0.05 \
    --contaminant_max 0.55 \
    --pure_prob 0.15 \
    --ic_index "$SLURM_ARRAY_TASK_ID" \
    --num_workers "${SLURM_CPUS_PER_TASK:-18}"

echo "[DONE] V2 extraction complete for IC index $SLURM_ARRAY_TASK_ID."
