#!/bin/bash
#SBATCH --job-name=Study2_Step1_ICs
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/step1_ics_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

module load gcc/13.2.0
module load python/3.11.7

export MPLBACKEND=Agg
source env_astro/bin/activate

N_WORKERS="${SLURM_NTASKS:-16}"

echo "=== STEP 1A: Selecting all physical progenitor orbits inside SWIFT box (R_p >= 0.0 kpc, |xyz| <= 480 kpc) ==="
python3 make_ICs/select_unbiased_orbits.py \
    --min_perigalacticon 0.0 \
    --max_box_coord 480.0

echo "=== STEP 1B: Generating virialized Plummer HDF5 initial conditions (linking existing Study 1 ICs) ==="
python3 make_ICs/generate_unbiased_ics.py \
    --nb_threads "$N_WORKERS" \
    --seed 42

echo "=== STEP 1 COMPLETED SUCCESSFULLY ==="
python3 scripts/live_monitor.py --once || true
