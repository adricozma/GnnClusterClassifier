#!/bin/bash
# Master SLURM pipeline launcher for Study 2 on EPFL Jed HPC.
# Chains Steps 1 -> 6 automatically using SLURM job dependencies (afterok).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

echo "Submitting Step 1: Orbit Selection (all 81 streams inside box, R_p >= 0) & Plummer HDF5 ICs..."
JOB1=$(sbatch --parsable slurm/step1_orbits_and_ics.sh)
echo "  -> Step 1 Job ID: $JOB1"

echo "Submitting Step 2: SWIFT N-body Simulations Array (links 75 existing + runs 6 new outer-halo clusters)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/step2_run_swift_array.sh)
echo "  -> Step 2 Job ID: $JOB2 (depends on $JOB1)"

echo "Submitting Step 3: Exact Bound Labels + Clean & Gaia DR3 Noisy Observables Array (81 clusters)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/step3_extract_observables.sh)
echo "  -> Step 3 Job ID: $JOB3 (depends on $JOB2)"

echo "Submitting Step 4: 5D k-NN Graph Construction in /scratch/cozma (Clean & Noisy, without v_los)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/step4_build_graphs.sh)
echo "  -> Step 4 Job ID: $JOB4 (depends on $JOB3)"

echo "Submitting Step 5: 5D 2-Stage Curriculum Training (Clean -> Gaia DR3 Noisy, GCN & GAT) + PNG/HTML Figures..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 slurm/step5_train_curriculum_and_plot.sh)
echo "  -> Step 5 Job ID: $JOB5 (depends on $JOB4)"

echo "Submitting Step 6: 4-Observable Ablation Array (5 combinations x Clean -> Gaia DR3 Noisy) + PNG/HTML Figures..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/step6_ablation_4obs.sh)
echo "  -> Step 6 Job ID: $JOB6 (depends on $JOB3)"

echo ""
echo "Full Study 2 pipeline submitted to SLURM!"
echo "Chain: $JOB1 (ICs) -> $JOB2 (SWIFT) -> $JOB3 (Observables) -> [$JOB4 (5D Graphs) -> $JOB5 (5D Train+Plots) | $JOB6 (5x 4D Ablation Clean+Noisy)]"
echo "Monitor progress with: squeue -u cozma"
