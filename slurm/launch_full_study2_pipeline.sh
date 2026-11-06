#!/bin/bash
# Master SLURM pipeline launcher for Study 2 on EPFL Jed HPC.
# Includes automatic multi-tier CPU/RAM fallback and extended walltime limits
# if sbatch rejects a high-resource request or hits QOS limits.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

submit_with_fallback() {
    local script="$1"
    local dep_arg="$2"
    local tiers=("$3" "$4" "$5" "$6")

    for tier in "${tiers[@]}"; do
        read -r cpus mem walltime <<< "$tier"
        local cmd=(sbatch --parsable --ntasks="$cpus" --mem="$mem" --time="$walltime")
        if [ -n "$dep_arg" ]; then
            cmd+=(--dependency="$dep_arg")
        fi
        cmd+=("$script")

        local jid
        if jid=$("${cmd[@]}" 2>/dev/null) && [ -n "$jid" ]; then
            echo "$jid"
            return 0
        fi
        echo "  [Fallback] sbatch rejected (${cpus} CPUs, ${mem}, ${walltime}) for $script; stepping down..." >&2
    done

    echo "FATAL: All resource tiers failed for $script" >&2
    return 1
}

echo "Submitting Step 1: Orbit Selection (81 streams) & Plummer HDF5 ICs..."
JOB1=$(submit_with_fallback "slurm/step1_orbits_and_ics.sh" "" \
    "16 64G 12:00:00" \
    "8 32G 18:00:00" \
    "4 16G 24:00:00" \
    "2 8G 24:00:00")
echo "  -> Step 1 Job ID: $JOB1"

echo "Submitting Step 2: SWIFT N-body Simulations Array (links 75 existing + runs 6 new outer-halo clusters)..."
JOB2=$(submit_with_fallback "slurm/step2_run_swift_array.sh" "afterok:$JOB1" \
    "36 180G 120:00:00" \
    "24 120G 144:00:00" \
    "16 80G 168:00:00" \
    "8 48G 168:00:00")
echo "  -> Step 2 Job ID: $JOB2 (depends on $JOB1)"

echo "Submitting Step 3: Exact Bound Labels + Clean & Gaia DR3 Noisy Observables Array (81 clusters)..."
JOB3=$(submit_with_fallback "slurm/step3_extract_observables.sh" "afterok:$JOB2" \
    "18 96G 24:00:00" \
    "12 64G 36:00:00" \
    "8 48G 48:00:00" \
    "4 24G 72:00:00")
echo "  -> Step 3 Job ID: $JOB3 (depends on $JOB2)"

echo "Submitting Step 4: 5D k-NN Graph Construction in /scratch/cozma (Clean & Noisy, without v_los)..."
JOB4=$(submit_with_fallback "slurm/step4_build_graphs.sh" "afterok:$JOB3" \
    "36 180G 24:00:00" \
    "24 120G 36:00:00" \
    "16 80G 48:00:00" \
    "8 48G 72:00:00")
echo "  -> Step 4 Job ID: $JOB4 (depends on $JOB3)"

echo "Submitting Step 5: 5D 2-Stage Curriculum Training (Clean -> Gaia DR3 Noisy, GCN & GAT) + PNG/HTML Figures..."
JOB5=$(submit_with_fallback "slurm/step5_train_curriculum_and_plot.sh" "afterok:$JOB4" \
    "36 220G 168:00:00" \
    "24 160G 168:00:00" \
    "16 110G 168:00:00" \
    "8 64G 168:00:00")
echo "  -> Step 5 Job ID: $JOB5 (depends on $JOB4)"

echo "Submitting Step 6: 4-Observable Ablation Array (5 combinations x Clean -> Gaia DR3 Noisy) + PNG/HTML Figures..."
JOB6=$(submit_with_fallback "slurm/step6_ablation_4obs.sh" "afterok:$JOB3" \
    "36 220G 168:00:00" \
    "24 160G 168:00:00" \
    "16 110G 168:00:00" \
    "8 64G 168:00:00")
echo "  -> Step 6 Job ID: $JOB6 (depends on $JOB3)"

echo ""
echo "Full Study 2 pipeline submitted to SLURM with adaptive CPU/RAM fallback!"
echo "Chain: $JOB1 (ICs) -> $JOB2 (SWIFT) -> $JOB3 (Observables) -> [$JOB4 (5D Graphs) -> $JOB5 (5D Train+Plots) | $JOB6 (5x 4D Ablation Clean+Noisy)]"
echo "Monitor progress with: squeue -u cozma"
