#!/bin/bash
# Master SLURM pipeline launcher for Study 2 on EPFL Jed HPC.
# Parallelizes:
#   - Branch A (Immediate, 0 wait): Step 3 (75 ready clusters) -> Step 4A (5D Graphs) -> Step 5A (Live GNN Training & Plots)
#   - Branch B (Parallel N-body):   Step 2 (6 new outer-halo SWIFT runs) -> Step 3B (6 clusters) -> Step 4B/5B/6 (Full 81-stream completion & 4D ablations)

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

submit_with_fallback() {
    local script="$1"
    local dep_arg="$2"
    local extra_arg="$3"
    local tiers=("$4" "$5" "$6" "$7")

    for tier in "${tiers[@]}"; do
        read -r cpus mem walltime <<< "$tier"
        local cmd=(sbatch --parsable --ntasks="$cpus" --mem="$mem" --time="$walltime")
        if [ -n "$dep_arg" ]; then
            cmd+=(--dependency="$dep_arg")
        fi
        if [ -n "$extra_arg" ]; then
            cmd+=("$extra_arg")
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

echo "Canceling any stale/blocked Study2_* SLURM jobs..."
STALE_JOBS=$(squeue -u cozma -h -o "%A %j" | awk '/^([0-9]+) Study2_/ {print $1}' | sort -u || true)
if [ -n "$STALE_JOBS" ]; then
    scancel $STALE_JOBS || true
    echo "  -> Canceled stale jobs: $STALE_JOBS"
fi

ARRAY_75="--array=0,2-7,9-24,26-42,44-46,48-61,63-80%8"
ARRAY_6_SWIFT="--array=1,8,25,43,47,62%6"
ARRAY_6_OBS="--array=1,8,25,43,47,62"

echo "=== BRANCH A (IMMEDIATE START ON 75 COMPLETED CLUSTERS) ==="
echo "Submitting Step 3A: Bound Labels + Clean & Gaia DR3 Noisy Observables (75 ready clusters, starts immediately)..."
JOB3A=$(submit_with_fallback "slurm/step3_extract_observables.sh" "" "$ARRAY_75" \
    "18 96G 24:00:00" \
    "12 64G 36:00:00" \
    "8 48G 48:00:00" \
    "4 24G 72:00:00")
echo "  -> Step 3A Job ID: $JOB3A (no dependency — starts immediately)"

echo "Submitting Step 4A: 5D k-NN Graph Construction for 75 clusters in /scratch/cozma..."
JOB4A=$(submit_with_fallback "slurm/step4_build_graphs.sh" "afterok:$JOB3A" "--job-name=Study2_Step4A_Graphs75" \
    "36 180G 24:00:00" \
    "24 120G 36:00:00" \
    "16 80G 48:00:00" \
    "8 48G 72:00:00")
echo "  -> Step 4A Job ID: $JOB4A (depends on $JOB3A)"

echo "Submitting Step 5A: 5D 2-Stage Curriculum Training (GCN & GAT) + Real-Time Live Plots on 75 clusters..."
JOB5A=$(submit_with_fallback "slurm/step5_train_curriculum_and_plot.sh" "afterok:$JOB4A" "--job-name=Study2_Step5A_Train75" \
    "36 220G 168:00:00" \
    "24 160G 168:00:00" \
    "16 110G 168:00:00" \
    "8 64G 168:00:00")
echo "  -> Step 5A Job ID: $JOB5A (depends on $JOB4A)"

echo ""
echo "=== BRANCH B (PARALLEL SWIFT SIMULATION OF 6 OUTER-HALO CLUSTERS + 81-STREAM COMPLETION) ==="
echo "Submitting Step 2B: SWIFT N-body Simulations for 6 outer-halo clusters (IC_2,10,27,45,49,67, starts immediately)..."
JOB2B=$(submit_with_fallback "slurm/step2_run_swift_array.sh" "" "$ARRAY_6_SWIFT" \
    "36 180G 120:00:00" \
    "24 120G 144:00:00" \
    "16 80G 168:00:00" \
    "8 48G 168:00:00")
echo "  -> Step 2B Job ID: $JOB2B (no dependency — starts immediately)"

echo "Submitting Step 3B: Bound Labels + Clean & Gaia DR3 Noisy Observables for the 6 outer-halo clusters..."
JOB3B=$(submit_with_fallback "slurm/step3_extract_observables.sh" "afterok:$JOB2B" "$ARRAY_6_OBS" \
    "18 96G 24:00:00" \
    "12 64G 36:00:00" \
    "8 48G 48:00:00" \
    "4 24G 72:00:00")
echo "  -> Step 3B Job ID: $JOB3B (depends on $JOB2B)"

echo "Submitting Step 4B: Incremental 5D k-NN Graph Completion (all 81 clusters)..."
JOB4B=$(submit_with_fallback "slurm/step4_build_graphs.sh" "afterok:$JOB3A:$JOB3B:$JOB4A" "--job-name=Study2_Step4B_Graphs81" \
    "36 180G 24:00:00" \
    "24 120G 36:00:00" \
    "16 80G 48:00:00" \
    "8 48G 72:00:00")
echo "  -> Step 4B Job ID: $JOB4B (depends on $JOB3A, $JOB3B, $JOB4A)"

echo "Submitting Step 5B: Full 81-Cluster 2-Stage Curriculum Training (GCN & GAT) + Final Figures..."
JOB5B=$(submit_with_fallback "slurm/step5_train_curriculum_and_plot.sh" "afterok:$JOB4B:$JOB5A" "--job-name=Study2_Step5B_Train81" \
    "36 220G 168:00:00" \
    "24 160G 168:00:00" \
    "16 110G 168:00:00" \
    "8 64G 168:00:00")
echo "  -> Step 5B Job ID: $JOB5B (depends on $JOB4B, $JOB5A)"

echo "Submitting Step 6: 4-Observable Ablation Array (5 combinations x Clean -> Gaia DR3 Noisy)..."
JOB6=$(submit_with_fallback "slurm/step6_ablation_4obs.sh" "afterok:$JOB3A:$JOB3B" "" \
    "36 220G 168:00:00" \
    "24 160G 168:00:00" \
    "16 110G 168:00:00" \
    "8 64G 168:00:00")
echo "  -> Step 6 Job ID: $JOB6 (depends on $JOB3A, $JOB3B)"

echo ""
echo "Full parallel Study 2 pipeline submitted to SLURM!"
echo "  * Immediate Fast-Track (75 clusters): $JOB3A (Obs) -> $JOB4A (5D Graphs) -> $JOB5A (5D Train + Live Plots)"
echo "  * Parallel Outer-Halo (6 clusters)  : $JOB2B (SWIFT) -> $JOB3B (Obs) -> $JOB4B (81 Graphs) -> $JOB5B (81 Train) + $JOB6 (4D Ablation)"
