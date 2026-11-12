#!/bin/bash
# Master SLURM pipeline launcher for Study 2 on EPFL Jed HPC.
# Respects the user's strict 72-task / 72-CPU total budget and 7000 MB/CPU SLURM limit:
#   - 2 nodes x 36 CPUs/node = 72 CPUs total per active job
#   - 246 GB RAM per node (36 * 6.83 GB <= 7000 MB/CPU) = 492 GB total RAM across the 2 nodes!
#   - Starts IMMEDIATELY (0 queue wait) on the 15+ half-free `mix` nodes (36/36 cores free) on `academic`.
#   - Step 5 (2 nodes x 36 CPUs, 246G/node = 72 CPUs, 492G total): Full 81-Cluster 5-Observable Curriculum Training
#   - Step 6 (Array 0-4%1, 2 nodes x 36 CPUs, 246G/node = 72 CPUs, 492G total, after Step 5):
#       4-Observable Ablations initialized from the trained 5-Observable Clean weights -> 4D Clean -> 4D Gaia Noisy.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

submit_multinode_with_fallback() {
    local script="$1"
    local dep_arg="$2"
    local extra_arg="$3"
    shift 3
    local tiers=("$@")

    for tier in "${tiers[@]}"; do
        read -r nodes cpus mem walltime <<< "$tier"
        local cmd=(
            sbatch --parsable
            --nodes="$nodes"
            --ntasks-per-node=1
            --cpus-per-task="$cpus"
            --mem="$mem"
            --time="$walltime"
        )
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
        echo "  [Fallback] sbatch rejected (${nodes} nodes, ${cpus} CPUs/node, ${mem}/node, ${walltime}) for $script; stepping down..." >&2
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

echo "=== STEP 5: FULL 81-CLUSTER 5-OBSERVABLE CURRICULUM TRAINING (2 NODES x 36 CORES = 72 CPUS TOTAL, 246G/NODE = 492G TOTAL) ==="
JOB5=$(submit_multinode_with_fallback "slurm/step5_train_curriculum_and_plot.sh" "" "--job-name=Study2_Step5_Train81" \
    "2 36 246G 168:00:00" \
    "3 24 164G 168:00:00" \
    "4 18 123G 168:00:00" \
    "1 36 246G 168:00:00")
echo "  -> Step 5 (5-Obs Clean -> Gaia Noisy, 81 streams) Job ID: $JOB5"

echo "=== STEP 6: 4-OBSERVABLE ABLATION ARRAY (2 NODES x 36 CORES = 72 CPUS TOTAL, 246G/NODE = 492G TOTAL, DEPENDS ON STEP 5) ==="
JOB6=$(submit_multinode_with_fallback "slurm/step6_ablation_4obs.sh" "afterok:$JOB5" "--array=0-4%1" \
    "2 36 246G 168:00:00" \
    "3 24 164G 168:00:00" \
    "4 18 123G 168:00:00" \
    "1 36 246G 168:00:00")
echo "  -> Step 6 (4-Obs Ablation initialized from 5-Obs weights) Job ID: $JOB6 (depends on $JOB5)"

echo "=== STEP 7: GCN RETRAINING WITH STOCHASTIC SO(3) ROTATIONS (AFTER ALL STEPS, DEPENDS ON STEP 6) ==="
JOB7=$(submit_multinode_with_fallback "slurm/step7_retrain_gcn_stochastic.sh" "afterok:$JOB6" "--job-name=Study2_Step7_GCNStochastic" \
    "2 36 246G 24:00:00" \
    "3 24 164G 24:00:00" \
    "4 18 123G 24:00:00" \
    "1 36 246G 24:00:00")
echo "  -> Step 7 (GCN Clean Retraining with stochastic rotations) Job ID: $JOB7 (depends on $JOB6)"

echo ""
echo "================================================================================"
echo "STUDY 2 72-TASK BUDGET (2 NODES x 36 CORES = 72 CPUS, 492G TOTAL) SUBMITTED!"
echo "  * Step 5 (5-Obs Clean -> Noisy GAT, 81 clusters, 2 nodes x 36 CPUs / 246G): $JOB5"
echo "  * Step 6 (4-Obs Ablations x5, 2 nodes x 36 CPUs / 246G, after $JOB5): $JOB6"
echo "  * Step 7 (GCN Baseline Retraining, 2 nodes x 36 CPUs / 246G, after $JOB6): $JOB7"
echo "================================================================================"
