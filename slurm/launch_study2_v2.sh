#!/bin/bash
# Automated SLURM dependency launcher for Study 2 V2 (Gaia DR3 adaptation).
#
# Automatically detects currently running or pending Study 2 V1 jobs
# (Study2_Step5_Train81, Study2_Step6_4Obs, Study2_Step7_GCNStochastic) in `squeue`
# and queues V2 Steps 3 -> 4 -> 5 -> 6 to start immediately once V1 finishes.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
NOW_MODE=0
DEP_STEP5=""
DEP_STEP6=""
DEP_STEP7=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --now)
            NOW_MODE=1
            shift
            ;;
        --dep5)
            DEP_STEP5="$2"
            shift 2
            ;;
        --dep6)
            DEP_STEP6="$2"
            shift 2
            ;;
        --dep7)
            DEP_STEP7="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p slurm_logs /scratch/cozma/study2_v2 GNN-jed/trained_models/study2_v2
fi

_find_active_job() {
    local job_name="$1"
    squeue -u "$USER" -h -o "%A %j %t" 2>/dev/null \
        | awk -v jn="$job_name" '$2 == jn {print $1}' \
        | sed 's/_.*//' \
        | sort -u \
        | tail -n 1
}

if [ "$NOW_MODE" -eq 0 ]; then
    [ -z "$DEP_STEP5" ] && DEP_STEP5="$(_find_active_job Study2_Step5_Train81 || true)"
    [ -z "$DEP_STEP6" ] && DEP_STEP6="$(_find_active_job Study2_Step6_4Obs || true)"
    [ -z "$DEP_STEP7" ] && DEP_STEP7="$(_find_active_job Study2_Step7_GCNStochastic || true)"
fi

ROOT_DEPS=()
[ -n "$DEP_STEP5" ] && ROOT_DEPS+=("afterok:${DEP_STEP5}")
[ -n "$DEP_STEP6" ] && ROOT_DEPS+=("afterok:${DEP_STEP6}")
[ -n "$DEP_STEP7" ] && ROOT_DEPS+=("afterany:${DEP_STEP7}")

DEP_ARG=""
if [ "${#ROOT_DEPS[@]}" -gt 0 ]; then
    DEP_STR=$(IFS=,; echo "${ROOT_DEPS[*]}")
    DEP_ARG="--dependency=${DEP_STR}"
fi

echo "=================================================================="
echo "Study 2 V2 (Gaia DR3 Adaptation) Automated SLURM Chain Launcher"
echo "Active V1 jobs detected:"
echo "  Step 5 (Clean 5D GAT)      : ${DEP_STEP5:-none (already finished)}"
echo "  Step 6 (4D Ablations x5)   : ${DEP_STEP6:-none (already finished)}"
echo "  Step 7 (Stochastic GCN)    : ${DEP_STEP7:-none (already finished)}"
echo "Root V2 Step 3 dependency    : ${DEP_ARG:-none (immediate start)}"
echo "=================================================================="

_submit() {
    local extra_flag="$1"
    local script="$2"
    local cmd=(sbatch --parsable)
    if [ -n "$extra_flag" ]; then
        cmd+=("$extra_flag")
    fi
    cmd+=("$script")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] ${cmd[*]}" >&2
        echo "MOCK_JOB_$(basename "$script" .sh)"
        return 0
    fi
    "${cmd[@]}"
}

JOB_V2_STEP3=$(_submit "$DEP_ARG" slurm/v2_step3_extract_obs.sh)
echo "Queued V2 Step 3 (Extract Gaia V2 Obs) : $JOB_V2_STEP3"

JOB_V2_STEP4=$(_submit "--dependency=afterok:${JOB_V2_STEP3}" slurm/v2_step4_build_graphs.sh)
echo "Queued V2 Step 4 (Best-4D & Sky Graphs): $JOB_V2_STEP4 (afterok:${JOB_V2_STEP3})"

JOB_V2_STEP5=$(_submit "--dependency=afterok:${JOB_V2_STEP4}" slurm/v2_step5_train_5d.sh)
echo "Queued V2 Step 5 (Train V2 5D Clean/Noisy): $JOB_V2_STEP5 (afterok:${JOB_V2_STEP4})"

JOB_V2_STEP6=$(_submit "--dependency=afterok:${JOB_V2_STEP5}" slurm/v2_step6_train_best4d.sh)
echo "Queued V2 Step 6 (Train V2 Best-4D + Gaia Infer): $JOB_V2_STEP6 (afterok:${JOB_V2_STEP5})"

echo "=================================================================="
echo "All Study 2 V2 pipeline stages queued!"
echo "Monitor queue with: squeue -u \$USER"
echo "=================================================================="
