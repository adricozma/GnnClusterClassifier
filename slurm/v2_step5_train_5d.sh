#!/bin/bash
#SBATCH --job-name=Study2V2_Step5_Train5D
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=246G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=120:00:00
#SBATCH --output=slurm_logs/v2_step5_train5d_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

GRAPH_V2_CLEAN="${GRAPH_V2_CLEAN:-/scratch/cozma/study2_v2/graphs_clean}"
GRAPH_V2_NOISY="${GRAPH_V2_NOISY:-/scratch/cozma/study2_v2/graphs_noisy}"
STUDY2_V2_MODELS="${STUDY2_V2_MODELS:-GNN-jed/trained_models/study2_v2}"
OUT_CLEAN="${STUDY2_V2_MODELS}/clean_5obs"
OUT_NOISY="${STUDY2_V2_MODELS}/noisy_5obs"

NNODES="${SLURM_NNODES:-1}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-36}"
export OMP_NUM_THREADS="$CPUS_PER_NODE"
export MKL_NUM_THREADS="$CPUS_PER_NODE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29700 + (SLURM_JOB_ID % 900)))
export MASTER_ADDR MASTER_PORT

mkdir -p slurm_logs "$OUT_CLEAN" "$OUT_NOISY"

module load gcc/13.2.0
module load python/3.11.7
export MPLBACKEND=Agg
source env_astro/bin/activate

run_ddp_with_oom_fallback() {
    local script="$1"
    local default_micro="$2"
    shift 2
    local extra_args=("$@")

    for micro in "$default_micro" $((default_micro / 2)) $((default_micro / 4)) 1; do
        if [ "$micro" -lt 1 ]; then
            micro=1
        fi
        echo "  [Launcher] Running $script across $NNODES node(s) ($CPUS_PER_NODE cores/node) with --micro_bs $micro..."
        if [ "$NNODES" -gt 1 ]; then
            if srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_NODE" \
                python3 "$script" --micro_bs "$micro" "${extra_args[@]}"; then
                return 0
            fi
        else
            if python3 "$script" --micro_bs "$micro" "${extra_args[@]}"; then
                return 0
            fi
        fi
        echo "  [OOM Fallback] Retrying with smaller --micro_bs and --force_disk_offload..."
        extra_args+=("--force_disk_offload")
    done
    return 1
}

echo "=== Study 2 V2 Step 5A: Curriculum Clean 5D GAT_V2 (3-class + MW contaminants) ==="
run_ddp_with_oom_fallback scripts/v2_train_gat.py 8 \
    --data_folder "$GRAPH_V2_CLEAN" \
    --feature_indices "0,1,2,3,4" \
    --num_classes 3 \
    --lr 0.001 \
    --bs 64 \
    --epochs 45 \
    --dropout 0.4 \
    --weight_decay 0.001 \
    --patience 10 \
    --scheduler onecycle \
    --save_model "${OUT_CLEAN}/best_v2_clean_5obs.pth" \
    --log_file "${OUT_CLEAN}/training_metrics_v2_clean_5obs.log"

echo "=== Study 2 V2 Step 5B: Curriculum Gaia DR3 Noisy 5D GAT_V2 Fine-Tune ==="
run_ddp_with_oom_fallback scripts/v2_train_gat.py 8 \
    --data_folder "$GRAPH_V2_NOISY" \
    --feature_indices "0,1,2,3,4" \
    --num_classes 3 \
    --lr 0.0005 \
    --bs 64 \
    --epochs 40 \
    --dropout 0.4 \
    --weight_decay 0.001 \
    --patience 10 \
    --scheduler cosine \
    --pretrained_weights "${OUT_CLEAN}/best_v2_clean_5obs.pth" \
    --save_model "${OUT_NOISY}/best_v2_noisy_5obs.pth" \
    --log_file "${OUT_NOISY}/training_metrics_v2_noisy_5obs.log"

echo "[DONE] V2 Step 5 Clean + Noisy 5D training complete."
