#!/bin/bash
#SBATCH --job-name=Study2_Step6_4Obs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=246G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=168:00:00
#SBATCH --array=0-4%1
#SBATCH --output=slurm_logs/step6_ablation4obs_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# 5 combinations of 4 astrometric observables (dropping 1 observable at a time in addition to v_los)
# Columns: 0=RA, 1=Dec, 2=Parallax, 3=pmRA, 4=pmDec
FEAT_SETS=("1,2,3,4" "0,2,3,4" "0,1,3,4" "0,1,2,4" "0,1,2,3")
TAGS=("no_RA" "no_Dec" "no_Parallax" "no_pmRA" "no_pmDec")

FEAT="${FEAT_SETS[$SLURM_ARRAY_TASK_ID]}"
TAG="4obs_${TAGS[$SLURM_ARRAY_TASK_ID]}"
GRAPH_DIR_CLEAN="/scratch/cozma/graph_data_${TAG}_clean"
GRAPH_DIR_NOISY="/scratch/cozma/graph_data_${TAG}_noisy"
OUT_DIR_5OBS_CLEAN="GNN-jed/trained_models/study2/clean_5obs"
OUT_DIR_CLEAN="GNN-jed/trained_models/study2/${TAG}_clean"
OUT_DIR_NOISY="GNN-jed/trained_models/study2/${TAG}_noisy"
SIM_DIR="${SIM_DIR:-/scratch/cozma/study2/outputs_unbiased}"

NNODES="${SLURM_NNODES:-1}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-36}"
export OMP_NUM_THREADS="$CPUS_PER_NODE"
export MKL_NUM_THREADS="$CPUS_PER_NODE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29600 + (SLURM_JOB_ID % 900) + SLURM_ARRAY_TASK_ID))
export MASTER_ADDR MASTER_PORT

mkdir -p slurm_logs "$GRAPH_DIR_CLEAN" "$GRAPH_DIR_NOISY" "$OUT_DIR_CLEAN" "$OUT_DIR_NOISY"

module load gcc/13.2.0
module load python/3.11.7

export MPLBACKEND=Agg
source env_astro/bin/activate

run_ddp_with_oom_fallback() {
    local script="$1"
    local default_micro="$2"
    shift 2
    local extra_args=("$@")

    for micro in "$default_micro" $((default_micro / 2)) $((default_micro / 4)) 2; do
        if [ "$micro" -lt 1 ]; then
            micro=1
        fi
        echo "  [Launcher] Running $script across $NNODES node(s) ($CPUS_PER_NODE cores, 246G/node) with --micro_bs $micro..."
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
        echo "  [OOM / Crash Fallback] Execution interrupted with --micro_bs $micro; enabling disk activation offload and retrying with smaller micro-batch from latest epoch checkpoint..."
        extra_args+=("--force_disk_offload")
    done
    return 1
}

echo "=== [1/4] Verifying / Building 4D Clean k-NN Graphs for $TAG (features=$FEAT) ==="
python3 scripts/build_graph_dataset.py \
    --obs_dir "/scratch/cozma/study2/obs_clean" \
    --labels_dir "/scratch/cozma/study2/labels_clean" \
    --output_dir "$GRAPH_DIR_CLEAN" \
    --rotations 5 \
    --k_neighbors 20 \
    --feature_indices "$FEAT" \
    --num_workers "$CPUS_PER_NODE"

echo "=== [2/4] Verifying / Building 4D Gaia DR3 Noisy k-NN Graphs for $TAG (features=$FEAT) ==="
python3 scripts/build_graph_dataset.py \
    --obs_dir "/scratch/cozma/study2/obs_gaia_noisy" \
    --labels_dir "/scratch/cozma/study2/labels_clean" \
    --output_dir "$GRAPH_DIR_NOISY" \
    --rotations 5 \
    --k_neighbors 20 \
    --feature_indices "$FEAT" \
    --num_workers "$CPUS_PER_NODE"

PRETRAIN_GAT_5D=""
if [ -f "$OUT_DIR_5OBS_CLEAN/best_gat_clean.pth" ]; then
    PRETRAIN_GAT_5D="$OUT_DIR_5OBS_CLEAN/best_gat_clean.pth"
fi

echo "=== [3/4] Stage 1: Clean GAT Pre-Training on 4 Observables ($TAG, initialized from 5D Clean GAT weights) ==="
if grep -q "=== END OF TRAINING ===" "$OUT_DIR_CLEAN/training_metrics_gat_${TAG}_clean.log" 2>/dev/null && [ -f "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" ]; then
    echo "  [Step 6] Stage 1 Clean GAT pre-training for $TAG already completed. Skipping."
else
    run_ddp_with_oom_fallback "scripts/train_gat.py" 32 \
        --data_folder "$GRAPH_DIR_CLEAN" \
        --n 5 \
        --stochastic_rotation \
        --bf16 \
        --lr 0.001 \
        --bs 64 \
        --epochs 20 \
        --patience 8 \
        --dropout 0.4 \
        --weight_decay 1e-3 \
        --scheduler onecycle \
        --split_mode ic \
        --pretrained_weights "$PRETRAIN_GAT_5D" \
        --save_model "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" \
        --log_file "$OUT_DIR_CLEAN/training_metrics_gat_${TAG}_clean.log"
fi

python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR_CLEAN" \
    --tag "${TAG}_clean" \
    --in_channels 4 \
    --gat_log "$OUT_DIR_CLEAN/training_metrics_gat_${TAG}_clean.log" \
    --gat_weights "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" \
    --graph_dir "$GRAPH_DIR_CLEAN" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean" || true

echo "=== [4/4] Stage 2: Gaia DR3 Noisy GAT Fine-Tuning on 4 Observables ($TAG) ==="
if grep -q "=== END OF TRAINING ===" "$OUT_DIR_NOISY/training_metrics_gat_${TAG}_noisy.log" 2>/dev/null && [ -f "$OUT_DIR_NOISY/best_gat_${TAG}_noisy.pth" ]; then
    echo "  [Step 6] Stage 2 Gaia DR3 Noisy GAT fine-tuning for $TAG already completed. Skipping."
else
    run_ddp_with_oom_fallback "scripts/train_gat.py" 32 \
        --data_folder "$GRAPH_DIR_NOISY" \
        --n 5 \
        --stochastic_rotation \
        --bf16 \
        --lr 0.0005 \
        --bs 64 \
        --epochs 20 \
        --patience 8 \
        --dropout 0.4 \
        --weight_decay 1e-3 \
        --scheduler onecycle \
        --split_mode ic \
        --pretrained_weights "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" \
        --save_model "$OUT_DIR_NOISY/best_gat_${TAG}_noisy.pth" \
        --log_file "$OUT_DIR_NOISY/training_metrics_gat_${TAG}_noisy.log"
fi

python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR_NOISY" \
    --tag "${TAG}_noisy" \
    --in_channels 4 \
    --gat_log "$OUT_DIR_NOISY/training_metrics_gat_${TAG}_noisy.log" \
    --gat_weights "$OUT_DIR_NOISY/best_gat_${TAG}_noisy.pth" \
    --graph_dir "$GRAPH_DIR_NOISY" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean" || true

echo "=== COMPLETED 4-OBSERVABLE CLEAN + GAIA NOISY GAT ABLATION FOR $TAG ==="
python3 scripts/live_monitor.py --once || true
