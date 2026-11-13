#!/bin/bash
#SBATCH --job-name=Study2V2_Step6_Best4D
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=246G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=120:00:00
#SBATCH --output=slurm_logs/v2_step6_best4d_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

GRAPH_V2_CLEAN="${GRAPH_V2_CLEAN:-/scratch/cozma/study2_v2/graphs_clean}"
GRAPH_V2_NOISY="${GRAPH_V2_NOISY:-/scratch/cozma/study2_v2/graphs_noisy}"
STUDY2_V1_MODELS="${STUDY2_V1_MODELS:-GNN-jed/trained_models/study2}"
STUDY2_V2_MODELS="${STUDY2_V2_MODELS:-GNN-jed/trained_models/study2_v2}"

NNODES="${SLURM_NNODES:-1}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-36}"
export OMP_NUM_THREADS="$CPUS_PER_NODE"
export MKL_NUM_THREADS="$CPUS_PER_NODE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29800 + (SLURM_JOB_ID % 900)))
export MASTER_ADDR MASTER_PORT

module load gcc/13.2.0
module load python/3.11.7
export MPLBACKEND=Agg
source env_astro/bin/activate

CONFIG_ENV="${STUDY2_V2_MODELS}/best_4d_config.env"
if [ ! -f "$CONFIG_ENV" ]; then
    python3 scripts/v2_select_best_4d.py \
        --study2_models_dir "$STUDY2_V1_MODELS" \
        --out_dir "$STUDY2_V2_MODELS"
fi
# shellcheck source=/dev/null
source "$CONFIG_ENV"

OUT_4D_CLEAN="${STUDY2_V2_MODELS}/${BEST_4D_TAG}_clean"
OUT_4D_NOISY="${STUDY2_V2_MODELS}/${BEST_4D_TAG}_noisy"
PRETRAINED_5D_CLEAN="${STUDY2_V2_MODELS}/clean_5obs/best_v2_clean_5obs.pth"
PRETRAINED_5D_NOISY="${STUDY2_V2_MODELS}/noisy_5obs/best_v2_noisy_5obs.pth"
INFER_OUT="${STUDY2_V2_MODELS}/inference"

mkdir -p slurm_logs "$OUT_4D_CLEAN" "$OUT_4D_NOISY" "$INFER_OUT"

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

echo "=================================================================="
echo "Study 2 V2 Step 6: Best-4D Training ($BEST_4D_TAG | indices=$BEST_4D_FEATURE_INDICES)"
echo "=================================================================="

echo "=== Step 6A: Clean Best-4D ($BEST_4D_TAG) via V2 Clean 5D -> 4D Subspace Transfer ==="
run_ddp_with_oom_fallback scripts/v2_train_gat.py 8 \
    --data_folder "$GRAPH_V2_CLEAN" \
    --feature_indices "$BEST_4D_FEATURE_INDICES" \
    --num_classes 3 \
    --lr 0.0005 \
    --bs 64 \
    --epochs 35 \
    --dropout 0.4 \
    --weight_decay 0.001 \
    --patience 10 \
    --scheduler cosine \
    --pretrained_weights "$PRETRAINED_5D_CLEAN" \
    --save_model "${OUT_4D_CLEAN}/best_v2_${BEST_4D_TAG}_clean.pth" \
    --log_file "${OUT_4D_CLEAN}/training_metrics_v2_${BEST_4D_TAG}_clean.log"

echo "=== Step 6B: Gaia DR3 Noisy Best-4D ($BEST_4D_TAG) Curriculum Fine-Tune ==="
run_ddp_with_oom_fallback scripts/v2_train_gat.py 8 \
    --data_folder "$GRAPH_V2_NOISY" \
    --feature_indices "$BEST_4D_FEATURE_INDICES" \
    --num_classes 3 \
    --lr 0.0005 \
    --bs 64 \
    --epochs 40 \
    --dropout 0.4 \
    --weight_decay 0.001 \
    --patience 10 \
    --scheduler cosine \
    --pretrained_weights "${OUT_4D_CLEAN}/best_v2_${BEST_4D_TAG}_clean.pth" \
    --save_model "${OUT_4D_NOISY}/best_v2_${BEST_4D_TAG}_noisy.pth" \
    --log_file "${OUT_4D_NOISY}/training_metrics_v2_${BEST_4D_TAG}_noisy.log"

echo "=== Step 6C: Automated Gaia DR3 Catalog Inference (Ibata 2024 6D/5D/4D Streams) ==="
python3 scripts/v2_infer_gaia_dr3.py \
    --input_csv "make_ICs/stars_6D_ibata2024.csv" \
    --out_csv "${INFER_OUT}/ibata2024_v2_5d_predictions.csv" \
    --out_summary "${INFER_OUT}/ibata2024_v2_5d_summary.json" \
    --model_5d "$PRETRAINED_5D_NOISY" \
    --mode 5d

python3 scripts/v2_infer_gaia_dr3.py \
    --input_csv "make_ICs/stars_6D_ibata2024.csv" \
    --out_csv "${INFER_OUT}/ibata2024_v2_${BEST_4D_TAG}_predictions.csv" \
    --out_summary "${INFER_OUT}/ibata2024_v2_${BEST_4D_TAG}_summary.json" \
    --model_4d "${OUT_4D_NOISY}/best_v2_${BEST_4D_TAG}_noisy.pth" \
    --config_4d "${STUDY2_V2_MODELS}/best_4d_config.json" \
    --mode 4d

echo "[DONE] Study 2 V2 Step 6 Best-4D training and Gaia DR3 inference complete."
