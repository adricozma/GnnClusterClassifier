#!/bin/bash
#SBATCH --job-name=Study2_Step5_Train81
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=492G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=168:00:00
#SBATCH --output=slurm_logs/step5_train_curriculum_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT_DIR="GNN-jed/trained_models/study2"
mkdir -p slurm_logs "$OUT_DIR/clean_5obs" "$OUT_DIR/noisy_5obs"

module load gcc/13.2.0
module load python/3.11.7

export MPLBACKEND=Agg
source env_astro/bin/activate

SIM_DIR="${SIM_DIR:-/scratch/cozma/study2/outputs_unbiased}"
NNODES="${SLURM_NNODES:-1}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-72}"
export OMP_NUM_THREADS="$CPUS_PER_NODE"
export MKL_NUM_THREADS="$CPUS_PER_NODE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
export MASTER_ADDR MASTER_PORT

run_ddp_with_oom_fallback() {
    local script="$1"
    local default_micro="$2"
    shift 2
    local extra_args=("$@")

    for micro in "$default_micro" $((default_micro / 2)) $((default_micro / 4)) 2; do
        if [ "$micro" -lt 1 ]; then
            micro=1
        fi
        echo "  [Launcher] Running $script across $NNODES node(s) (72 cores, 492G/node) with --micro_bs $micro..."
        if [ "$NNODES" -gt 1 ]; then
            if srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_NODE" \
                torchrun \
                --nnodes="$NNODES" \
                --nproc_per_node=1 \
                --rdzv_id="${SLURM_JOB_ID}_${RANDOM}" \
                --rdzv_backend=c10d \
                --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
                "$script" --micro_bs "$micro" "${extra_args[@]}"; then
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

echo "=================================================================="
echo "STAGE 1: CLEAN PRE-TRAINING (Noiseless 5D Astrometric Graphs, 81 Streams)"
echo "Nodes: $NNODES x 72 CPUs (492G RAM/node)"
echo "=================================================================="

run_ddp_with_oom_fallback "scripts/train_gcn.py" 16 \
    --data_folder "/scratch/cozma/graph_data_clean" \
    --n 5 \
    --lr 0.001 \
    --bs 256 \
    --epochs 50 \
    --split_mode ic \
    --save_model "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
    --log_file "$OUT_DIR/clean_5obs/training_metrics_gcn_clean.log"

run_ddp_with_oom_fallback "scripts/train_gat.py" 4 \
    --data_folder "/scratch/cozma/graph_data_clean" \
    --n 5 \
    --lr 0.001 \
    --bs 64 \
    --epochs 50 \
    --split_mode ic \
    --save_model "$OUT_DIR/clean_5obs/best_gat_clean.pth" \
    --log_file "$OUT_DIR/clean_5obs/training_metrics_gat_clean.log"

echo "Generating Stage 1 PNG & Interactive HTML Figures..."
python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR/clean_5obs" \
    --tag "clean_5obs" \
    --gcn_log "$OUT_DIR/clean_5obs/training_metrics_gcn_clean.log" \
    --gat_log "$OUT_DIR/clean_5obs/training_metrics_gat_clean.log" \
    --gcn_weights "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
    --gat_weights "$OUT_DIR/clean_5obs/best_gat_clean.pth" \
    --graph_dir "/scratch/cozma/graph_data_clean" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean"
python3 scripts/live_monitor.py --once || true

echo "=================================================================="
echo "STAGE 2: CURRICULUM FINE-TUNING ON GAIA DR3 NOISY GRAPHS (81 Streams)"
echo "=================================================================="

run_ddp_with_oom_fallback "scripts/train_gcn.py" 16 \
    --data_folder "/scratch/cozma/graph_data_noisy" \
    --n 5 \
    --lr 0.0005 \
    --bs 256 \
    --epochs 50 \
    --split_mode ic \
    --pretrained_weights "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
    --save_model "$OUT_DIR/noisy_5obs/best_gcn_noisy.pth" \
    --log_file "$OUT_DIR/noisy_5obs/training_metrics_gcn_noisy.log"

run_ddp_with_oom_fallback "scripts/train_gat.py" 4 \
    --data_folder "/scratch/cozma/graph_data_noisy" \
    --n 5 \
    --lr 0.0005 \
    --bs 64 \
    --epochs 50 \
    --split_mode ic \
    --pretrained_weights "$OUT_DIR/clean_5obs/best_gat_clean.pth" \
    --save_model "$OUT_DIR/noisy_5obs/best_gat_noisy.pth" \
    --log_file "$OUT_DIR/noisy_5obs/training_metrics_gat_noisy.log"

echo "Generating Stage 2 PNG & Interactive HTML Figures..."
python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR/noisy_5obs" \
    --tag "noisy_5obs" \
    --gcn_log "$OUT_DIR/noisy_5obs/training_metrics_gcn_noisy.log" \
    --gat_log "$OUT_DIR/noisy_5obs/training_metrics_gat_noisy.log" \
    --gcn_weights "$OUT_DIR/noisy_5obs/best_gcn_noisy.pth" \
    --gat_weights "$OUT_DIR/noisy_5obs/best_gat_noisy.pth" \
    --graph_dir "/scratch/cozma/graph_data_noisy" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean"

echo "=== STEP 5 CURRICULUM TRAINING & FIGURE GENERATION COMPLETED ==="
python3 scripts/live_monitor.py --once || true
