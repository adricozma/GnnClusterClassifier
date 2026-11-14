#!/bin/bash
#SBATCH --job-name=Study2_Step7_GCNStochastic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=246G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/step7_gcn_stochastic_%j.out

# Final step of Study 2: Retrain GCN Clean baseline with stochastic SO(3) rotations
# for rigorous, apples-to-apples comparison with the stochastic GAT.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT_DIR="GNN-jed/trained_models/study2"
mkdir -p slurm_logs "$OUT_DIR/clean_5obs"

module load gcc/13.2.0
module load python/3.11.7

export MPLBACKEND=Agg
source env_astro/bin/activate

SIM_DIR="${SIM_DIR:-/scratch/cozma/study2/outputs_unbiased}"
NNODES="${SLURM_NNODES:-1}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-36}"
export OMP_NUM_THREADS="$CPUS_PER_NODE"
export MKL_NUM_THREADS="$CPUS_PER_NODE"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
export MASTER_ADDR MASTER_PORT

echo "=================================================================="
echo "FINAL STEP 7: RETRAIN GCN CLEAN BASELINE WITH STOCHASTIC SO(3) ROTATIONS"
echo "Nodes: $NNODES x 36 CPUs (246G RAM/node, 72 CPUs total)"
echo "=================================================================="

# Archive static GCN to legacy archive if not already done
LEGACY_DIR="GNN-jed/trained_models/study2_legacy_archive/clean_5obs_static_5rot"
mkdir -p "$LEGACY_DIR"
if [ -f "$OUT_DIR/clean_5obs/best_gcn_clean.pth" ] && [ ! -f "$LEGACY_DIR/best_gcn_clean.pth" ]; then
    cp "$OUT_DIR/clean_5obs/best_gcn_clean.pth" "$LEGACY_DIR/" 2>/dev/null || true
    cp "$OUT_DIR/clean_5obs/training_metrics_gcn_clean.log" "$LEGACY_DIR/" 2>/dev/null || true
fi

# Run GCN Clean retraining with stochastic SO(3) rotations
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_NODE" \
    python3 scripts/train_gcn.py \
        --data_folder "/scratch/cozma/graph_data_clean" \
        --n 5 \
        --stochastic_rotation \
        --bf16 \
        --lr 0.001 \
        --bs 256 \
        --micro_bs 32 \
        --epochs 25 \
        --patience 10 \
        --split_mode ic \
        --allow_non_baseline_gcn \
        --save_model "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
        --log_file "$OUT_DIR/clean_5obs/training_metrics_gcn_clean.log"

echo "Regenerating Stage 1 Figures with Retrained Stochastic GCN Baseline..."
python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR/clean_5obs" \
    --tag "clean_5obs" \
    --gcn_log "$OUT_DIR/clean_5obs/training_metrics_gcn_clean.log" \
    --gat_log "$OUT_DIR/clean_5obs/training_metrics_gat_clean.log" \
    --gcn_weights "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
    --gat_weights "$OUT_DIR/clean_5obs/best_gat_clean.pth" \
    --graph_dir "/scratch/cozma/graph_data_clean" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean" || true

python3 scripts/live_monitor.py --once || true

echo "=== STUDY 2 COMPLETE: GCN AND GAT COMPARISON FINISHED ==="
