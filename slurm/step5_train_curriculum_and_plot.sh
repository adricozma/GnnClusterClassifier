#!/bin/bash
#SBATCH --job-name=Study2_Step5_Train
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --mem=220G
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
module load py-numpy/1.26.4
module load py-scipy/1.13.0
module load py-matplotlib/3.3.4
module load py-h5py/3.11.0

export MPLBACKEND=Agg
source env_astro/bin/activate

SIM_DIR="${SIM_DIR:-/home/cozma/swift_sims/outputs_unbiased}"

echo "=================================================================="
echo "STAGE 1: CLEAN PRE-TRAINING (Noiseless 5D Astrometric Graphs)"
echo "=================================================================="

python3 scripts/train_gcn.py \
    --data_folder "/scratch/cozma/graph_data_clean" \
    --n 5 \
    --lr 0.001 \
    --bs 256 \
    --epochs 50 \
    --split_mode ic \
    --save_model "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
    --log_file "$OUT_DIR/clean_5obs/training_metrics_gcn_clean.log"

python3 scripts/train_gat.py \
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
    --labels_dir "GNN-jed/labels_clean"
python3 scripts/live_monitor.py --once || true

echo "=================================================================="
echo "STAGE 2: CURRICULUM FINE-TUNING ON GAIA DR3 NOISY GRAPHS"
echo "=================================================================="

python3 scripts/train_gcn.py \
    --data_folder "/scratch/cozma/graph_data_noisy" \
    --n 5 \
    --lr 0.0005 \
    --bs 256 \
    --epochs 50 \
    --split_mode ic \
    --pretrained_weights "$OUT_DIR/clean_5obs/best_gcn_clean.pth" \
    --save_model "$OUT_DIR/noisy_5obs/best_gcn_noisy.pth" \
    --log_file "$OUT_DIR/noisy_5obs/training_metrics_gcn_noisy.log"

python3 scripts/train_gat.py \
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
    --labels_dir "GNN-jed/labels_clean"

echo "=== STEP 5 CURRICULUM TRAINING & FIGURE GENERATION COMPLETED ==="
python3 scripts/live_monitor.py --once || true
