#!/bin/bash
#SBATCH --job-name=Study2_Step6_4Obs
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --mem=220G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=168:00:00
#SBATCH --array=0-4%3
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
OUT_DIR_CLEAN="GNN-jed/trained_models/study2/${TAG}_clean"
OUT_DIR_NOISY="GNN-jed/trained_models/study2/${TAG}_noisy"
SIM_DIR="${SIM_DIR:-/scratch/cozma/study2/outputs_unbiased}"
N_WORKERS="${SLURM_NTASKS:-36}"

mkdir -p slurm_logs "$GRAPH_DIR_CLEAN" "$GRAPH_DIR_NOISY" "$OUT_DIR_CLEAN" "$OUT_DIR_NOISY"

module load gcc/13.2.0
module load python/3.11.7

export MPLBACKEND=Agg
source env_astro/bin/activate

echo "=== [1/4] Building 4D Clean k-NN Graphs for $TAG (features=$FEAT) ==="
python3 scripts/build_graph_dataset.py \
    --obs_dir "/scratch/cozma/study2/obs_clean" \
    --labels_dir "/scratch/cozma/study2/labels_clean" \
    --output_dir "$GRAPH_DIR_CLEAN" \
    --rotations 5 \
    --k_neighbors 20 \
    --feature_indices "$FEAT" \
    --num_workers "$N_WORKERS"

echo "=== [2/4] Building 4D Gaia DR3 Noisy k-NN Graphs for $TAG (features=$FEAT) ==="
python3 scripts/build_graph_dataset.py \
    --obs_dir "/scratch/cozma/study2/obs_gaia_noisy" \
    --labels_dir "/scratch/cozma/study2/labels_clean" \
    --output_dir "$GRAPH_DIR_NOISY" \
    --rotations 5 \
    --k_neighbors 20 \
    --feature_indices "$FEAT" \
    --num_workers "$N_WORKERS"

echo "=== [3/4] Stage 1: Clean Pre-Training on 4 Observables ($TAG) ==="
python3 scripts/train_gcn.py \
    --data_folder "$GRAPH_DIR_CLEAN" \
    --n 5 \
    --lr 0.001 \
    --bs 256 \
    --epochs 50 \
    --split_mode ic \
    --save_model "$OUT_DIR_CLEAN/best_gcn_${TAG}_clean.pth" \
    --log_file "$OUT_DIR_CLEAN/training_metrics_gcn_${TAG}_clean.log"

python3 scripts/train_gat.py \
    --data_folder "$GRAPH_DIR_CLEAN" \
    --n 5 \
    --lr 0.001 \
    --bs 64 \
    --epochs 50 \
    --split_mode ic \
    --save_model "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" \
    --log_file "$OUT_DIR_CLEAN/training_metrics_gat_${TAG}_clean.log"

python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR_CLEAN" \
    --tag "${TAG}_clean" \
    --in_channels 4 \
    --gcn_log "$OUT_DIR_CLEAN/training_metrics_gcn_${TAG}_clean.log" \
    --gat_log "$OUT_DIR_CLEAN/training_metrics_gat_${TAG}_clean.log" \
    --gcn_weights "$OUT_DIR_CLEAN/best_gcn_${TAG}_clean.pth" \
    --gat_weights "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" \
    --graph_dir "$GRAPH_DIR_CLEAN" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean"

echo "=== [4/4] Stage 2: Gaia DR3 Noisy Fine-Tuning on 4 Observables ($TAG) ==="
python3 scripts/train_gcn.py \
    --data_folder "$GRAPH_DIR_NOISY" \
    --n 5 \
    --lr 0.0005 \
    --bs 256 \
    --epochs 50 \
    --split_mode ic \
    --pretrained_weights "$OUT_DIR_CLEAN/best_gcn_${TAG}_clean.pth" \
    --save_model "$OUT_DIR_NOISY/best_gcn_${TAG}_noisy.pth" \
    --log_file "$OUT_DIR_NOISY/training_metrics_gcn_${TAG}_noisy.log"

python3 scripts/train_gat.py \
    --data_folder "$GRAPH_DIR_NOISY" \
    --n 5 \
    --lr 0.0005 \
    --bs 64 \
    --epochs 50 \
    --split_mode ic \
    --pretrained_weights "$OUT_DIR_CLEAN/best_gat_${TAG}_clean.pth" \
    --save_model "$OUT_DIR_NOISY/best_gat_${TAG}_noisy.pth" \
    --log_file "$OUT_DIR_NOISY/training_metrics_gat_${TAG}_noisy.log"

python3 scripts/generate_study2_figures.py \
    --output_dir "$OUT_DIR_NOISY" \
    --tag "${TAG}_noisy" \
    --in_channels 4 \
    --gcn_log "$OUT_DIR_NOISY/training_metrics_gcn_${TAG}_noisy.log" \
    --gat_log "$OUT_DIR_NOISY/training_metrics_gat_${TAG}_noisy.log" \
    --gcn_weights "$OUT_DIR_NOISY/best_gcn_${TAG}_noisy.pth" \
    --gat_weights "$OUT_DIR_NOISY/best_gat_${TAG}_noisy.pth" \
    --graph_dir "$GRAPH_DIR_NOISY" \
    --sim_dir "$SIM_DIR" \
    --labels_dir "/scratch/cozma/study2/labels_clean"

echo "=== COMPLETED 4-OBSERVABLE CLEAN + GAIA NOISY ABLATION FOR $TAG ==="
python3 scripts/live_monitor.py --once || true
