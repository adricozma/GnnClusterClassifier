#!/bin/bash
#SBATCH --job-name=Study2V2_Step4_BuildGraphs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/v2_step4_graphs_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OBS_V2_DIR="${OBS_V2_DIR:-/scratch/cozma/study2_v2/obs_unbiased}"
GRAPH_V2_CLEAN="${GRAPH_V2_CLEAN:-/scratch/cozma/study2_v2/graphs_clean}"
GRAPH_V2_NOISY="${GRAPH_V2_NOISY:-/scratch/cozma/study2_v2/graphs_noisy}"
STUDY2_V1_MODELS="${STUDY2_V1_MODELS:-GNN-jed/trained_models/study2}"
STUDY2_V2_MODELS="${STUDY2_V2_MODELS:-GNN-jed/trained_models/study2_v2}"
K_NEIGHBORS="${K_NEIGHBORS:-32}"

mkdir -p slurm_logs "$GRAPH_V2_CLEAN" "$GRAPH_V2_NOISY" "$STUDY2_V2_MODELS"

module load gcc/13.2.0
module load python/3.11.7
export MPLBACKEND=Agg
source env_astro/bin/activate

echo "=================================================================="
echo "Study 2 V2 Step 4: Best-4D Selection & 2D-Sky k-NN Graph Construction"
echo "=================================================================="

python3 scripts/v2_select_best_4d.py \
    --study2_models_dir "$STUDY2_V1_MODELS" \
    --out_dir "$STUDY2_V2_MODELS"

python3 scripts/v2_build_graph_dataset.py \
    --obs_dir "$OBS_V2_DIR" \
    --output_dir "$GRAPH_V2_CLEAN" \
    --regime clean \
    --k_neighbors "$K_NEIGHBORS" \
    --feature_indices "0,1,2,3,4" \
    --num_workers "${SLURM_CPUS_PER_TASK:-36}"

python3 scripts/v2_build_graph_dataset.py \
    --obs_dir "$OBS_V2_DIR" \
    --output_dir "$GRAPH_V2_NOISY" \
    --regime noisy \
    --k_neighbors "$K_NEIGHBORS" \
    --feature_indices "0,1,2,3,4" \
    --num_workers "${SLURM_CPUS_PER_TASK:-36}"

echo "[DONE] V2 Clean and Noisy sky-kNN graph datasets ready."
