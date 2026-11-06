#!/bin/bash
#SBATCH --job-name=Study2_Step4_Graphs
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --mem=180G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/step4_graphs_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs /scratch/cozma/graph_data_clean /scratch/cozma/graph_data_noisy

module load gcc/13.2.0
module load python/3.11.7
module load py-numpy/1.26.4
module load py-scipy/1.13.0
module load py-matplotlib/3.3.4

export MPLBACKEND=Agg
source env_astro/bin/activate

N_WORKERS="${SLURM_NTASKS:-36}"

echo "=== Building 5D Clean k-NN Graphs (k=20, cKDTree) in /scratch/cozma/graph_data_clean ($N_WORKERS workers) ==="
python3 scripts/build_graph_dataset.py \
    --obs_dir "GNN-jed/obs_clean" \
    --labels_dir "GNN-jed/labels_clean" \
    --output_dir "/scratch/cozma/graph_data_clean" \
    --rotations 5 \
    --k_neighbors 20 \
    --feature_indices "0,1,2,3,4" \
    --num_workers "$N_WORKERS"

echo "=== Building 5D Gaia DR3 Noisy k-NN Graphs (k=20, cKDTree) in /scratch/cozma/graph_data_noisy ($N_WORKERS workers) ==="
python3 scripts/build_graph_dataset.py \
    --obs_dir "GNN-jed/obs_gaia_noisy" \
    --labels_dir "GNN-jed/labels_clean" \
    --output_dir "/scratch/cozma/graph_data_noisy" \
    --rotations 5 \
    --k_neighbors 20 \
    --feature_indices "0,1,2,3,4" \
    --num_workers "$N_WORKERS"

echo "=== STEP 4 GRAPH CONSTRUCTION COMPLETED ==="
python3 scripts/live_monitor.py --once || true
