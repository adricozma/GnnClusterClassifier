#!/bin/bash
#SBATCH --job-name=Study2_Step2_SWIFT
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --mem=180G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=120:00:00
#SBATCH --array=0-80%6
#SBATCH --output=slurm_logs/step2_swift_%A_%a.out

set -euo pipefail
REPO_DIR="$SLURM_SUBMIT_DIR"
SWIFT_BIN="/home/cozma/swift_sims/swift"
PARAMS_TEMPLATE="$REPO_DIR/configs/params.yml"
ICS_DIR="$REPO_DIR/make_ICs/initial_conditions_unbiased"
STUDY1_OUTPUT_BASE="/home/cozma/swift_sims/outputs"
OUTPUT_BASE="/scratch/cozma/study2/outputs_unbiased"
REUSE_EXISTING_SWIFT="${REUSE_EXISTING_SWIFT:-1}"
N_THREADS="${SLURM_NTASKS:-36}"

validate_sim() {
    local target_dir="$1"
    [ -d "$target_dir" ] || return 1
    python3 -c "
import sys, os
d = sys.argv[1]
magic = b'\x89HDF\r\n\x1a\n'
for i in range(101):
    p = os.path.join(d, f'snapshot_{i:04d}.hdf5')
    if not (os.path.isfile(p) and os.path.getsize(p) > 50000):
        sys.exit(1)
    with open(p, 'rb') as f:
        if f.read(8) != magic:
            sys.exit(1)
sys.exit(0)
" "$target_dir" 2>/dev/null
}

mkdir -p "$REPO_DIR/slurm_logs" "$OUTPUT_BASE"

mapfile -t FILES_IC < <(ls "$ICS_DIR"/IC_*.hdf5 | sort -V)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#FILES_IC[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of IC files (${#FILES_IC[@]}). Exiting cleanly."
    exit 0
fi

CURRENT_FILE_IC="${FILES_IC[$SLURM_ARRAY_TASK_ID]}"
BASENAME_EXT="${CURRENT_FILE_IC##*/}"
BASENAME="${BASENAME_EXT%.*}"

STUDY1_FOLDER="$STUDY1_OUTPUT_BASE/${BASENAME}"
OUTPUT_FOLDER="$OUTPUT_BASE/${BASENAME}"

# Check existing OUTPUT_FOLDER: validate with Python (all 101 snapshots exist, >50KB, HDF5 magic)
# If invalid symlink or incomplete/corrupted, remove it and run fresh.
if [ -e "$OUTPUT_FOLDER" ] || [ -L "$OUTPUT_FOLDER" ]; then
    if validate_sim "$OUTPUT_FOLDER"; then
        echo "Simulation $BASENAME already valid at $OUTPUT_FOLDER. Skipping."
        exit 0
    else
        echo "Simulation $OUTPUT_FOLDER is an invalid symlink or incomplete/corrupted. Removing..."
        rm -rf "$OUTPUT_FOLDER"
    fi
fi

# If Study 1 folder is valid, symlink it to OUTPUT_FOLDER
if [ "$REUSE_EXISTING_SWIFT" = "1" ] && validate_sim "$STUDY1_FOLDER"; then
    ln -s "$STUDY1_FOLDER" "$OUTPUT_FOLDER"
    echo "Linked verified Study 1 SWIFT run from $STUDY1_FOLDER -> $OUTPUT_FOLDER"
    exit 0
fi

mkdir -p "$OUTPUT_FOLDER"

# Clean any incomplete artifacts before starting fresh SWIFT integration
rm -rf "$OUTPUT_FOLDER/restart" "$OUTPUT_FOLDER"/snapshot_*.hdf5 "$OUTPUT_FOLDER"/statistics.txt

CONFIG_FILE="$OUTPUT_FOLDER/config_task_${BASENAME}.yaml"
sed -e "s|__OUTPUT_DIR__|$OUTPUT_FOLDER|g" \
    -e "s|__IC_FILE_PATH__|$CURRENT_FILE_IC|g" \
    "$PARAMS_TEMPLATE" > "$CONFIG_FILE"

cd "$OUTPUT_FOLDER"
module load gcc/13.2.0 gsl hdf5 metis

echo "Running SWIFT on $BASENAME (task $SLURM_ARRAY_TASK_ID) with $N_THREADS threads..."
"$SWIFT_BIN" \
    --self-gravity \
    --external-gravity \
    --threads="$N_THREADS" \
    --pin \
    "$CONFIG_FILE"

echo "Finished SWIFT simulation for $BASENAME."

FINAL_SNAP="$OUTPUT_FOLDER/snapshot_0100.hdf5"
if ! python3 -c "import os, sys; p = sys.argv[1]; sys.exit(0 if (os.path.isfile(p) and os.path.getsize(p) > 50000) else 1)" "$FINAL_SNAP"; then
    echo "ERROR: Final snapshot $FINAL_SNAP missing or size <= 50000 bytes after SWIFT run!" >&2
    exit 1
fi

cd "$REPO_DIR"
module load python/3.11.7
source env_astro/bin/activate
python3 scripts/live_monitor.py --once || true
