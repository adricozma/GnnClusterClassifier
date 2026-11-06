#!/bin/bash
#SBATCH --job-name=Study2_Step2_SWIFT
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --mem=400G
#SBATCH --account=master
#SBATCH --qos=academic
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --array=0-80
#SBATCH --output=slurm_logs/step2_swift_%A_%a.out

set -euo pipefail
REPO_DIR="$SLURM_SUBMIT_DIR"
SWIFT_BIN="/home/cozma/swift_sims/swift"
PARAMS_TEMPLATE="$REPO_DIR/configs/params.yml"
ICS_DIR="$REPO_DIR/make_ICs/initial_conditions_unbiased"
STUDY1_OUTPUT_BASE="/home/cozma/swift_sims/outputs"
OUTPUT_BASE="/home/cozma/swift_sims/outputs_unbiased"
REUSE_EXISTING_SWIFT="${REUSE_EXISTING_SWIFT:-1}"

mkdir -p "$REPO_DIR/slurm_logs" "$OUTPUT_BASE"

FILES_IC=("$ICS_DIR"/IC_*.hdf5)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#FILES_IC[@]}" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of IC files (${#FILES_IC[@]}). Exiting cleanly."
    exit 0
fi

CURRENT_FILE_IC="${FILES_IC[$SLURM_ARRAY_TASK_ID]}"
BASENAME_EXT="${CURRENT_FILE_IC##*/}"
BASENAME="${BASENAME_EXT%.*}"

OUTPUT_FOLDER="$OUTPUT_BASE/${BASENAME}"

# If REUSE_EXISTING_SWIFT=1 and Study 1 already has all 101 snapshots for this cluster,
# create a read-only symlink in outputs_unbiased/ to save 88% of N-body compute time.
if [ "$REUSE_EXISTING_SWIFT" = "1" ] && [ -f "$STUDY1_OUTPUT_BASE/${BASENAME}/snapshot_0100.hdf5" ]; then
    if [ ! -e "$OUTPUT_FOLDER" ]; then
        ln -s "$STUDY1_OUTPUT_BASE/${BASENAME}" "$OUTPUT_FOLDER"
        echo "Linked existing completed SWIFT run from $STUDY1_OUTPUT_BASE/${BASENAME} -> $OUTPUT_FOLDER"
    else
        echo "Simulation $BASENAME already present at $OUTPUT_FOLDER. Skipping."
    fi
    exit 0
fi

mkdir -p "$OUTPUT_FOLDER"

if [ -f "$OUTPUT_FOLDER/snapshot_0100.hdf5" ]; then
    echo "Simulation $BASENAME already completed (snapshot_0100.hdf5 exists). Skipping."
    exit 0
fi

CONFIG_FILE="$OUTPUT_FOLDER/config_task_${BASENAME}.yaml"
sed -e "s|__OUTPUT_DIR__|$OUTPUT_FOLDER|g" \
    -e "s|__IC_FILE_PATH__|$CURRENT_FILE_IC|g" \
    "$PARAMS_TEMPLATE" > "$CONFIG_FILE"

cd "$OUTPUT_FOLDER"
module load gcc/13.2.0 gsl hdf5 metis

echo "Running SWIFT on $BASENAME with $SLURM_NTASKS threads..."
"$SWIFT_BIN" \
    --self-gravity \
    --external-gravity \
    --threads="$SLURM_NTASKS" \
    --pin \
    "$CONFIG_FILE"

echo "Finished SWIFT simulation for $BASENAME."
