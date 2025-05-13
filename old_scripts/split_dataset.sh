#!/bin/bash
set -e

# === Path to your dataset ===
BASE_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse(OLD)"
TOTAL_DIR="${BASE_DIR}/total_dataset"
TRAIN_DIR="${BASE_DIR}/training_data(75%)"
VAL_DIR="${BASE_DIR}/validation_data(15%)"
TEST_DIR="${BASE_DIR}/testing_data(10%)"

# === Ensure output directories exist ===
for OUTDIR in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
    mkdir -p "${OUTDIR}/pdb_data/mmcifs"
    mkdir -p "${OUTDIR}/alignment_data/alignments"
    mkdir -p "${OUTDIR}/sequences"
done

# === Get protein IDs from FASTA filenames ===
cd "${TOTAL_DIR}/sequences"
ALL_IDS=($(ls *.fasta | sed 's/\.fasta$//' | tr '[:upper:]' '[:lower:]' | sort -u))
TOTAL_COUNT=${#ALL_IDS[@]}

# === Shuffle and split ===
SHUFFLED_IDS=($(printf "%s\n" "${ALL_IDS[@]}" | shuf))
TRAIN_COUNT=$((TOTAL_COUNT * 75 / 100))
VAL_COUNT=$((TOTAL_COUNT * 15 / 100))
TEST_COUNT=$((TOTAL_COUNT - TRAIN_COUNT - VAL_COUNT))

TRAIN_IDS=("${SHUFFLED_IDS[@]:0:TRAIN_COUNT}")
VAL_IDS=("${SHUFFLED_IDS[@]:TRAIN_COUNT:VAL_COUNT}")
TEST_IDS=("${SHUFFLED_IDS[@]:TRAIN_COUNT+VAL_COUNT}")

# === Copy function (case-insensitive match) ===
copy_data() {
    local ID="$1"
    local DEST="$2"
    local LOWER_ID=$(echo "$ID" | tr '[:upper:]' '[:lower:]')

    # === Copy .fasta
    fasta_file=$(find "${TOTAL_DIR}/sequences" -iname "${ID}.fasta" | head -n 1)
    if [[ -n "$fasta_file" ]]; then
        cp "$fasta_file" "${DEST}/sequences/"
    fi

    # === Copy .cif
    cif_file=$(find "${TOTAL_DIR}/pdb_data/mmcifs" -iname "${ID}.cif" | head -n 1)
    if [[ -n "$cif_file" ]]; then
        cp "$cif_file" "${DEST}/pdb_data/mmcifs/"
    fi

    # === Copy new alignment folder structure: e.g., 6kwc_A/, 6kwc_B/
    for align_subdir in $(find "${TOTAL_DIR}/alignment_data/alignments" -mindepth 1 -maxdepth 1 -type d -iname "${ID}_*"); do
        cp -r "$align_subdir" "${DEST}/alignment_data/alignments/"
    done
}

# === Apply splits ===
echo "📦 Copying training set ($TRAIN_COUNT entries)..."
for id in "${TRAIN_IDS[@]}"; do
    copy_data "$id" "$TRAIN_DIR"
done

echo "📦 Copying validation set ($VAL_COUNT entries)..."
for id in "${VAL_IDS[@]}"; do
    copy_data "$id" "$VAL_DIR"
done

echo "📦 Copying test set ($TEST_COUNT entries)..."
for id in "${TEST_IDS[@]}"; do
    copy_data "$id" "$TEST_DIR"
done

echo "✅ Dataset successfully split and copied."
