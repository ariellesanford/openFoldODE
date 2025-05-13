#!/bin/bash
set -e

# === Path to your dataset ===
BASE_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"
TOTAL_DIR="${BASE_DIR}/total_dataset"
TRAIN_DIR="${BASE_DIR}/training_data(75%)"
VAL_DIR="${BASE_DIR}/validation_data(15%)"
TEST_DIR="${BASE_DIR}/testing_data(10%)"
MMCIF_DIR="${TOTAL_DIR}/pdb_data/mmcifs"

# === Ensure output directories exist ===
for OUTDIR in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
    mkdir -p "${OUTDIR}/pdb_data/mmcifs"
    mkdir -p "${OUTDIR}/alignment_data/alignments"
    mkdir -p "${OUTDIR}/sequences"
done

# === Extract release dates from .cif files ===
declare -A ID_TO_DATE
for cif_file in "${MMCIF_DIR}"/*.cif; do
    id=$(basename "$cif_file" .cif | tr '[:upper:]' '[:lower:]')
    date=$(grep "_pdbx_audit_revision_history.revision_date" -A 1 "$cif_file" | tail -n 1 | tr -d ' ')
    if [[ "$date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        ID_TO_DATE["$id"]=$date
    fi
done

# === Sort by date and extract IDs ===
sorted_ids=($(for id in "${!ID_TO_DATE[@]}"; do echo "${ID_TO_DATE[$id]} $id"; done | sort | awk '{print $2}'))
TOTAL_COUNT=${#sorted_ids[@]}
TEST_COUNT=$((TOTAL_COUNT / 10))
REMAINING_COUNT=$((TOTAL_COUNT - TEST_COUNT))
TRAIN_COUNT=$((REMAINING_COUNT * 75 / 100))
VAL_COUNT=$((REMAINING_COUNT - TRAIN_COUNT))

# === Split into test and remaining ===
TEST_IDS=("${sorted_ids[@]: -$TEST_COUNT}")
REMAINING_IDS=("${sorted_ids[@]:0:$REMAINING_COUNT}")

# === Shuffle remaining and split ===
SHUFFLED_IDS=($(printf "%s\n" "${REMAINING_IDS[@]}" | shuf))
TRAIN_IDS=("${SHUFFLED_IDS[@]:0:$TRAIN_COUNT}")
VAL_IDS=("${SHUFFLED_IDS[@]:$TRAIN_COUNT:$VAL_COUNT}")

# === Copy function (case-insensitive match) ===
copy_data() {
    local ID="$1"
    local DEST="$2"

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

    # === Copy alignment dirs
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

# === Print test set cutoff date ===
CUTOFF_DATE="${ID_TO_DATE[${TEST_IDS[0]}]}"
echo "🗓️  Test set cutoff date (most recent 10%): $CUTOFF_DATE"
echo "✅ Dataset successfully split and copied."
