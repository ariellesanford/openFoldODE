#!/bin/bash
set -e

# === Paths ===
BASE_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"
TOTAL_DIR="${BASE_DIR}/total_dataset"
TRAIN_DIR="${BASE_DIR}/training_data(75%)"
VAL_DIR="${BASE_DIR}/validation_data(15%)"
TEST_DIR="${BASE_DIR}/testing_data(10%)"
MMCIF_DIR="${TOTAL_DIR}/pdb_data/mmcifs"

# === Check mmCIF directory exists ===
if [ ! -d "$MMCIF_DIR" ]; then
    echo "❌ Error: MMCIF directory not found at $MMCIF_DIR"
    exit 1
fi

# === Ensure output directories exist ===
for OUTDIR in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
    mkdir -p "${OUTDIR}/pdb_data/mmcifs"
    mkdir -p "${OUTDIR}/alignment_data/alignments"
    mkdir -p "${OUTDIR}/sequences"
done

# === Step 1: Collect all .cif files and extract release dates ===
declare -A ID_TO_DATE
CIF_FILES=($(find "$MMCIF_DIR" -name "*.cif"))
echo "🔍 Found ${#CIF_FILES[@]} .cif files"

for cif_file in "${CIF_FILES[@]}"; do
    filename=$(basename "$cif_file")
    id="${filename%.*}"

    # Extract release date via inline Python
    release_date=$(python3 -c "
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
try:
    d = MMCIF2Dict('$cif_file')
    dates = d.get('_pdbx_audit_revision_history.revision_date', [])
    if isinstance(dates, list): print(min(dates))
    else: print(dates)
except Exception as e:
    print('error')
")

    if [[ "$release_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        ID_TO_DATE["$id"]="$release_date"
    else
        echo "⚠️  Skipping $id — no valid release date found"
    fi
done

# === Step 2: Sort proteins by date ===
SORTED_IDS=($(for id in "${!ID_TO_DATE[@]}"; do
    echo "$id ${ID_TO_DATE[$id]}"
done | sort -k2 | awk '{print $1}'))

TOTAL_COUNT=${#SORTED_IDS[@]}
TEST_COUNT=$((TOTAL_COUNT * 10 / 100))
VAL_COUNT=$((TOTAL_COUNT * 15 / 100))
TRAIN_COUNT=$((TOTAL_COUNT - TEST_COUNT - VAL_COUNT))

# === Split datasets ===
TEST_IDS=("${SORTED_IDS[@]: -$TEST_COUNT}")
REMAINING_IDS=("${SORTED_IDS[@]:0:$((TOTAL_COUNT - TEST_COUNT))}")
SHUFFLED_REMAINING=($(printf "%s\n" "${REMAINING_IDS[@]}" | shuf))

TRAIN_IDS=("${SHUFFLED_REMAINING[@]:0:$TRAIN_COUNT}")
VAL_IDS=("${SHUFFLED_REMAINING[@]:$TRAIN_COUNT:$VAL_COUNT}")

# === Print test cutoff date ===
CUTOFF_ID="${TEST_IDS[0]}"
CUTOFF_DATE="${ID_TO_DATE[$CUTOFF_ID]}"
echo "📅 Test set cutoff date: $CUTOFF_DATE"

# === Copy function ===
copy_data() {
    local ID="$1"
    local DEST="$2"

    # Copy .fasta
    fasta_file=$(find "${TOTAL_DIR}/sequences" -iname "${ID}.fasta" | head -n 1)
    if [[ -n "$fasta_file" ]]; then
        cp "$fasta_file" "${DEST}/sequences/"
    else
        echo "❗ Missing .fasta for $ID"
    fi

    # Copy .cif
    cif_file=$(find "$MMCIF_DIR" -iname "${ID}.cif" | head -n 1)
    if [[ -n "$cif_file" ]]; then
        cp "$cif_file" "${DEST}/pdb_data/mmcifs/"
    else
        echo "❗ Missing .cif for $ID"
    fi

    # Copy alignments
    align_dir="${TOTAL_DIR}/alignment_data/alignments"
    for d in $(find "$align_dir" -maxdepth 1 -type d -iname "${ID}_*"); do
        cp -r "$d" "${DEST}/alignment_data/alignments/"
    done
}

# === Copy files ===
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

echo "✅ Dataset split completed using date-aware logic."
