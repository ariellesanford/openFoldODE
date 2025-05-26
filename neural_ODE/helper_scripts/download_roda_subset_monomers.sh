#!/bin/bash
#
# Download and flatten RODA alignment data from S3 and mmCIFs from PDB.org
# Usage: bash download_roda_subset.sh

set -e

# === CONFIG ===
#RAW_LIST='["1H2P_A", "1H2Q_A", "1K58_A", "1LFH_A", "1MD7_A", "1MD8_A", "1N76_A", "1NUH_A", "1RS0_A", "1SK3_A",
#    "1UN5_A", "1UOT_A", "2B3O_A", "2O84_A", "2QZH_A", "2VJ3_A", "3GPD_A", "3J24_A", "3PS5_A", "3QYT_A",
#    "3U3Q_A", "3U3S_A", "3U3T_A", "3U3V_A", "41BI_A", "4AHI_A", "4AHN_A", "4CUE_A", "4D0F_A", "4J6Q_A",
#    "4LOT_A", "4N4X_A", "4NZQ_A", "4O03_A", "5FM9_A", "5FN6_A", "5FR6_A", "5H52_A", "5T1A_A", "5TCX_A",
#    "5WTD_A", "5X5P_A", "5Y6K_A", "6A7U_A", "6CTC_A", "6GPS_A", "6JAS_A", "6MV7_A", "6T8U_A", "6UJ6_A",
#    "6YNH_A", "7F1T_A", "7FFM_A", "7FFU_A", "7PNJ_A", "7VON_A", "8BRC_A", "8DNS_A", "8EX4_A", "8EX5_A",
#    "8EX6_A", "8EX7_A", "8EX8_A", "8G92_A", "8JHQ_A", "8JHR_A", "8RL8_A", "8RLD_A", "8UQC_A", "8UQD_A",
#    "8UQE_A", "9F6Q_A", "9F6P_A", "8WZS_A"]'

RAW_LIST='["7d60_A","3foa_A","3eps_A"]'
#RAW_LIST='["7ok6_AAA","7bl2_6","3jbn_Ab","1bho_1"]'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BASE_DIR="${ROOT_DIR}/neural_ODE/data"
RODA_TMP="${BASE_DIR}/alignment_dir_roda_subset"
ALIGNMENT_OUT="${BASE_DIR}"
FASTA_DIR="${BASE_DIR}/fasta_data"
FLATTEN_SCRIPT="${ROOT_DIR}/openfold/scripts/flatten_roda.sh"

# === Extract and normalize PDB_CHAIN IDs (lowercase pdb ID, uppercase chain) ===
PDB_IDS=$(echo "$RAW_LIST" | sed -E 's/[^A-Za-z0-9_]+/\n/g' | grep -E '^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]+$' | sort -u)

# === Normalize: lowercase PDB ID, preserve chain ===
NORMALIZED_IDS=()
for pdbid in $PDB_IDS; do
    pdb_part=$(echo "$pdbid" | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]')
    chain_part=$(echo "$pdbid" | cut -d'_' -f2)
    normalized="${pdb_part}_${chain_part}"
    NORMALIZED_IDS+=("$normalized")
done

# === Create directories ===
mkdir -p "$RODA_TMP"
mkdir -p "$FASTA_DIR"

echo "📥 Downloading RODA alignment data from OpenFold S3..."

for pdbid in "${NORMALIZED_IDS[@]}"; do
    echo "➡️  Downloading alignments for $pdbid"
    aws s3 cp "s3://openfold/pdb/${pdbid}/" "${RODA_TMP}/${pdbid}/" --recursive --no-sign-request
done


# === Download FASTAs only if alignment data exists ===
echo "📥 Downloading FASTA files from RCSB PDB (only for proteins with alignment data)..."

for pdbid in "${NORMALIZED_IDS[@]}"; do
    alignment_path="${RODA_TMP}/${pdbid}"
    if [ "$(ls -A "$alignment_path" 2>/dev/null)" ]; then
        pdb_code=$(echo "$pdbid" | cut -d'_' -f1)
        fasta_url="https://www.rcsb.org/fasta/entry/${pdb_code^^}"
        output_path="${FASTA_DIR}/${pdbid}.fasta"

        echo "➡️  Downloading FASTA for $pdbid from $fasta_url"
        env -i /usr/bin/curl -s -f "$fasta_url" -o "$output_path" || echo "⚠️  Failed to download FASTA for $pdbid"
    else
        echo "⏭  Skipping $pdbid — no alignment folder found, so no FASTA download."
    fi
done


# === Flatten ===
echo "🛠 Flattening RODA alignment structure..."
bash "$FLATTEN_SCRIPT" "$RODA_TMP" "$ALIGNMENT_OUT"

echo "🧹 Cleaning up temporary alignment folder..."
rm -rf "$RODA_TMP"

echo "✅ Done!"
echo "   Alignments → $ALIGNMENT_OUT/alignments/"
echo "   Fastas → $FASTA_DIR/"
