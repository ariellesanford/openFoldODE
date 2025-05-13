#!/bin/bash
set -e

TOTAL_DIR="/home/visitor/PycharmProjects/openFold/homoSapien_humoralImmuneResponse/total_dataset"
SEQUENCES_DIR="${TOTAL_DIR}/sequences"
ALIGNMENTS_DIR="${TOTAL_DIR}/alignment_data/alignments"

echo "🔍 Searching for FASTA files without corresponding alignments (case-insensitive)..."

# Collect all alignment folders (converted to lowercase, without path)
alignment_ids=$(find "$ALIGNMENTS_DIR" -mindepth 1 -maxdepth 1 -type d | \
  xargs -n1 basename | sed -E 's/_.*//' | tr '[:upper:]' '[:lower:]' | sort -u)

# Loop through all FASTA files
for fasta in "$SEQUENCES_DIR"/*.fasta; do
    id=$(basename "$fasta" .fasta | tr '[:upper:]' '[:lower:]')

    if ! echo "$alignment_ids" | grep -qx "$id"; then
        echo "❌ Missing alignment for: $id"
    fi
done
