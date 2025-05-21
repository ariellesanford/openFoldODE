#!/bin/bash
set -e

BASE_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"
REL_PATHS=(
    "training_data(75%)/alignment_data/alignments"
    "validation_data(15%)/alignment_data/alignments"
    "testing_data(10%)/alignment_data/alignments"
    "total_dataset/alignment_data/alignments"
)

echo "🔠 Renaming alignment folders to lowercase (first 4 characters)..."

for REL_PATH in "${REL_PATHS[@]}"; do
    ALIGN_DIR="${BASE_DIR}/${REL_PATH}"

    echo "📂 Processing: $ALIGN_DIR"

    cd "$ALIGN_DIR"
    for d in */; do
        folder="${d%/}"  # Remove trailing slash
        prefix="${folder:0:4}"
        rest="${folder:4}"
        lower_prefix=$(echo "$prefix" | tr 'A-Z' 'a-z')
        new_name="${lower_prefix}${rest}"

        if [[ "$folder" != "$new_name" ]]; then
            echo "  ➤ Renaming $folder → $new_name"
            mv "$folder" "$new_name"
        fi
    done
done

echo "✅ All folder names normalized (first 4 characters lowercased)."
