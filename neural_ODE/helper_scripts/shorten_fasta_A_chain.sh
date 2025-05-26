#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SRC_DIR="${ROOT_DIR}/neural_ODE/data/fasta_data"
DEST_DIR="${ROOT_DIR}/neural_ODE/data/fasta_data"

echo "🔧 Keeping only chain A from FASTA files..."
echo "📂 Processing source: $SRC_DIR → destination: $DEST_DIR"

for fasta in "$SRC_DIR"/*.fasta; do
    [[ -f "$fasta" ]] || continue

    fname=$(basename "$fasta")
    dest_fasta="${DEST_DIR}/${fname}"

    echo "  ➤ Filtering: $fname"

    awk '
    BEGIN { keep = 0 }
    /^>/ {
        if (tolower($0) ~ /^>...._a$/) {
            keep = 1
        } else {
            keep = 0
        }
    }
    {
        if (keep) print $0
    }
    ' "$fasta" > temp && mv temp "$dest_fasta"
done

echo "✅ Chain A retained; all other chains removed."
