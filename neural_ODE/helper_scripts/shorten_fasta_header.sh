#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SRC_DIR="${ROOT_DIR}/neural_ODE/data/fasta_data"
DEST_DIR="${ROOT_DIR}/neural_ODE/data/fasta_data"

echo "🔧 Rewriting FASTA headers and replacing destination files..."
echo "📂 Processing source: $SRC_DIR → destination: $DEST_DIR"

for fasta in "$SRC_DIR"/*.fasta; do
    [[ -f "$fasta" ]] || continue

    fname=$(basename "$fasta")
    dest_fasta="${DEST_DIR}/${fname}"

    echo "  ➤ Checking format for: $fname"

    awk '
    BEGIN { OFS = ""; rewriting = 1 }

    NR == 1 && /^>[a-z0-9]{4}_[A-Za-z0-9]$/ {
        rewriting = 0
    }

    {
        if (rewriting == 0) {
            print $0
            next
        }
    }

    rewriting == 1 && /^>/ {
        header = substr($0, 2)
        split(header, parts, "|")
        pdb_chain_raw = parts[1]
        chain_info = parts[2]

        split(pdb_chain_raw, id_parts, "_")
        pdb_id = tolower(id_parts[1])

        chains_str = chain_info
        sub(/^Chains? /, "", chains_str)
        gsub(/,/, "", chains_str)
        gsub(/ /, "", chains_str)

        num_chains = length(chains_str)
        for (i = 1; i <= num_chains; i++) {
            chains[i] = substr(chains_str, i, 1)
        }

        next
    }

    rewriting == 1 {
        for (i = 1; i <= num_chains; i++) {
            print ">", pdb_id, "_", chains[i]
            print $0
        }
    }
    ' "$fasta" > temp && mv temp "$dest_fasta"
done

echo "✅ All FASTA headers rewritten and destination files updated if needed."
