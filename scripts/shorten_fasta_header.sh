#!/bin/bash
set -e

# Source (original FASTAs to fix)
SOURCE_BASE="/home/visitor/Desktop/homoSapien_humoralImmuneResponse(OLD)"

# Destination (where fixed FASTAs go)
DEST_BASE="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"

REL_PATHS=(
    "total_dataset/sequences"
    "training_data(75%)/sequences"
    "validation_data(15%)/sequences"
    "testing_data(10%)/sequences"
)

echo "🔧 Rewriting FASTA headers and replacing destination files..."

for REL_PATH in "${REL_PATHS[@]}"; do
    SRC_DIR="${SOURCE_BASE}/${REL_PATH}"
    DEST_DIR="${DEST_BASE}/${REL_PATH}"

    echo "📂 Processing source: $SRC_DIR → destination: $DEST_DIR"

    for fasta in "$SRC_DIR"/*.fasta; do
        [[ -f "$fasta" ]] || continue

        fname=$(basename "$fasta")
        dest_fasta="${DEST_DIR}/${fname}"

        echo "  ➤ Rewriting: $fname"

        awk '
        BEGIN { OFS = "" }
        /^>/ {
            header = substr($0, 2)
            split(header, parts, "|")
            pdb_chain_raw = parts[1]
            chain_info = parts[2]

            # Lowercase the PDB ID
            split(pdb_chain_raw, id_parts, "_")
            pdb_id = tolower(id_parts[1])

            # Extract chain letters from "Chains D, E" or "Chain A"
            match(chain_info, /Chain[s]* ([A-Za-z0-9, ]+)/, m)
            chains_str = m[1]
            gsub(/,/, "", chains_str)  # Remove commas
            gsub(/ /, "", chains_str)  # Remove spaces

            # Save all chains for multi-line output
            num_chains = length(chains_str)
            for (i = 1; i <= num_chains; i++) {
                chains[i] = substr(chains_str, i, 1)
            }

            next
        }
        {
            # For each chain, output a header and sequence
            for (i = 1; i <= num_chains; i++) {
                print ">", pdb_id, "_", chains[i]
                print $0
            }
        }
        ' "$fasta" > temp && mv temp "$dest_fasta"
    done
done

echo "✅ All FASTA headers rewritten and destination files updated."
