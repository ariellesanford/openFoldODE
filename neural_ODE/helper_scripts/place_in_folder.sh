#!/bin/bash
set -e

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root


# Find path to required directories
FASTA_DIR="${ROOT_DIR}/data/fasta_data"

# Loop over all FASTA files in the directory
for fasta_file in "${FASTA_DIR}"/*.fasta; do
    echo "Processing file: $fasta_file"

    # Extract the PDB ID (basename without the .fasta extension)
    pdb_id=$(basename "${fasta_file}" .fasta)

    # Create a directory named after the PDB ID
    target_dir="${FASTA_DIR}/${pdb_id}"
    mkdir -p "${target_dir}"

    # Move the FASTA file to the new directory
    mv "${fasta_file}" "${target_dir}/"
done

echo "✅ FASTA files organized into PDB ID folders."