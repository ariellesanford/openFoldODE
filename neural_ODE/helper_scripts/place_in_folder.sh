#!/bin/bash
set -e

# Directory containing the FASTA files
FASTA_DIR="/home/visitor/PycharmProjects/openFold/checkpointing/monomers/fasta_dir"

# Loop over all FASTA files in the directory
for fasta_file in "${FASTA_DIR}"/*.fasta; do
    # Extract the PDB ID (basename without the .fasta extension)
    pdb_id=$(basename "${fasta_file}" .fasta)

    # Create a directory named after the PDB ID
    target_dir="${FASTA_DIR}/${pdb_id}"
    mkdir -p "${target_dir}"

    # Move the FASTA file to the new directory
    mv "${fasta_file}" "${target_dir}/"
done

echo "✅ FASTA files organized into PDB ID folders."
