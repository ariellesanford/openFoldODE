#!/bin/bash
set -e

# Path to root alignment folder containing both chainwise and original folders
ALIGN_ROOT="/home/visitor/Desktop/homoSapien_humoralImmuneResponse(OLD)/total_dataset/alignment_data/alignments"

echo "🔄 Moving chainwise files back into original folders..."

cd "$ALIGN_ROOT"

# Loop through all folders that have an underscore (e.g., 7ZGJ_A)
for chain_folder in *_*; do
  [[ -d "$chain_folder" ]] || continue

  # Extract original PDB ID (first 4 characters) and uppercase
  pdbid=$(echo "$chain_folder" | cut -c1-4 | tr 'a-z' 'A-Z')

  # If no matching target folder exists, skip
  if [[ ! -d "$pdbid" ]]; then
    echo "⚠️  Skipping $chain_folder (no matching folder $pdbid)"
    continue
  fi

  echo "📁 Moving files from $chain_folder → $pdbid"

  # Move all files back into the original folder
  for item in "$chain_folder"/*; do
    [[ -e "$item" ]] || continue
    mv "$item" "$pdbid/"
  done

  # Remove the empty chainwise folder
  rmdir "$chain_folder" && echo "🗑️  Deleted empty folder: $chain_folder"
done

echo "✅ All files returned to original folders."
