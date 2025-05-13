#!/bin/bash
set -e

# === Input root ===
ALIGN_ROOT="/home/visitor/Desktop/homoSapien_humoralImmuneResponse/total_dataset/alignment_data/alignments"

if [[ ! -d "$ALIGN_ROOT" ]]; then
  echo "Usage: $0 /path/to/alignment_data/alignments"
  exit 1
fi

# === Iterate over each protein folder ===
for folder in "$ALIGN_ROOT"/*; do
  [[ -d "$folder" ]] || continue  # skip files

  echo "🔄 Processing: $(basename "$folder")"
  cd "$folder"

  for file in *Chain_*; do
    [[ -f "$file" ]] || continue

    # Extract base ID and chain ID
    base=$(echo "$file" | sed -E 's/^([0-9A-Za-z]+)_[0-9]+_Chain_([A-Za-z0-9]+)_.*/\1_\2/')
    pdbid=$(echo "$base" | cut -d_ -f1)
    chain=$(echo "$base" | cut -d_ -f2)

    # Create destination folder like 1A6A_A
    dest_folder="${ALIGN_ROOT}/${pdbid}_${chain}"
    mkdir -p "$dest_folder"

    # Move all files that match this chain
    chain_prefix="${pdbid}_"*"Chain_${chain}"*
    for match in $chain_prefix; do
      [[ -f "$match" ]] && mv "$match" "$dest_folder/"
    done
  done

  # Optional: delete original folder if empty
  rmdir "$folder" 2>/dev/null || true
done

echo "✅ Restructuring complete."
