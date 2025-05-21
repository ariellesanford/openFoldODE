#!/bin/bash
set -e

# Input: root folder with original messy alignment folders
ALIGN_SRC="/home/visitor/Desktop/homoSapien_humoralImmuneResponse(OLD)/total_dataset/alignment_data/alignments"

# Output: clean, per-chain folder structure
ALIGN_DEST="/home/visitor/Desktop/homoSapien_humoralImmuneResponse/total_dataset/alignment_data/chainwise_alignments"

mkdir -p "$ALIGN_DEST"

echo "📦 Moving _env folders into clean per-chain structure..."

for folder in "$ALIGN_SRC"/*; do
  [[ -d "$folder" ]] || continue
  folder_name=$(basename "$folder")

  # Skip folders that already look like: 5nbq_A or 3utt_C
  if [[ "$folder_name" =~ ^[a-z0-9]{4}_[A-Za-z0-9]+$ ]]; then
    echo "⏭️  Skipping cleaned folder: $folder_name"
    continue
  fi

  echo "🔍 Processing: $folder_name"

  cd "$folder"
  for env_dir in *_env; do
    [[ -d "$env_dir" ]] || continue

    # Extract PDB ID (first 4 characters, lowercased)
    pdbid=$(echo "$env_dir" | cut -c1-4 | tr 'A-Z' 'a-z')

    # Extract all real chain IDs from filename before any "auth"
    chains_raw=$(echo "$env_dir" | grep -oP "Chain[s]?_\K[^_]+" | cut -d_ -f1)
    # Handle multiple chains separated by double underscores
    chains=$(echo "$chains_raw" | tr '__' ' ')

    # If no valid chains found, skip
    [[ -z "$chains" ]] && echo "⚠️  Skipping $env_dir (no valid chain IDs found)" && continue

    for chain in $chains; do
      # One env folder gets copied for each chain
      target="${ALIGN_DEST}/${pdbid}_${chain}"
      echo "  ➤ Moving $env_dir → $target/"
      mkdir -p "$target"
      mv "$env_dir" "$target/" 2>/dev/null || cp -r "$env_dir" "$target/"
    done
  done
done

echo "✅ All valid _env folders moved to clean chainwise folders."
