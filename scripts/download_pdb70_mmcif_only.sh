#!/bin/bash
set -e

# === CONFIG ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root
DOWNLOAD_DIR="${PROJECT_DIR}/openfold/data"
ROOT_DIR="${DOWNLOAD_DIR}/pdb70_mmcif"
RAW_DIR="${ROOT_DIR}/raw"
FLAT_DIR="${ROOT_DIR}/mmcif_files"

# === Step 1: Download compressed mmCIFs via rsync ===
echo "📥 Downloading all mmCIF .gz files via rsync..."
mkdir -p "$RAW_DIR"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
  "$RAW_DIR"

# === Step 2: Flatten to one folder ===
echo "📂 Flattening mmCIF .gz files into $FLAT_DIR..."
mkdir -p "$FLAT_DIR"
find "$RAW_DIR" -type f -name "*.gz" -exec mv -n {} "$FLAT_DIR" \;

# === Step 3: Download and parse the PDB70 list ===
# Use the local PDB70 ffindex file
PDB70_INDEX_PATH="/home/visitor/PycharmProjects/openFold/openfold/data/pdb70/pdb70_a3m.ffindex"
echo "📥 Parsing local PDB70 index: $PDB70_INDEX_PATH"
# Extract the 4-letter PDB IDs (before the first underscore) from each line
cut -f1 "$PDB70_INDEX_PATH" | sed -E 's/^([0-9a-zA-Z]{4})_.*/\1/' | tr '[:upper:]' '[:lower:]' | sort -u > "${ROOT_DIR}/pdb70_ids.txt"
echo "✅ Found $(wc -l < "${ROOT_DIR}/pdb70_ids.txt") unique PDB IDs in PDB70"

# === Step 4: Delete all .cif.gz files not in PDB70 ===
echo "🧹 Removing .cif.gz files not in PDB70..."
cd "$FLAT_DIR"
for file in *.cif.gz; do
    pdbid="${file%%.cif.gz}"
    if ! grep -qx "$pdbid" "${ROOT_DIR}/pdb70_ids.txt"; then
        rm "$file"
    fi
done

# === Step 5: Unzip remaining PDB70 mmCIFs ===
echo "📦 Unzipping remaining PDB70 .cif.gz files..."
gunzip *.cif.gz

echo "✅ Done! All PDB70 mmCIFs downloaded and extracted to:"
echo "$FLAT_DIR"
