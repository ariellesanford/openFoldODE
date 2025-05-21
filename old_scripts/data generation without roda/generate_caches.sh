#!/bin/bash
set -e

# === CONFIG ===
OF_DIR="/home/visitor/PycharmProjects/openFold/openfold"
DATA_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"
MMCIF_DIR="${DATA_DIR}/training_data(75%)/pdb_data/mmcifs"
SEQUENCE_CLUSTER_FILE="${DATA_DIR}/training_data(75%)/alignment_data/all-seqs_clusters-40.txt"
CACHE_DIR="${DATA_DIR}/training_data(75%)/pdb_data/data_caches"
NUM_WORKERS=4 # You can reduce this if your machine has fewer threads

# === Create cache directory ===
mkdir -p "$CACHE_DIR"

echo "🧠 Generating mmCIF cache..."
python "${OF_DIR}/scripts/generate_mmcif_cache.py" \
    "$MMCIF_DIR" \
    "${CACHE_DIR}/mmcif_cache.json" \
    --no_workers "$NUM_WORKERS"

echo "🔗 Generating chain data cache..."
python "${OF_DIR}/scripts/generate_chain_data_cache.py" \
    "$MMCIF_DIR" \
    "${CACHE_DIR}/chain_data_cache.json" \
    --cluster_file "$SEQUENCE_CLUSTER_FILE" \
    --no_workers "$NUM_WORKERS"

echo "✅ Done generating caches."
