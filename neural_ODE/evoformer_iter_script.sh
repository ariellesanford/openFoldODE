#!/bin/bash
set -e

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root

# === Define variables based on project root ===
PDB_ID="5i5h_A"
DATA_DIR="${SCRIPT_DIR}/data/training/blocks/${PDB_ID}_evoformer_blocks/recycle_0"
OUTPUT_DIR="${DATA_DIR}"

# Get path to python interpreter
PYTHON_PATH=$(which python)

# === Change to working directory ===
cd "${ROOT_DIR}/evoformer_iter"

# === Determine last existing block index ===
last_m=$(ls "${DATA_DIR}"/m_block_*.pt 2>/dev/null | sed -n 's/.*m_block_\([0-9]\+\)\.pt/\1/p' | sort -n | tail -1)
last_z=$(ls "${DATA_DIR}"/z_block_*.pt 2>/dev/null | sed -n 's/.*z_block_\([0-9]\+\)\.pt/\1/p' | sort -n | tail -1)

if [[ -z "$last_m" || -z "$last_z" || "$last_m" -ne "$last_z" ]]; then
  echo "Error: Could not determine consistent last block index for m and z"
  exit 1
fi

next_idx=$((last_m + 1))

# Construct input and output paths
M_PATH="${DATA_DIR}/m_block_${last_m}.pt"
Z_PATH="${DATA_DIR}/z_block_${last_z}.pt"
NEW_M_PATH="${DATA_DIR}/m_block_${next_idx}.pt"
NEW_Z_PATH="${DATA_DIR}/z_block_${next_idx}.pt"

echo "Running Evoformer iteration $next_idx based on previous index $last_m..."

# Run Evoformer Block
${PYTHON_PATH} run_evoformer_iter.py \
  --m_path "${M_PATH}" \
  --z_path "${Z_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_preset model_1_ptm \
  --device "cuda:0"

# Check if the new files were generated
if [[ ! -f "${NEW_M_PATH}" || ! -f "${NEW_Z_PATH}" ]]; then
  echo "Error: Expected output files not found: ${NEW_M_PATH} or ${NEW_Z_PATH}"
  exit 1
fi

echo "Iteration $next_idx completed successfully."
