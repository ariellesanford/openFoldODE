#!/bin/bash
set -e

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root

# === Define variables based on project root ===
PDB_ID="5fr6_A"
DATA_DIR="${SCRIPT_DIR}/quick_inference_data/${PDB_ID}_evoformer_blocks/recycle_0"
OUTPUT_DIR="${DATA_DIR}"

# Initial paths for m and z
INDEX="0"
M_PATH="${DATA_DIR}/m_block_${INDEX}.pt"
Z_PATH="${DATA_DIR}/z_block_${INDEX}.pt"

# Number of iterations
NUM_ITERATIONS=48

# Get path to python interpreter (use the system's python if not in specific environment)
PYTHON_PATH=$(which python)

# Start total timer
TOTAL_START=$(date +%s)

for (( i=0; i<NUM_ITERATIONS; i++ ))
do
  echo "Running Evoformer iteration $((i+1)) / $NUM_ITERATIONS..."

  # Start timer for current iteration
  START_TIME=$(date +%s)

  # Find the path to run_evoformer_iter.py relative to the project structure
  EVOFORMER_ITER_SCRIPT="${ROOT_DIR}/evoformer_iter/run_evoformer_iter.py"

  # Run Evoformer Block
  ${PYTHON_PATH} "${EVOFORMER_ITER_SCRIPT}" \
    --m_path "${M_PATH}" \
    --z_path "${Z_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --config_preset model_1_ptm \
    --device "cuda:0"

  # Update paths to the newly generated output files
  M_IDX=$(basename "${M_PATH}" .pt | awk -F "_" '{print $3}')
  Z_IDX=$(basename "${Z_PATH}" .pt | awk -F "_" '{print $3}')
  NEW_IDX=$((M_IDX + 1))

  # Update M_PATH and Z_PATH to the next iteration output
  M_PATH="${DATA_DIR}/m_block_${NEW_IDX}.pt"
  Z_PATH="${DATA_DIR}/z_block_${NEW_IDX}.pt"

  # Check if the new files exist
  if [[ ! -f "${M_PATH}" || ! -f "${Z_PATH}" ]]; then
    echo "Error: Expected output files not found: ${M_PATH} or ${Z_PATH}"
    exit 1
  fi
done

# Calculate and display total time taken
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))
echo "✅ Completed ${NUM_ITERATIONS} Evoformer iterations in ${TOTAL_TIME}s."