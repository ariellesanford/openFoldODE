#!/bin/bash
set -e

# =======================================================================================
# CONFIGURATION SETTINGS - Modify these variables directly
# =======================================================================================

# Enable CPU-only mode (set to true even if CUDA is available)
CPU_ONLY=true

# Test only a specific protein (set to "all" for all proteins, or a specific name like "1fme_A")
TEST_PROTEIN="all"

# Use fast ODE implementation
USE_FAST_ODE=true

# =======================================================================================
# END OF CONFIGURATION SETTINGS - No need to modify below this line
# =======================================================================================

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# === Define variables based on project root ===
DATA_DIR="${SCRIPT_DIR}/quick_inference_data"
OUTPUT_DIR="${SCRIPT_DIR}/config_test_outputs"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Get path to python interpreter (use the system's python if not in specific environment)
if [ -n "${CONDA_PREFIX}" ]; then
    # Use conda environment's python if available
    PYTHON_PATH="${CONDA_PREFIX}/bin/python"
else
    # Otherwise use system python
    PYTHON_PATH=$(which python)
fi

# Print the current configuration
echo "=== CONFIGURATION SETTINGS ==="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Python path: ${PYTHON_PATH}"
echo "CPU-only mode: ${CPU_ONLY}"
echo "Test protein: ${TEST_PROTEIN}"
echo "Fast ODE: ${USE_FAST_ODE}"
echo "============================"

# Add CPU-only flag if needed
CPU_FLAG=""
if [ "$CPU_ONLY" = true ]; then
    CPU_FLAG="--cpu-only"
    echo "Running in CPU-only mode (CUDA will be disabled even if available)"
else
    echo "Running in auto mode (will use CUDA if available)"
fi

# Add fast ODE flag if needed
FAST_ODE_FLAG=""
if [ "$USE_FAST_ODE" = true ]; then
    FAST_ODE_FLAG="--use_fast_ode"
fi

# === Run memory configuration tester ===
${PYTHON_PATH} "${SCRIPT_DIR}/memory_config_tester.py" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test-protein "${TEST_PROTEIN}" \
  --python_path "${PYTHON_PATH}" \
  --script_path "${SCRIPT_DIR}/train_evoformer_ode.py" \
  ${CPU_FLAG} \
  ${FAST_ODE_FLAG}

echo "Configuration testing complete. Report saved to ${OUTPUT_DIR}/memory_optimization_report.txt"