#!/bin/bash
set -e

# =======================================================================================
# CONFIGURATION SETTINGS - Modify these variables directly
# =======================================================================================

# Enable CPU-only mode (set to true even if CUDA is available)
CPU_ONLY=false

# Test only a specific protein (leave empty for all proteins, or set a specific name like "1fme_A")
TEST_PROTEIN=""

# Use fast ODE implementation
USE_FAST_ODE=true

# Number of training epochs
EPOCHS=5

# Memory optimizations
MEMORY_SPLIT_SIZE=128      # Memory split size (MB) to avoid fragmentation
REDUCED_CLUSTER_SIZE=96    # Maximum number of MSA clusters to process (original=128)
REDUCED_HIDDEN_DIM=96      # Hidden dimension size for neural networks (original=128)
NUM_TIME_POINTS=25         # Number of integration time points (original=49)
BATCH_SIZE=1               # Number of sequential time steps to process together
INTEGRATOR="rk4"           # ODE solver method (options: rk4, dopri5, euler)
GRADIENT_ACCUMULATION=1    # Number of steps to accumulate gradients over
CHUNK_SIZE=0               # Size of chunks for numerical integration (0 = disable)

# Enable/disable features
USE_AMP=true               # Automatic Mixed Precision (16-bit) for faster training with CUDA
USE_CHECKPOINT=true        # Gradient checkpointing to save memory during backprop
MONITOR_MEMORY=true        # Print memory usage statistics
CLEAN_MEMORY=false         # Aggressively clean GPU memory between steps
REDUCED_PRECISION=false    # Use reduced precision for ODE integration (faster but less accurate)

# =======================================================================================
# END OF CONFIGURATION SETTINGS - No need to modify below this line
# =======================================================================================

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# === Define variables based on project root ===
DATA_DIR="${SCRIPT_DIR}/quick_inference_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

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

# Print configuration settings
echo "=== CONFIGURATION SETTINGS ==="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Python path: ${PYTHON_PATH}"
echo "CPU-only mode: ${CPU_ONLY}"
echo "Test protein: ${TEST_PROTEIN}"
echo "Fast ODE: ${USE_FAST_ODE}"
echo "Epochs: ${EPOCHS}"
echo "============================="

# Build command with all options
CMD="${PYTHON_PATH} ${SCRIPT_DIR}/train_evoformer_ode.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --memory_split_size ${MEMORY_SPLIT_SIZE} \
  --reduced_cluster_size ${REDUCED_CLUSTER_SIZE} \
  --reduced_hidden_dim ${REDUCED_HIDDEN_DIM} \
  --num_time_points ${NUM_TIME_POINTS} \
  --batch_size ${BATCH_SIZE} \
  --integrator ${INTEGRATOR} \
  --gradient_accumulation ${GRADIENT_ACCUMULATION} \
  --chunk_size ${CHUNK_SIZE} \
  --epochs ${EPOCHS}"

# Add test protein flag if specified
if [ -n "${TEST_PROTEIN}" ]; then
    CMD="${CMD} --test-protein ${TEST_PROTEIN}"
    echo "Testing protein: ${TEST_PROTEIN}"
fi

# Add boolean flags based on settings
if [ "${CPU_ONLY}" = true ]; then
    CMD="${CMD} --cpu-only"
    echo "Running in CPU-only mode"
else
    CMD="${CMD} --no-cpu-only"
    echo "Running in auto mode (will use CUDA if available)"
fi

if [ "${USE_FAST_ODE}" = true ]; then
    CMD="${CMD} --use_fast_ode"
else
    CMD="${CMD} --no-use_fast_ode"
fi

if [ "${USE_AMP}" = true ]; then
    CMD="${CMD} --use_amp"
else
    CMD="${CMD} --no-use_amp"
fi

if [ "${USE_CHECKPOINT}" = true ]; then
    CMD="${CMD} --use_checkpoint"
else
    CMD="${CMD} --no-use_checkpoint"
fi

if [ "${MONITOR_MEMORY}" = true ]; then
    CMD="${CMD} --monitor_memory"
else
    CMD="${CMD} --no-monitor_memory"
fi

if [ "${CLEAN_MEMORY}" = true ]; then
    CMD="${CMD} --clean_memory"
else
    CMD="${CMD} --no-clean_memory"
fi

if [ "${REDUCED_PRECISION}" = true ]; then
    CMD="${CMD} --reduced_precision_integration"
else
    CMD="${CMD} --no-reduced_precision_integration"
fi

# Print the final command
echo "Running command:"
echo "${CMD}"
echo

# Execute the command
eval "${CMD}"