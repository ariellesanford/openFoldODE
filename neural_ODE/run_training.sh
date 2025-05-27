#!/bin/bash
set -e

# =======================================================================================
# MEMORY-OPTIMIZED CONFIGURATION SETTINGS
# =======================================================================================

# Enable CPU-only mode (set to true if you want to force CPU even with CUDA available)
CPU_ONLY=false
TEST_PROTEIN=""

# Use fast ODE implementation (recommended for memory efficiency)
USE_FAST_ODE=true

# Training settings
EPOCHS=25
LEARNING_RATE=1e-3  # Start with 1e-5 for stability, can increase to 1e-4 or 1e-3 if stable

# Memory optimizations (keep your existing settings)
MEMORY_SPLIT_SIZE=128
REDUCED_CLUSTER_SIZE=32  # Start small for stability
REDUCED_HIDDEN_DIM=32    # Start small for stability
NUM_TIME_POINTS=5        # Start small for stability
BATCH_SIZE=1
INTEGRATOR="euler"       # Most stable
GRADIENT_ACCUMULATION=1
CHUNK_SIZE=0

# Enable/disable features
USE_AMP=true
USE_CHECKPOINT=true
MONITOR_MEMORY=true
CLEAN_MEMORY=false
REDUCED_PRECISION=true   # For stability

# =======================================================================================
# AUTO-SAVE OUTPUT SETUP
# =======================================================================================

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# === Define variables based on project root ===
DATA_DIR="${SCRIPT_DIR}/mini_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Create timestamped output filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/training_output_${TIMESTAMP}.txt"

# Get path to python interpreter
if [ -n "${CONDA_PREFIX}" ]; then
    PYTHON_PATH="${CONDA_PREFIX}/bin/python"
else
    PYTHON_PATH=$(which python)
fi

# Print configuration settings
# Print configuration settings
echo "=== TRAINING CONFIGURATION ===" | tee "${OUTPUT_FILE}"
echo "📁 Directories:" | tee -a "${OUTPUT_FILE}"
echo "   Data directory: ${DATA_DIR}" | tee -a "${OUTPUT_FILE}"
echo "   Output directory: ${OUTPUT_DIR}" | tee -a "${OUTPUT_FILE}"
echo "   Python path: ${PYTHON_PATH}" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"
echo "⚙️  Training settings:" | tee -a "${OUTPUT_FILE}"
echo "   CPU-only mode: ${CPU_ONLY}" | tee -a "${OUTPUT_FILE}"
echo "   Test protein: ${TEST_PROTEIN:-"All proteins"}" | tee -a "${OUTPUT_FILE}"
echo "   Fast ODE: ${USE_FAST_ODE}" | tee -a "${OUTPUT_FILE}"
echo "   Epochs: ${EPOCHS}" | tee -a "${OUTPUT_FILE}"
echo "   Learning rate: ${LEARNING_RATE}" | tee -a "${OUTPUT_FILE}"
echo "   Integrator: ${INTEGRATOR}" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"
echo "🧠 Memory settings:" | tee -a "${OUTPUT_FILE}"
echo "   Cluster size: ${REDUCED_CLUSTER_SIZE}" | tee -a "${OUTPUT_FILE}"
echo "   Hidden dim: ${REDUCED_HIDDEN_DIM}" | tee -a "${OUTPUT_FILE}"
echo "   Time points: ${NUM_TIME_POINTS}" | tee -a "${OUTPUT_FILE}"
echo "   Memory cleaning: ${CLEAN_MEMORY}" | tee -a "${OUTPUT_FILE}"
echo "   Mixed precision (AMP): ${USE_AMP}" | tee -a "${OUTPUT_FILE}"
echo "   Checkpointing: ${USE_CHECKPOINT}" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"
echo "🧹 File cleanup: ENABLED (auto-cleanup generated iteration files)" | tee -a "${OUTPUT_FILE}"
echo "   Output will be saved to: ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"
echo "=============================" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"

# Check available GPU memory before starting
if [ "${CPU_ONLY}" = false ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "🔍 GPU Memory Check:" | tee -a "${OUTPUT_FILE}"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r used total; do
        # Remove any whitespace
        used=$(echo "$used" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')
        available=$((total - used))
        echo "   Available GPU Memory: ${available} MB / ${total} MB" | tee -a "${OUTPUT_FILE}"
        if [ "${available}" -lt 4000 ]; then
            echo "   ⚠️  WARNING: Available GPU memory is ${available} MB" | tee -a "${OUTPUT_FILE}"
            echo "   With your current settings (cluster_size=${REDUCED_CLUSTER_SIZE}, hidden_dim=${REDUCED_HIDDEN_DIM})," | tee -a "${OUTPUT_FILE}"
            echo "   you may still encounter OOM errors. Monitor memory usage closely." | tee -a "${OUTPUT_FILE}"
        fi
    done
    echo "" | tee -a "${OUTPUT_FILE}"
fi

# Build command with all options
CMD="${PYTHON_PATH} ${SCRIPT_DIR}/train_evoformer_ode.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --learning_rate ${LEARNING_RATE} \
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
fi

# Add boolean flags based on settings
if [ "${CPU_ONLY}" = true ]; then
    CMD="${CMD} --cpu-only"
else
    CMD="${CMD} --no-cpu-only"
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
echo "🚀 Running memory-optimized training command:" | tee -a "${OUTPUT_FILE}"
echo "${CMD}" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"

# Check if we're using the optimized training script
if [ ! -f "${SCRIPT_DIR}/train_evoformer_ode.py" ]; then
    echo "❌ ERROR: train_evoformer_ode.py not found!" | tee -a "${OUTPUT_FILE}"
    echo "Please make sure you've replaced the training script with the memory-optimized version." | tee -a "${OUTPUT_FILE}"
    exit 1
fi

# Final memory warning and confirmation
echo "🚀 Ready to start training with automatic file cleanup!" | tee -a "${OUTPUT_FILE}"
echo "✅ Your settings will be used:" | tee -a "${OUTPUT_FILE}"
echo "   - Cluster size: ${REDUCED_CLUSTER_SIZE} (your preferred setting)" | tee -a "${OUTPUT_FILE}"
echo "   - Hidden dimensions: ${REDUCED_HIDDEN_DIM} (your preferred setting)" | tee -a "${OUTPUT_FILE}"
echo "   - Time points: ${NUM_TIME_POINTS} (your preferred setting)" | tee -a "${OUTPUT_FILE}"
echo "   - Checkpointing: ENABLED (your preferred setting)" | tee -a "${OUTPUT_FILE}"
echo "   - Memory cleaning: DISABLED (your preferred setting)" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"
echo "🧹 AUTOMATIC CLEANUP:" | tee -a "${OUTPUT_FILE}"
echo "   - Generated iteration files will be automatically deleted during training" | tee -a "${OUTPUT_FILE}"
echo "   - Only initial block_0 files will be kept" | tee -a "${OUTPUT_FILE}"
echo "   - This saves disk space without affecting training quality" | tee -a "${OUTPUT_FILE}"
echo "" | tee -a "${OUTPUT_FILE}"

# Execute the command and save ALL output to file
echo "Training started at: $(date)" | tee -a "${OUTPUT_FILE}"
echo "======================================" | tee -a "${OUTPUT_FILE}"

# Run the command and capture ALL output
eval "${CMD}" 2>&1 | tee -a "${OUTPUT_FILE}"

# Capture the exit code
EXIT_CODE=${PIPESTATUS[0]}

# Add completion timestamp
echo "" | tee -a "${OUTPUT_FILE}"
echo "======================================" | tee -a "${OUTPUT_FILE}"
echo "Training completed at: $(date)" | tee -a "${OUTPUT_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${OUTPUT_FILE}"
echo "Full output saved to: ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"

# Print final status
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code ${EXIT_CODE}"
    echo "Check the output file for details: ${OUTPUT_FILE}"
fi