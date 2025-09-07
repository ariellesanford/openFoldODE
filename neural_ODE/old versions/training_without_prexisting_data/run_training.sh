#!/bin/bash
set -e

# =======================================================================================
# TRAINING CONFIGURATION
# =======================================================================================

# Core settings
CPU_ONLY=false
USE_FAST_ODE=true
EPOCHS=25
LEARNING_RATE=1e-3

# Memory optimizations
REDUCED_CLUSTER_SIZE=32
REDUCED_HIDDEN_DIM=32
NUM_TIME_POINTS=5
INTEGRATOR="euler"

# Optimizations
USE_AMP=true
USE_CHECKPOINT=true
MONITOR_MEMORY=true
REDUCED_PRECISION=true

# Optional overrides
TEST_PROTEIN=""  # Set to specific protein ID to test single protein

# =======================================================================================
# SETUP AND EXECUTION
# =======================================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATA_DIR="${SCRIPT_DIR}/mini_data"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/training_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"

# Get python path
PYTHON_PATH="${CONDA_PREFIX:-}/bin/python"
[ ! -f "$PYTHON_PATH" ] && PYTHON_PATH=$(which python)

# Brief configuration summary
echo "🚀 Evoformer ODE Training"
echo "📁 Data: ${DATA_DIR} → Output: ${OUTPUT_DIR}"
echo "⚙️  Config: LR=${LEARNING_RATE}, Epochs=${EPOCHS}, TimePoints=${NUM_TIME_POINTS}"
echo "🧠 Memory: Cluster=${REDUCED_CLUSTER_SIZE}, Hidden=${REDUCED_HIDDEN_DIM}, AMP=${USE_AMP}"
echo "📄 Log: ${OUTPUT_FILE}"
echo ""

# Build command
CMD="${PYTHON_PATH} ${SCRIPT_DIR}/train_evoformer_ode.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --learning_rate ${LEARNING_RATE} \
  --reduced_cluster_size ${REDUCED_CLUSTER_SIZE} \
  --reduced_hidden_dim ${REDUCED_HIDDEN_DIM} \
  --num_time_points ${NUM_TIME_POINTS} \
  --integrator ${INTEGRATOR} \
  --epochs ${EPOCHS}"

# Add conditional flags
[ "${CPU_ONLY}" = true ] && CMD="${CMD} --cpu-only" || CMD="${CMD} --no-cpu-only"
[ "${USE_FAST_ODE}" = true ] && CMD="${CMD} --use_fast_ode" || CMD="${CMD} --no-use_fast_ode"
[ "${USE_AMP}" = true ] && CMD="${CMD} --use_amp" || CMD="${CMD} --no-use_amp"
[ "${USE_CHECKPOINT}" = true ] && CMD="${CMD} --use_checkpoint" || CMD="${CMD} --no-use_checkpoint"
[ "${MONITOR_MEMORY}" = true ] && CMD="${CMD} --monitor_memory" || CMD="${CMD} --no-monitor_memory"
[ "${REDUCED_PRECISION}" = true ] && CMD="${CMD} --reduced_precision_integration" || CMD="${CMD} --no-reduced_precision_integration"
[ -n "${TEST_PROTEIN}" ] && CMD="${CMD} --test-protein ${TEST_PROTEIN}"

# GPU memory check
if [ "${CPU_ONLY}" = false ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "💾 Available GPU Memory: ${GPU_MEM} MB"
    [ "${GPU_MEM}" -lt 4000 ] && echo "⚠️  Warning: Low GPU memory, monitor for OOM errors"
fi

# Execute training
echo "⏳ Starting training..."
echo "Command: ${CMD}" > "${OUTPUT_FILE}"
echo "Started: $(date)" >> "${OUTPUT_FILE}"
echo "========================================" >> "${OUTPUT_FILE}"

eval "${CMD}" 2>&1 | tee -a "${OUTPUT_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

# Summary
echo "" >> "${OUTPUT_FILE}"
echo "Completed: $(date)" >> "${OUTPUT_FILE}"
echo "Exit code: ${EXIT_CODE}" >> "${OUTPUT_FILE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📄 Full log: ${OUTPUT_FILE}"
else
    echo "❌ Training failed (exit code: ${EXIT_CODE})"
    echo "📄 Check log: ${OUTPUT_FILE}"
fi