#!/bin/bash
set -e

# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/neural_ODE

# === Define variables ===
DATA_DIR="/home/visitor/PycharmProjects/openFold/neural_ODE/quick_inference_data"
OUTPUT_DIR="/home/visitor/PycharmProjects/openFold/neural_ODE/outputs"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "Using data directory: ${DATA_DIR}"
echo "Using output directory: ${OUTPUT_DIR}"

# === Run training with memory optimized baseline settings ===
/home/visitor/anaconda3/envs/openfold_env/bin/python train_evoformer_ode.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --memory_split_size 128 \
  --reduced_cluster_size 96 \
  --reduced_hidden_dim 96 \
  --num_time_points 25 \
  --batch_size 1 \
  --integrator rk4 \
  --gradient_accumulation 1 \
  --chunk_size 0 \
  --use_amp \
  --use_checkpoint \
  --use_fast_ode \
  --monitor_memory

# === Description of settings ===
# --memory_split_size 128      : Maximum CUDA memory block size (MB) to avoid fragmentation
# --reduced_cluster_size 128   : Maximum number of MSA clusters to process (original=128)
# --reduced_hidden_dim 128     : Hidden dimension size for neural networks (original=128)
# --num_time_points 49         : Number of integration points in time (original=49)
#
# --batch_size 1               : Number of sequential time steps to process together in one forward pass
#                                Unlike traditional batching, this doesn't process multiple independent samples,
#                                but rather chunks of the continuous time trajectory. Smaller values use less
#                                memory but may require more computation overall.
#
# --integrator rk4             : ODE solver method (options: rk4, dopri5, euler)
#
# --gradient_accumulation 1    : Number of batches to accumulate gradients over before updating model weights.
#                                This allows effective training with larger batch sizes while using less memory.
#                                Value of 1 means update weights after each batch.
#
# --chunk_size 0               : Size of chunks for numerical integration within each time step batch.
#                                When > 0, breaks integration within each batch into smaller chunks to save memory.
#                                These chunks are processed sequentially to reduce peak memory usage.
#                                Value of 0 means no chunking is used.
#
# --use_amp                    : Enable Automatic Mixed Precision (16-bit) for faster training
# --use_checkpoint             : Enable gradient checkpointing to save memory during backprop
# --monitor_memory             : Print memory usage statistics during training
# --epochs 5                   : Number of complete passes through the training dataset
# --use_fast_ode              : Use faster implementation of EvoformerODEFunc

# === Disabled optimizations (add if needed) ===
# --reduced_precision_integration  : Use reduced precision for ODE integration (faster but less accurate)
# --clean_memory                   : Aggressively clean GPU memory between steps