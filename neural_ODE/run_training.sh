#!/bin/bash
set -e

# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/neural_ODE

# === Define variables ===
DATA_DIR="/home/visitor/PycharmProjects/openFold/neural_ODE/data"
OUTPUT_DIR="/home/visitor/PycharmProjects/openFold/neural_ODE/outputs"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# === Run training with all flags visible ===
/home/visitor/anaconda3/envs/openfold_env/bin/python train_evoformer_ode.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --memory_split_size 128 \
  --use_amp \
  --use_checkpoint \
  --gradient_accumulation 4 \
  --chunk_size 10 \
  --reduced_precision_integration \
  --clean_memory \
  --reduced_cluster_size 64 \
  --reduced_hidden_dim 128 \
  --num_time_points 25 \
  --integrator dopri5 \
  --batch_size 5 \
  --monitor_memory
