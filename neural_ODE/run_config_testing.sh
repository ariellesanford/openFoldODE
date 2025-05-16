#!/bin/bash
set -e

# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/neural_ODE

# === Define variables ===
DATA_DIR="/home/visitor/PycharmProjects/openFold/neural_ODE/quick_inference_data/data"
OUTPUT_DIR="/home/visitor/PycharmProjects/openFold/neural_ODE/config_test_outputs"
TEST_PROTEIN="4cue_A_evoformer_blocks"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# === Run memory configuration tester directly ===
/home/visitor/anaconda3/envs/openfold_env/bin/python memory_config_tester.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test-protein "${TEST_PROTEIN}"

echo "Configuration testing complete. Report saved to ${OUTPUT_DIR}/memory_optimization_report.txt"