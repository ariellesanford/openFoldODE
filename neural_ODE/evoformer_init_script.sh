#!/bin/bash
set -e

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root

# === Define variables based on project root ===
PDB_ID="4d0f_A"
INPUT_FASTA_DIR="${SCRIPT_DIR}/data/fasta_data/${PDB_ID}"
TEMPLATE_MMCIF_DIR="${ROOT_DIR}/openfold/data/pdb70_mmcif/mmcif_files"
OUTPUT_DIR="${SCRIPT_DIR}/data/quick_inference_data"
PRECOMPUTED_ALIGNMENTS="${SCRIPT_DIR}/data/alignments"
# === Change to the desired working directory ===
cd "${ROOT_DIR}/evoformer_init"

# Get path to python interpreter (use the system's python if not in specific environment)
PYTHON_PATH=$(which python)

# === Run OpenFold ===
${PYTHON_PATH} run_evoformer_init.py \
  "${INPUT_FASTA_DIR}" \
  "${TEMPLATE_MMCIF_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_precomputed_alignments "${PRECOMPUTED_ALIGNMENTS}" \
  --config_preset model_1_ptm \
  --model_device "cuda:0" \
  --save_intermediates \
  --save_outputs