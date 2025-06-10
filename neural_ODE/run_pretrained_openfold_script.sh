#!/bin/bash
set -e

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root
DATA_DIR="/media/visitor/Extreme SSD/data"

# === Define variables based on project root ===
PDB_ID="1tpg_A"
INPUT_FASTA_DIR="${DATA_DIR}/fasta_data/${PDB_ID}"
TEMPLATE_MMCIF_DIR="${DATA_DIR}/template_data/pdb70_mmcif/mmcif_files"
OUTPUT_DIR="${DATA_DIR}/predictions/${PDB_ID}"
PRECOMPUTED_ALIGNMENTS="${DATA_DIR}/alignments"

PDB_ID="6kwc"
INPUT_FASTA_DIR="/home/visitor/PycharmProjects/openFold/openfold/examples/monomer/fasta_dir"
TEMPLATE_MMCIF_DIR="${DATA_DIR}/template_data/pdb70_mmcif/mmcif_files"
OUTPUT_DIR="${DATA_DIR}/predictions/${PDB_ID}"
PRECOMPUTED_ALIGNMENTS="/home/visitor/PycharmProjects/openFold/openfold/examples/monomer/alignments"

# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/openfold

# === Run OpenFold ===
/home/visitor/anaconda3/envs/openfold_env/bin/python run_pretrained_openfold.py \
  "${INPUT_FASTA_DIR}" \
  "${TEMPLATE_MMCIF_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_precomputed_alignments "${PRECOMPUTED_ALIGNMENTS}" \
  --config_preset model_1_ptm \
  --model_device "cuda:0" \
  --save_outputs \
#  --save_intermediates

#  --experiment_config_json "/home/visitor/PycharmProjects/openFold/custom_config.json" \