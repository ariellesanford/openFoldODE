#!/bin/bash
set -e

# =======================================================================================
# CONFIGURATION SETTINGS - Modify these variables directly
# =======================================================================================

# Use CUDA if available (set to false to force CPU)
USE_CUDA=true

# Root directory (project root)
ROOT_DIR="/home/visitor/PycharmProjects/openFold"

# Data directory
DATA_DIR="/media/visitor/Extreme SSD/data"

# Protein ID to process (or "all" for all available proteins)
PDB_ID="1fv5_A"  # Set to "all" to process all proteins in evoformer outputs

# Paths based on your structure
FASTA_DIR="${DATA_DIR}/fasta_data/${PDB_ID}"  # Directory containing .fasta files
TEMPLATE_MMCIF_DIR="${DATA_DIR}/template_data/pdb70_mmcif/mmcif_files" # Template structures
EVOFORMER_OUTPUTS_DIR="${ROOT_DIR}/neural_ODE/post_evoformer_predictions/predictions_20250605_183403"  # Neural ODE outputs
OUTPUT_DIR="${ROOT_DIR}/neural_ODE/structure_predictions"  # Where to save structures
PRECOMPUTED_ALIGNMENTS="${DATA_DIR}/alignments"

# Model settings
CONFIG_PRESET="model_1_ptm"  # OpenFold model preset
MODEL_DEVICE="cuda:0"  # GPU device (will be set to 'cpu' if USE_CUDA=false)

# Optional settings
SKIP_RELAXATION=false  # Set to true to skip amber relaxation (faster)
CIF_OUTPUT=false  # Set to true for ModelCIF format instead of PDB
SAVE_OUTPUTS=false  # Set to true to save intermediate outputs
LONG_SEQUENCE_INFERENCE=false  # Set to true for long sequence optimizations
USE_DEEPSPEED_ATTENTION=false  # Set to true to use DeepSpeed attention

# =======================================================================================
# END OF CONFIGURATION SETTINGS - No need to modify below this line
# =======================================================================================

# Change to structure module working directory
cd "${ROOT_DIR}/structure_module"

# Set device
if [ "$USE_CUDA" = false ] || ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    MODEL_DEVICE="cpu"
fi

# Get path to python interpreter
if [ -n "${CONDA_PREFIX}" ]; then
    PYTHON_PATH="${CONDA_PREFIX}/bin/python"
else
    PYTHON_PATH=$(which python)
fi

echo "========================================"
echo "Structure Module Processing Configuration"
echo "========================================"
echo "Root Directory: ${ROOT_DIR}"
echo "FASTA Directory: ${FASTA_DIR}"
echo "Evoformer Outputs: ${EVOFORMER_OUTPUTS_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Model Device: ${MODEL_DEVICE}"
echo "Config Preset: ${CONFIG_PRESET}"
echo "Protein ID: ${PDB_ID}"
echo "========================================"

# Build command
CMD=(
    "${PYTHON_PATH}"
    "run_structure_module.py"
    "${FASTA_DIR}"
    "${TEMPLATE_MMCIF_DIR}"
    "--evoformer_outputs_dir" "${EVOFORMER_OUTPUTS_DIR}"
    "--output_dir" "${OUTPUT_DIR}"
    "--protein_id" "${PDB_ID}"
    "--model_device" "${MODEL_DEVICE}"
    "--config_preset" "${CONFIG_PRESET}"
)

# Add optional arguments
if [ -n "${PRECOMPUTED_ALIGNMENTS}" ] && [ -d "${PRECOMPUTED_ALIGNMENTS}" ]; then
    CMD+=("--use_precomputed_alignments" "${PRECOMPUTED_ALIGNMENTS}")
fi

if [ "$SKIP_RELAXATION" = true ]; then
    CMD+=("--skip_relaxation")
fi

if [ "$CIF_OUTPUT" = true ]; then
    CMD+=("--cif_output")
fi

if [ "$SAVE_OUTPUTS" = true ]; then
    CMD+=("--save_outputs")
fi

if [ "$LONG_SEQUENCE_INFERENCE" = true ]; then
    CMD+=("--long_sequence_inference")
fi

if [ "$USE_DEEPSPEED_ATTENTION" = true ]; then
    CMD+=("--use_deepspeed_evoformer_attention")
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "Running command: ${CMD[*]}"
echo "========================================"

# Run the command
"${CMD[@]}"

echo "========================================"
echo "Structure module processing completed!"
echo "Check ${OUTPUT_DIR} for results"
echo "========================================"