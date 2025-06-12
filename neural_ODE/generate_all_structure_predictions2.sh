#!/bin/bash
set -e

# =======================================================================================
# MULTI-METHOD STRUCTURE PREDICTION SCRIPT
# Generates structure predictions using 3 different methods for a given PDB_ID
# =======================================================================================

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="/home/visitor/PycharmProjects/openFold"
DATA_DIR="/media/visitor/Extreme SSD/data"

# =======================================================================================
# CONFIGURATION - Modify these variables
# =======================================================================================

# Protein ID to process (REQUIRED - set this!)
PDB_ID="6l8f_A"  # Change this to your desired protein

# Use CUDA if available (set to false to force CPU)
USE_CUDA=true

# Model settings
CONFIG_PRESET="model_1_ptm"
MODEL_DEVICE="cuda:0"

# Optional settings
SKIP_RELAXATION=false
CIF_OUTPUT=false
SAVE_OUTPUTS=false
LONG_SEQUENCE_INFERENCE=false
USE_DEEPSPEED_ATTENTION=false

# =======================================================================================
# DERIVED PATHS - These are automatically set based on PDB_ID
# =======================================================================================

# Common paths
FASTA_DIR="${DATA_DIR}/fasta_data/${PDB_ID}"
TEMPLATE_MMCIF_DIR="${DATA_DIR}/template_data/pdb70_mmcif/mmcif_files"
PRECOMPUTED_ALIGNMENTS="${DATA_DIR}/alignments"

# Method 1: Neural ODE predictions (will be set dynamically)
NEURAL_ODE_PREDICTIONS_DIR=""
NEURAL_ODE_MSA_PATH=""
NEURAL_ODE_PAIR_PATH=""
NEURAL_ODE_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE"

# Method 2: OpenFold Deconstructed (48th evoformer block)
OPENFOLD_DECON_MSA_PATH="${DATA_DIR}/complete_blocks/${PDB_ID}_evoformer_blocks/recycle_0/m_block_48.pt"
OPENFOLD_DECON_PAIR_PATH="${DATA_DIR}/complete_blocks/${PDB_ID}_evoformer_blocks/recycle_0/z_block_48.pt"
OPENFOLD_DECON_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_deconstructed"

# Method 3: Full OpenFold
OPENFOLD_FULL_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_0recycles"

# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================

# Set device based on configuration
set_device() {
    if [ "$USE_CUDA" = false ] || ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
        MODEL_DEVICE="cpu"
    fi
}

# Get python interpreter
get_python_path() {
    if [ -n "${CONDA_PREFIX}" ]; then
        echo "${CONDA_PREFIX}/bin/python"
    else
        echo $(which python)
    fi
}

# Find the most recent Neural ODE predictions directory
find_latest_neural_ode_predictions() {
    local base_dir="${DATA_DIR}/post_evoformer_predictions"

    if [ ! -d "$base_dir" ]; then
        echo "❌ Neural ODE predictions base directory not found: $base_dir"
        return 1
    fi

    # Find all prediction directories matching pattern predictions_YYYYMMDD_HHMMSS
    local latest_dir=""
    local latest_timestamp=""

    for dir in "$base_dir"/predictions_*; do
        if [ -d "$dir" ]; then
            # Extract timestamp from directory name (predictions_20250605_183403 -> 20250605_183403)
            local timestamp=$(basename "$dir" | sed 's/predictions_//')

            # Check if this directory contains our protein
            if [ -d "$dir/$PDB_ID" ]; then
                # Compare timestamps (string comparison works for YYYYMMDD_HHMMSS format)
                if [ -z "$latest_timestamp" ] || [ "$timestamp" \> "$latest_timestamp" ]; then
                    latest_timestamp="$timestamp"
                    latest_dir="$dir"
                fi
            fi
        fi
    done

    if [ -n "$latest_dir" ]; then
        echo "🔍 Found latest Neural ODE predictions: $(basename "$latest_dir")"
        NEURAL_ODE_PREDICTIONS_DIR="$latest_dir/$PDB_ID"
        NEURAL_ODE_MSA_PATH="$NEURAL_ODE_PREDICTIONS_DIR/msa_representation.pt"
        NEURAL_ODE_PAIR_PATH="$NEURAL_ODE_PREDICTIONS_DIR/pair_representation.pt"
        return 0
    else
        echo "❌ No Neural ODE predictions found for $PDB_ID in $base_dir"
        return 1
    fi
}

# Add common optional arguments to a command array
add_optional_args() {
    local -n cmd_ref=$1  # Pass array by reference

    if [ -n "${PRECOMPUTED_ALIGNMENTS}" ] && [ -d "${PRECOMPUTED_ALIGNMENTS}" ]; then
        cmd_ref+=("--use_precomputed_alignments" "${PRECOMPUTED_ALIGNMENTS}")
    fi

    if [ "$SKIP_RELAXATION" = true ]; then
        cmd_ref+=("--skip_relaxation")
    fi

    if [ "$CIF_OUTPUT" = true ]; then
        cmd_ref+=("--cif_output")
    fi

    if [ "$SAVE_OUTPUTS" = true ]; then
        cmd_ref+=("--save_outputs")
    fi

    if [ "$LONG_SEQUENCE_INFERENCE" = true ]; then
        cmd_ref+=("--long_sequence_inference")
    fi

    if [ "$USE_DEEPSPEED_ATTENTION" = true ]; then
        cmd_ref+=("--use_deepspeed_evoformer_attention")
    fi
}

# Check if required files exist for a method
check_method_requirements() {
    local method_name="$1"
    shift
    local required_files=("$@")

    echo "🔍 Checking requirements for $method_name..."

    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ] && [ ! -d "$file" ]; then
            echo "❌ Missing required file/directory: $file"
            return 1
        fi
    done

    echo "✅ All requirements satisfied for $method_name"
    return 0
}

# Run structure module prediction
run_structure_module() {
    local method_name="$1"
    local msa_path="$2"
    local pair_path="$3"
    local output_dir="$4"
    local python_path="$5"

    echo ""
    echo "🧬 Running $method_name..."
    echo "   MSA: $msa_path"
    echo "   Pair: $pair_path"
    echo "   Output: $output_dir"
    echo ""

    mkdir -p "$output_dir"

    # Change to structure module directory
    cd "${ROOT_DIR}/save_intermediates"

    # Build command array
    local cmd=(
        "$python_path"
        "run_structure_module.py"
        "$FASTA_DIR"
        "$TEMPLATE_MMCIF_DIR"
        "--msa_path" "$msa_path"
        "--pair_path" "$pair_path"
        "--output_dir" "$output_dir"
        "--model_device" "$MODEL_DEVICE"
        "--config_preset" "$CONFIG_PRESET"
    )

    # Add optional arguments using the new function
    add_optional_args cmd

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"

    echo "✅ $method_name completed successfully!"
}

# Run full OpenFold
run_full_openfold() {
    local output_dir="$1"
    local python_path="$2"

    echo ""
    echo "🧬 Running Full OpenFold..."
    echo "   Output: $output_dir"
    echo ""

    mkdir -p "$output_dir"

    # Change to working directory (based on original script)
    cd "${ROOT_DIR}/save_intermediates"

    # Build command array
    local cmd=(
        "$python_path"
        "run_pretrained_openfold.py"
        "$FASTA_DIR"
        "$TEMPLATE_MMCIF_DIR"
        "--output_dir" "$output_dir"
        "--config_preset" "$CONFIG_PRESET"
        "--model_device" "$MODEL_DEVICE"
        "--save_intermediates"
        "--data_random_seed" "3"
    )

    # Add optional arguments using the new function
    add_optional_args cmd

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"

    echo "✅ Full OpenFold completed successfully!"
}

run_full_openfold2() {
    local output_dir="$1"
    local python_path="$2"

    echo ""
    echo "🧬 Running Full OpenFold..."
    echo "   Output: $output_dir"
    echo ""

    mkdir -p "$output_dir"

    # Change to working directory (based on original script)
    cd "${ROOT_DIR}/save_intermediates"

    # Build command array
    local cmd=(
        "$python_path"
        "run_pretrained_openfold.py"
        "$FASTA_DIR"
        "$TEMPLATE_MMCIF_DIR"
        "--output_dir" "$output_dir"
        "--config_preset" "$CONFIG_PRESET"
        "--model_device" "$MODEL_DEVICE"
        "--save_intermediates"
        "--data_random_seed" "65"
    )

    # Add optional arguments using the new function
    add_optional_args cmd

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"

    echo "✅ Full OpenFold completed successfully!"
}
# =======================================================================================
# MAIN EXECUTION
# =======================================================================================

echo "========================================="
echo "MULTI-METHOD STRUCTURE PREDICTION"
echo "========================================="
echo "Protein ID: $PDB_ID"
echo "Root Directory: $ROOT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Model Device: $MODEL_DEVICE"
echo "Config Preset: $CONFIG_PRESET"
echo "========================================="

# Validate PDB_ID
if [ -z "$PDB_ID" ] || [ "$PDB_ID" = "CHANGE_ME" ]; then
    echo "❌ Error: Please set PDB_ID in the script configuration section"
    echo "   Edit the PDB_ID variable at the top of this script"
    exit 1
fi

# Set up environment
set_device
PYTHON_PATH=$(get_python_path)

# Find the latest Neural ODE predictions directory
echo "🔍 Searching for latest Neural ODE predictions..."
if ! find_latest_neural_ode_predictions; then
    echo "⚠️  Neural ODE predictions not found - will skip Method 1"
fi

echo "🔧 Configuration:"
echo "   Python: $PYTHON_PATH"
echo "   Device: $MODEL_DEVICE"
echo "   FASTA Directory: $FASTA_DIR"
if [ -n "$NEURAL_ODE_PREDICTIONS_DIR" ]; then
    echo "   Neural ODE Predictions: $NEURAL_ODE_PREDICTIONS_DIR"
fi
echo ""

# Check if FASTA directory exists
if [ ! -d "$FASTA_DIR" ]; then
    echo "❌ Error: FASTA directory not found: $FASTA_DIR"
    echo "   Make sure the protein data exists for $PDB_ID"
    exit 1
fi

# Track which methods succeed
declare -a successful_methods=()
declare -a failed_methods=()

# =======================================================================================
# METHOD 1: NEURAL ODE PREDICTIONS
# =======================================================================================

echo ""
echo "🎯 METHOD 1: Neural ODE Predictions"
echo "========================================="

# Check if Neural ODE predictions were found
if [ -z "$NEURAL_ODE_PREDICTIONS_DIR" ]; then
    echo "⏭️  Skipping Neural ODE method - no predictions found"
    failed_methods+=("Neural ODE (no predictions found)")
elif check_method_requirements "Neural ODE" \
    "$NEURAL_ODE_MSA_PATH" \
    "$NEURAL_ODE_PAIR_PATH" \
    "$FASTA_DIR" \
    "$TEMPLATE_MMCIF_DIR"; then

    if run_structure_module \
        "Neural ODE" \
        "$NEURAL_ODE_MSA_PATH" \
        "$NEURAL_ODE_PAIR_PATH" \
        "$NEURAL_ODE_OUTPUT_DIR" \
        "$PYTHON_PATH"; then
        successful_methods+=("Neural ODE")
    else
        failed_methods+=("Neural ODE")
    fi
else
    echo "⏭️  Skipping Neural ODE method due to missing requirements"
    failed_methods+=("Neural ODE (missing files)")
fi

# =======================================================================================
# METHOD 2: OPENFOLD DECONSTRUCTED (48th Evoformer Block)
# =======================================================================================

echo ""
echo "🎯 METHOD 2: OpenFold Deconstructed (48th Evoformer Block)"
echo "========================================="

if check_method_requirements "OpenFold Deconstructed" \
    "$OPENFOLD_DECON_MSA_PATH" \
    "$OPENFOLD_DECON_PAIR_PATH" \
    "$FASTA_DIR" \
    "$TEMPLATE_MMCIF_DIR"; then

    if run_structure_module \
        "OpenFold Deconstructed" \
        "$OPENFOLD_DECON_MSA_PATH" \
        "$OPENFOLD_DECON_PAIR_PATH" \
        "$OPENFOLD_DECON_OUTPUT_DIR" \
        "$PYTHON_PATH"; then
        successful_methods+=("OpenFold Deconstructed")
    else
        failed_methods+=("OpenFold Deconstructed")
    fi
else
    echo "⏭️  Skipping OpenFold Deconstructed method due to missing requirements"
    failed_methods+=("OpenFold Deconstructed (missing files)")
fi

# =======================================================================================
# METHOD 3: FULL OPENFOLD
# =======================================================================================

echo ""
echo "🎯 METHOD 3: Full OpenFold"
echo "========================================="

if check_method_requirements "Full OpenFold" \
    "$FASTA_DIR" \
    "$TEMPLATE_MMCIF_DIR"; then

    if run_full_openfold \
        "$OPENFOLD_FULL_OUTPUT_DIR" \
        "$PYTHON_PATH"; then
        successful_methods+=("Full OpenFold")
    else
        failed_methods+=("Full OpenFold")
    fi
else
    echo "⏭️  Skipping Full OpenFold method due to missing requirements"
    failed_methods+=("Full OpenFold (missing files)")
fi

# =======================================================================================
# METHOD 4: FULL OPENFOLD AGAIN
# =======================================================================================

echo ""
echo "🎯 METHOD 4: Full OpenFold AGAIN"
echo "========================================="

if check_method_requirements "Full OpenFold" \
    "$FASTA_DIR" \
    "$TEMPLATE_MMCIF_DIR"; then

    if run_full_openfold2 \
        "$NEURAL_ODE_OUTPUT_DIR" \
        "$PYTHON_PATH"; then
        successful_methods+=("Full OpenFold")
    else
        failed_methods+=("Full OpenFold")
    fi
else
    echo "⏭️  Skipping Full OpenFold method due to missing requirements"
    failed_methods+=("Full OpenFold (missing files)")
fi

# =======================================================================================
# FINAL SUMMARY
# =======================================================================================

echo ""
echo "========================================="
echo "FINAL SUMMARY FOR $PDB_ID"
echo "========================================="

echo "✅ Successful methods (${#successful_methods[@]}):"
if [ ${#successful_methods[@]} -eq 0 ]; then
    echo "   None"
else
    for method in "${successful_methods[@]}"; do
        echo "   - $method"
    done
fi

echo ""
echo "❌ Failed methods (${#failed_methods[@]}):"
if [ ${#failed_methods[@]} -eq 0 ]; then
    echo "   None"
else
    for method in "${failed_methods[@]}"; do
        echo "   - $method"
    done
fi

echo ""
echo "📁 Output directories:"
echo "   Neural ODE: $NEURAL_ODE_OUTPUT_DIR"
echo "   OpenFold Deconstructed: $OPENFOLD_DECON_OUTPUT_DIR"
echo "   Full OpenFold: $OPENFOLD_FULL_OUTPUT_DIR"

echo ""
echo "🎯 Structure prediction pipeline completed!"

# Exit with error code if no methods succeeded
if [ ${#successful_methods[@]} -eq 0 ]; then
    echo "❌ No methods completed successfully!"
    exit 1
else
    echo "✅ At least one method completed successfully!"
    exit 0
fi