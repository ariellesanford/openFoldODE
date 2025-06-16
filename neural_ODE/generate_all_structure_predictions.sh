#!/bin/bash
set -e

# =======================================================================================
# MULTI-METHOD STRUCTURE PREDICTION SCRIPT
# Generates structure predictions using 4 different methods for a given PDB_ID
# MODIFIED: Explicit Neural ODE predictions directory handling
# =======================================================================================

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="/home/visitor/PycharmProjects/openFold"
DATA_DIR="/media/visitor/Extreme SSD/data"

# =======================================================================================
# CONFIGURATION - Modify these variables
# =======================================================================================

# Protein ID to process (REQUIRED - set this!)
PDB_ID="1fv5_A"  # Change this to your desired protein

# Neural ODE predictions base directory (auto-discover all available predictions)
NEURAL_ODE_PREDICTIONS_BASE_DIR="${DATA_DIR}/post_evoformer_predictions"

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
# DERIVED PATHS - These are automatically set based on PDB_ID and Neural ODE filename
# =======================================================================================

# Common paths
FASTA_DIR="${DATA_DIR}/fasta_data/${PDB_ID}"
TEMPLATE_MMCIF_DIR="${DATA_DIR}/template_data/pdb70_mmcif/mmcif_files"
PRECOMPUTED_ALIGNMENTS="${DATA_DIR}/alignments"

# Method 1: Neural ODE predictions (explicitly configured)
NEURAL_ODE_PREDICTIONS_DIR="${NEURAL_ODE_PREDICTIONS_BASE_DIR}/${NEURAL_ODE_PREDICTIONS_FILENAME}/${PDB_ID}"
NEURAL_ODE_MSA_PATH="${NEURAL_ODE_PREDICTIONS_DIR}/msa_representation.pt"
NEURAL_ODE_PAIR_PATH="${NEURAL_ODE_PREDICTIONS_DIR}/pair_representation.pt"
NEURAL_ODE_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE_${NEURAL_ODE_PREDICTIONS_FILENAME}"

# Method 2: OpenFold Deconstructed (48th evoformer block)
OPENFOLD_DECON_MSA_PATH="${DATA_DIR}/complete_blocks/${PDB_ID}_evoformer_blocks/recycle_0/m_block_48.pt"
OPENFOLD_DECON_PAIR_PATH="${DATA_DIR}/complete_blocks/${PDB_ID}_evoformer_blocks/recycle_0/z_block_48.pt"
OPENFOLD_DECON_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_deconstructed"

# Method 3: Full OpenFold
OPENFOLD_FULL_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_0recycles"

# Method 4: Half Evoformer (block 24 only)
HALF_EVOFORMER_MSA_PATH="${DATA_DIR}/complete_blocks/${PDB_ID}_evoformer_blocks/recycle_0/m_block_24.pt"
HALF_EVOFORMER_PAIR_PATH="${DATA_DIR}/complete_blocks/${PDB_ID}_evoformer_blocks/recycle_0/z_block_24.pt"
HALF_EVOFORMER_OUTPUT_DIR="${DATA_DIR}/structure_predictions/${PDB_ID}/half_evoformer"

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

# Discover available Neural ODE predictions for a protein
discover_neural_ode_predictions() {
    local pdb_id="$1"
    local predictions_base_dir="$2"

    discovered_predictions=()

    if [ -d "$predictions_base_dir" ]; then
        for pred_dir in "$predictions_base_dir"/predictions_*; do
            if [ -d "$pred_dir" ]; then
                pred_name=$(basename "$pred_dir")
                protein_pred_dir="${pred_dir}/${pdb_id}"

                # Check if this prediction contains data for our protein
                if [ -d "$protein_pred_dir" ]; then
                    msa_file="${protein_pred_dir}/msa_representation.pt"
                    pair_file="${protein_pred_dir}/pair_representation.pt"

                    if [ -f "$msa_file" ] && [ -f "$pair_file" ]; then
                        discovered_predictions+=("$pred_name")
                    fi
                fi
            fi
        done
    fi

    printf '%s\n' "${discovered_predictions[@]}"
}

# Check if Neural ODE structure prediction already exists
check_neural_ode_output_exists() {
    local pdb_id="$1"
    local pred_name="$2"
    local output_dir="${DATA_DIR}/structure_predictions/${pdb_id}/neuralODE/${pred_name}"

    # Check for both relaxed and unrelaxed structures
    local relaxed_file="${output_dir}/${pdb_id}_model_1_ptm_relaxed.pdb"
    local unrelaxed_file="${output_dir}/${pdb_id}_model_1_ptm_unrelaxed.pdb"

    if [ -f "$relaxed_file" ] || [ -f "$unrelaxed_file" ]; then
        return 0  # Exists
    else
        return 1  # Doesn't exist
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

# Run structure module prediction for Neural ODE
run_neural_ode_structure_module() {
    local pdb_id="$1"
    local pred_name="$2"
    local python_path="$3"

    local predictions_dir="${NEURAL_ODE_PREDICTIONS_BASE_DIR}/${pred_name}/${pdb_id}"
    local msa_path="${predictions_dir}/msa_representation.pt"
    local pair_path="${predictions_dir}/pair_representation.pt"
    local output_dir="${DATA_DIR}/structure_predictions/${pdb_id}/neuralODE/${pred_name}"

    echo ""
    echo "🧬 Running Neural ODE Structure Module (${pred_name})..."
    echo "   Predictions dir: $predictions_dir"
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

    # Add optional arguments
    add_optional_args cmd

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"

    echo "✅ Neural ODE Structure Module (${pred_name}) completed successfully!"
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
echo "Neural ODE Predictions: $NEURAL_ODE_PREDICTIONS_FILENAME"
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

# Validate Neural ODE base directory exists
if [ ! -d "$NEURAL_ODE_PREDICTIONS_BASE_DIR" ]; then
    echo "⚠️  Warning: Neural ODE predictions base directory not found: $NEURAL_ODE_PREDICTIONS_BASE_DIR"
    echo "   Neural ODE methods will be skipped"
fi

# Set up environment
set_device
PYTHON_PATH=$(get_python_path)

echo "🔧 Configuration:"
echo "   Python: $PYTHON_PATH"
echo "   Device: $MODEL_DEVICE"
echo "   FASTA Directory: $FASTA_DIR"
echo "   Neural ODE Predictions Directory: $NEURAL_ODE_PREDICTIONS_DIR"
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
# METHOD 1: NEURAL ODE PREDICTIONS (ALL AVAILABLE)
# =======================================================================================

echo ""
echo "🎯 METHOD 1: Neural ODE Predictions (All Available)"
echo "========================================="

if [ ${#NEURAL_ODE_PREDICTIONS[@]} -eq 0 ]; then
    echo "⏭️  No Neural ODE predictions available - skipping"
    failed_methods+=("Neural ODE (no predictions available)")
else
    neural_ode_processed=0
    neural_ode_skipped=0
    neural_ode_failed=0

    for pred_name in "${NEURAL_ODE_PREDICTIONS[@]}"; do
        echo ""
        echo "🔄 Processing Neural ODE prediction: $pred_name"

        # Check if structure prediction already exists
        if check_neural_ode_output_exists "$PDB_ID" "$pred_name"; then
            echo "   ✅ Structure prediction already exists - skipping"
            ((neural_ode_skipped++))
            continue
        fi

        # Verify input files exist
        predictions_dir="${NEURAL_ODE_PREDICTIONS_BASE_DIR}/${pred_name}/${PDB_ID}"
        msa_path="${predictions_dir}/msa_representation.pt"
        pair_path="${predictions_dir}/pair_representation.pt"

        if ! check_method_requirements "Neural ODE ($pred_name)" \
            "$msa_path" \
            "$pair_path" \
            "$FASTA_DIR" \
            "$TEMPLATE_MMCIF_DIR"; then
            echo "   ❌ Missing requirements - skipping"
            ((neural_ode_failed++))
            continue
        fi

        # Run structure prediction
        if run_neural_ode_structure_module "$PDB_ID" "$pred_name" "$PYTHON_PATH"; then
            echo "   ✅ Successfully processed $pred_name"
            ((neural_ode_processed++))
        else
            echo "   ❌ Failed to process $pred_name"
            ((neural_ode_failed++))
        fi
    done

    echo ""
    echo "📊 Neural ODE Summary:"
    echo "   Processed: $neural_ode_processed"
    echo "   Skipped (already exists): $neural_ode_skipped"
    echo "   Failed: $neural_ode_failed"

    if [ $neural_ode_processed -gt 0 ]; then
        successful_methods+=("Neural ODE ($neural_ode_processed models)")
    fi
    if [ $neural_ode_failed -gt 0 ]; then
        failed_methods+=("Neural ODE ($neural_ode_failed failed)")
    fi
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
# METHOD 4: HALF EVOFORMER (Block 24 Only)
# =======================================================================================

echo ""
echo "🎯 METHOD 4: Half Evoformer (Block 24 Only)"
echo "========================================="

if check_method_requirements "Half Evoformer" \
    "$HALF_EVOFORMER_MSA_PATH" \
    "$HALF_EVOFORMER_PAIR_PATH" \
    "$FASTA_DIR" \
    "$TEMPLATE_MMCIF_DIR"; then

    if run_structure_module \
        "Half Evoformer" \
        "$HALF_EVOFORMER_MSA_PATH" \
        "$HALF_EVOFORMER_PAIR_PATH" \
        "$HALF_EVOFORMER_OUTPUT_DIR" \
        "$PYTHON_PATH"; then
        successful_methods+=("Half Evoformer")
    else
        failed_methods+=("Half Evoformer")
    fi
else
    echo "⏭️  Skipping Half Evoformer method due to missing requirements"
    failed_methods+=("Half Evoformer (missing files)")
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
if [ ${#NEURAL_ODE_PREDICTIONS[@]} -gt 0 ]; then
    echo "   Neural ODE predictions:"
    for pred_name in "${NEURAL_ODE_PREDICTIONS[@]}"; do
        output_dir="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE/${pred_name}"
        echo "     - ${pred_name}: $output_dir"
    done
else
    echo "   Neural ODE: No predictions available"
fi
echo "   OpenFold Deconstructed: $OPENFOLD_DECON_OUTPUT_DIR"
echo "   Full OpenFold: $OPENFOLD_FULL_OUTPUT_DIR"
echo "   Half Evoformer: $HALF_EVOFORMER_OUTPUT_DIR"

echo ""
echo "🔧 Neural ODE Configuration Used:"
echo "   Base directory: $NEURAL_ODE_PREDICTIONS_BASE_DIR"
echo "   Available predictions: ${#NEURAL_ODE_PREDICTIONS[@]}"
for pred_name in "${NEURAL_ODE_PREDICTIONS[@]}"; do
    echo "     - $pred_name"
done

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