#!/bin/bash

# =======================================================================================
# MULTI-METHOD PYMOL STRUCTURE VIEWER WITH NEURAL ODE VARIANTS
# Launches PyMOL instances for all prediction methods including multiple Neural ODE models
# =======================================================================================

# Base data directory
DATA_DIR="/media/visitor/Extreme SSD/data"

# =======================================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO RUN IN PYCHARM
# =======================================================================================

# CONFIGURATION SECTION - Edit these variables directly:
PDB_ID="5yaa_A"                    # Change this to your protein ID
STRUCTURE_TYPE="relaxed"           # Change to "unrelaxed" if you want unrelaxed structures

# Neural ODE predictions to include (leave empty to auto-discover all available)
# Example: NEURAL_ODE_PREDICTIONS=("predictions_20250613_180436_baseline_no_prelim" "predictions_20250614_120000_fast_ode")
NEURAL_ODE_PREDICTIONS=()  # Empty = auto-discover all available Neural ODE predictions

# =======================================================================================
# COMMAND LINE SUPPORT (for terminal usage)
# =======================================================================================

show_usage() {
    echo "Usage: $0 <PDB_ID> [relaxed|unrelaxed] [neural_ode_prediction_1] [neural_ode_prediction_2] ..."
    echo ""
    echo "Examples:"
    echo "  $0 1fv5_A                                    # View relaxed structures (auto-discover Neural ODE)"
    echo "  $0 1fv5_A relaxed                            # View relaxed structures (auto-discover Neural ODE)"
    echo "  $0 1fv5_A unrelaxed                          # View unrelaxed structures (auto-discover Neural ODE)"
    echo "  $0 1fv5_A relaxed predictions_20250613_180436_baseline_no_prelim"
    echo "  $0 1fv5_A relaxed pred1 pred2 pred3          # View specific Neural ODE predictions"
    echo ""
    echo "OR edit the script directly for PyCharm usage:"
    echo "  Set PDB_ID=\"your_protein\" and STRUCTURE_TYPE=\"relaxed|unrelaxed\""
    echo "  Set NEURAL_ODE_PREDICTIONS array for specific predictions or leave empty for all"
    echo ""
    echo "This will launch PyMOL windows showing:"
    echo "  1. All Neural ODE predictions found/specified"
    echo "  2. OpenFold Deconstructed (48th block)"
    echo "  3. Full OpenFold prediction"
    echo "  4. Half Evoformer (block 24 only)"
    echo ""
}

# Parse command line arguments (override config if provided)
if [ $# -ge 1 ]; then
    PDB_ID="$1"
    STRUCTURE_TYPE="${2:-relaxed}"  # Default to relaxed if not provided

    # If more than 2 arguments, treat the rest as Neural ODE predictions
    if [ $# -gt 2 ]; then
        shift 2  # Remove PDB_ID and STRUCTURE_TYPE
        NEURAL_ODE_PREDICTIONS=("$@")  # Use remaining arguments
    fi
elif [ $# -gt 0 ]; then
    show_usage
    exit 1
fi
# If no arguments provided, use the configured values above

# Validate structure type
if [ "$STRUCTURE_TYPE" != "relaxed" ] && [ "$STRUCTURE_TYPE" != "unrelaxed" ]; then
    echo "❌ Error: Structure type must be 'relaxed' or 'unrelaxed'"
    echo "   Current value: '$STRUCTURE_TYPE'"
    echo "   Edit the STRUCTURE_TYPE variable in the script or use command line"
    show_usage
    exit 1
fi

# =======================================================================================
# AUTO-DISCOVERY FUNCTIONS
# =======================================================================================

discover_neural_ode_predictions() {
    local pdb_id="$1"
    local neural_ode_base_dir="${DATA_DIR}/structure_predictions/${pdb_id}/neuralODE"

    discovered_predictions=()

    if [ -d "$neural_ode_base_dir" ]; then
        for pred_dir in "$neural_ode_base_dir"/*; do
            if [ -d "$pred_dir" ]; then
                pred_name=$(basename "$pred_dir")
                # Check if this directory contains actual prediction files
                pred_file="${pred_dir}/${pdb_id}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
                if [ -f "$pred_file" ]; then
                    discovered_predictions+=("$pred_name")
                fi
            fi
        done
    fi

    # Return the array properly (one element per line)
    printf '%s\n' "${discovered_predictions[@]}"
}

# =======================================================================================
# PATH CONSTRUCTION
# =======================================================================================

# Auto-discover Neural ODE predictions if none specified
if [ ${#NEURAL_ODE_PREDICTIONS[@]} -eq 0 ]; then
    echo "🔍 Auto-discovering Neural ODE predictions for $PDB_ID..."

    # Read the function output into an array, one element per line
    while IFS= read -r prediction; do
        if [ -n "$prediction" ]; then
            NEURAL_ODE_PREDICTIONS+=("$prediction")
        fi
    done < <(discover_neural_ode_predictions "$PDB_ID")

    if [ ${#NEURAL_ODE_PREDICTIONS[@]} -eq 0 ]; then
        echo "⚠️  No Neural ODE predictions found for $PDB_ID"
        echo "   Looking in: ${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE/"
    else
        echo "✅ Found ${#NEURAL_ODE_PREDICTIONS[@]} Neural ODE predictions:"
        for pred in "${NEURAL_ODE_PREDICTIONS[@]}"; do
            echo "   - $pred"
        done
    fi
fi

# Construct file paths for each method
METHOD_NAMES=()
PDB_PATHS=()

# Add Neural ODE predictions
for neural_ode_pred in "${NEURAL_ODE_PREDICTIONS[@]}"; do
    neural_ode_path="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE/${neural_ode_pred}/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
    METHOD_NAMES+=("Neural ODE (${neural_ode_pred})")
    PDB_PATHS+=("$neural_ode_path")
done

# Add other methods
OPENFOLD_DECON_PATH="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_deconstructed/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
OPENFOLD_FULL_PATH="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_0recycles/predictions/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
HALF_EVOFORMER_PATH="${DATA_DIR}/structure_predictions/${PDB_ID}/half_evoformer/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"

METHOD_NAMES+=("OpenFold Deconstructed" "Full OpenFold" "Half Evoformer")
PDB_PATHS+=("$OPENFOLD_DECON_PATH" "$OPENFOLD_FULL_PATH" "$HALF_EVOFORMER_PATH")

# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================

# Launch PyMOL for a single structure
launch_pymol() {
    local pdb_path="$1"
    local method_name="$2"

    if [ ! -f "$pdb_path" ]; then
        echo "⚠️  File not found: $pdb_path"
        echo "   Skipping $method_name"
        return 1
    fi

    echo "🚀 Launching PyMOL for $method_name..."
    echo "   File: $pdb_path"

    # Create a clean object name (PyMOL-safe)
    local object_name=""
    case "$method_name" in
        "Neural ODE"*)
            # Extract prediction name from method name
            local pred_name=$(echo "$method_name" | sed 's/Neural ODE (\(.*\))/\1/')
            # Shorten prediction name for PyMOL object
            local short_name=$(echo "$pred_name" | sed 's/predictions_//' | sed 's/_[0-9]*_[0-9]*_/_/')
            object_name="${short_name}_${PDB_ID}"
            ;;
        "OpenFold Deconstructed")
            object_name="OpenFoldDecon_${PDB_ID}"
            ;;
        "Full OpenFold")
            object_name="FullOpenFold_${PDB_ID}"
            ;;
        "Half Evoformer")
            object_name="HalfEvoformer_${PDB_ID}"
            ;;
        *)
            object_name="Structure_${PDB_ID}"
            ;;
    esac

    # Launch PyMOL in background
    (
        source ~/anaconda3/bin/activate pymol_env
        python - <<EOF &
import pymol
import sys
import os

# Launch PyMOL
pymol.finish_launching()

# Load the structure with a clean object name
object_name = "$object_name"
pymol.cmd.load('$pdb_path', object_name)

# Set up nice visualization
pymol.cmd.show('cartoon', object_name)
pymol.cmd.spectrum('b', 'rainbow', object_name)  # Color by B-factor
pymol.cmd.zoom(object_name)

# Print info to console
print(f"Loaded: $method_name - $PDB_ID ($STRUCTURE_TYPE)")
print(f"Object: {object_name}")

# Keep PyMOL running
EOF
    ) &

    # Store the process ID
    local pymol_pid=$!
    echo "   PyMOL PID: $pymol_pid"

    # Small delay to avoid overwhelming the system
    sleep 2

    return 0
}

# =======================================================================================
# MAIN EXECUTION
# =======================================================================================

echo "========================================="
echo "MULTI-METHOD PYMOL STRUCTURE VIEWER"
echo "========================================="
echo "🔧 Configuration:"
echo "   Protein ID: $PDB_ID"
echo "   Structure Type: $STRUCTURE_TYPE"
echo "   Data Directory: $DATA_DIR"
echo "   Neural ODE Predictions: ${#NEURAL_ODE_PREDICTIONS[@]}"
for pred in "${NEURAL_ODE_PREDICTIONS[@]}"; do
    echo "     - $pred"
done
echo ""
echo "💡 To change settings for PyCharm:"
echo "   Edit PDB_ID, STRUCTURE_TYPE, and NEURAL_ODE_PREDICTIONS variables at the top of this script"
echo "========================================="

echo "📁 Looking for structures for: $PDB_ID"
echo ""

# Track successful launches
successful_launches=0
failed_launches=0

# Launch PyMOL for each method
for i in "${!METHOD_NAMES[@]}"; do
    method_name="${METHOD_NAMES[$i]}"
    pdb_path="${PDB_PATHS[$i]}"

    echo "[$((i+1))/${#METHOD_NAMES[@]}] $method_name"

    if launch_pymol "$pdb_path" "$method_name"; then
        ((successful_launches++))
        echo "   ✅ Launched successfully"
    else
        ((failed_launches++))
        echo "   ❌ Launch failed"
    fi
    echo ""
done

# Summary
echo "========================================="
echo "LAUNCH SUMMARY"
echo "========================================="
echo "✅ Successful launches: $successful_launches"
echo "❌ Failed launches: $failed_launches"
echo ""

if [ $successful_launches -eq 0 ]; then
    echo "❌ No PyMOL instances were launched successfully"
    echo "   Check that the structure prediction files exist"
    exit 1
elif [ $successful_launches -lt ${#METHOD_NAMES[@]} ]; then
    echo "⚠️  Some PyMOL instances failed to launch"
    echo "   This usually means some prediction methods haven't been run yet"
else
    echo "🎉 All PyMOL instances launched successfully!"
fi

echo ""
echo "💡 Tips:"
echo "   - Each PyMOL window shows a different prediction method"
echo "   - You can arrange windows side-by-side for comparison"
echo "   - Use 'quit' in each PyMOL window to close it"
echo "   - Press Ctrl+C in this terminal to see the summary again"
echo ""
echo "🔬 Method comparison:"
if [ ${#NEURAL_ODE_PREDICTIONS[@]} -gt 0 ]; then
    echo "   - Neural ODE variants: AI-learned Evoformer dynamics (${#NEURAL_ODE_PREDICTIONS[@]} models)"
else
    echo "   - Neural ODE: Not available"
fi
echo "   - OpenFold Deconstructed: Traditional 48 blocks"
echo "   - Full OpenFold: Complete pipeline"
echo "   - Half Evoformer: Structure module only (baseline)"
echo ""

# Wait for user to exit
echo "Press Enter to see this summary again, or Ctrl+C to exit..."
read -r

# Show final status
echo ""
echo "🔍 Currently launched PyMOL instances for $PDB_ID:"
for i in "${!METHOD_NAMES[@]}"; do
    method_name="${METHOD_NAMES[$i]}"
    pdb_path="${PDB_PATHS[$i]}"

    if [ -f "$pdb_path" ]; then
        echo "   ✅ $method_name: Available"
    else
        echo "   ❌ $method_name: File not found"
    fi
done

echo ""
echo "📊 Neural ODE predictions status:"
if [ ${#NEURAL_ODE_PREDICTIONS[@]} -eq 0 ]; then
    echo "   No Neural ODE predictions found or specified"
    echo "   Run Neural ODE testing first to generate predictions"
else
    for pred in "${NEURAL_ODE_PREDICTIONS[@]}"; do
        pred_file="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE/${pred}/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
        if [ -f "$pred_file" ]; then
            echo "   ✅ $pred"
        else
            echo "   ❌ $pred (file missing)"
        fi
    done
fi

echo ""
echo "👋 Thanks for using the multi-method PyMOL viewer!"