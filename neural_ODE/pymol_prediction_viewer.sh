#!/bin/bash

# =======================================================================================
# MULTI-METHOD PYMOL STRUCTURE VIEWER WITH NEURAL ODE VARIANTS
# Launches PyMOL instances for all prediction methods including multiple Neural ODE models
# =======================================================================================

# Base data directory
DATA_DIR="/Volumes/Extreme SSD/data"

# =======================================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO RUN DIRECTLY
# =======================================================================================

PDB_ID="1fv5_A"                    # Change this to your protein ID
STRUCTURE_TYPE="relaxed"           # Change to "unrelaxed" if you want unrelaxed structures
NEURAL_ODE_PREDICTIONS=()          # Empty = auto-discover all available Neural ODE predictions

# =======================================================================================
# COMMAND LINE SUPPORT
# =======================================================================================

show_usage() {
    echo "Usage: $0 <PDB_ID> [relaxed|unrelaxed]"
    echo ""
    echo "Examples:"
    echo "  $0 1fv5_A"
    echo "  $0 1fv5_A relaxed"
    echo "  $0 1fv5_A unrelaxed"
    echo ""
}

# Parse command line arguments
if [ $# -ge 1 ]; then
    PDB_ID="$1"
    STRUCTURE_TYPE="${2:-relaxed}"
fi

# Validate structure type
if [ "$STRUCTURE_TYPE" != "relaxed" ] && [ "$STRUCTURE_TYPE" != "unrelaxed" ]; then
    echo "Error: Structure type must be 'relaxed' or 'unrelaxed'"
    show_usage
    exit 1
fi

# =======================================================================================
# AUTO-DISCOVERY
# =======================================================================================

discover_neural_ode_predictions() {
    local pdb_id="$1"
    local neural_ode_base_dir="${DATA_DIR}/structure_predictions/${pdb_id}/neuralODE"

    if [ -d "$neural_ode_base_dir" ]; then
        for pred_dir in "$neural_ode_base_dir"/*; do
            if [ -d "$pred_dir" ]; then
                pred_name=$(basename "$pred_dir")
                pred_file="${pred_dir}/${pdb_id}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
                if [ -f "$pred_file" ]; then
                    echo "$pred_name"
                fi
            fi
        done
    fi
}

# =======================================================================================
# MAIN
# =======================================================================================

echo "========================================="
echo "MULTI-METHOD PYMOL STRUCTURE VIEWER"
echo "========================================="
echo "Protein ID: $PDB_ID"
echo "Structure Type: $STRUCTURE_TYPE"
echo "Data Directory: $DATA_DIR"
echo "========================================="

# Auto-discover Neural ODE predictions
echo ""
echo "Discovering Neural ODE predictions..."
while IFS= read -r prediction; do
    if [ -n "$prediction" ]; then
        NEURAL_ODE_PREDICTIONS+=("$prediction")
    fi
done < <(discover_neural_ode_predictions "$PDB_ID")

if [ ${#NEURAL_ODE_PREDICTIONS[@]} -eq 0 ]; then
    echo "No Neural ODE predictions found for $PDB_ID"
else
    echo "Found ${#NEURAL_ODE_PREDICTIONS[@]} Neural ODE predictions"
fi

# Build list of all PDB files to load
PDB_FILES=()
OBJECT_NAMES=()

# Add Neural ODE predictions
for pred in "${NEURAL_ODE_PREDICTIONS[@]}"; do
    pdb_file="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE/${pred}/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
    if [ -f "$pdb_file" ]; then
        PDB_FILES+=("$pdb_file")
        # Create short name for PyMOL object
        short_name=$(echo "$pred" | sed 's/predictions_//' | cut -c1-20)
        OBJECT_NAMES+=("NeuralODE_${short_name}")
    fi
done

# Add OpenFold Deconstructed
decon_file="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_deconstructed/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
if [ -f "$decon_file" ]; then
    PDB_FILES+=("$decon_file")
    OBJECT_NAMES+=("OpenFold_Decon")
fi

# Add Full OpenFold
full_file="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_0recycles/predictions/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
if [ -f "$full_file" ]; then
    PDB_FILES+=("$full_file")
    OBJECT_NAMES+=("OpenFold_Full")
fi

# Add Half Evoformer
half_file="${DATA_DIR}/structure_predictions/${PDB_ID}/half_evoformer/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
if [ -f "$half_file" ]; then
    PDB_FILES+=("$half_file")
    OBJECT_NAMES+=("Half_Evoformer")
fi

echo ""
echo "Found ${#PDB_FILES[@]} structures to load:"
for i in "${!OBJECT_NAMES[@]}"; do
    echo "  - ${OBJECT_NAMES[$i]}"
done

if [ ${#PDB_FILES[@]} -eq 0 ]; then
    echo "Error: No structure files found"
    exit 1
fi

# Create PyMOL script
PYMOL_SCRIPT=$(mktemp /tmp/pymol_view_XXXXXX.pml)

echo "# PyMOL script for ${PDB_ID}" > "$PYMOL_SCRIPT"
echo "bg_color white" >> "$PYMOL_SCRIPT"

# Load all structures
for i in "${!PDB_FILES[@]}"; do
    echo "load ${PDB_FILES[$i]}, ${OBJECT_NAMES[$i]}" >> "$PYMOL_SCRIPT"
done

# Set up visualization
echo "" >> "$PYMOL_SCRIPT"
echo "# Visualization settings" >> "$PYMOL_SCRIPT"
echo "hide all" >> "$PYMOL_SCRIPT"
echo "show cartoon, all" >> "$PYMOL_SCRIPT"
echo "spectrum b, blue_white_red, all" >> "$PYMOL_SCRIPT"

# Align all to first structure (reference)
if [ ${#OBJECT_NAMES[@]} -gt 1 ]; then
    echo "" >> "$PYMOL_SCRIPT"
    echo "# Align all structures to ${OBJECT_NAMES[0]}" >> "$PYMOL_SCRIPT"
    for i in "${!OBJECT_NAMES[@]}"; do
        if [ $i -gt 0 ]; then
            echo "align ${OBJECT_NAMES[$i]}, ${OBJECT_NAMES[0]}" >> "$PYMOL_SCRIPT"
        fi
    done
fi

echo "" >> "$PYMOL_SCRIPT"
echo "zoom all" >> "$PYMOL_SCRIPT"
echo "center all" >> "$PYMOL_SCRIPT"

echo ""
echo "========================================="
echo "Launching PyMOL..."
echo "========================================="
echo ""
echo "Structures will be aligned to: ${OBJECT_NAMES[0]}"
echo "Color: blue=low pLDDT, red=high pLDDT"
echo ""

# Launch PyMOL with the script
pymol "$PYMOL_SCRIPT" &

echo "PyMOL launched. Script saved to: $PYMOL_SCRIPT"
echo ""
echo "Tips:"
echo "  - Toggle structures on/off in the right panel"
echo "  - Use 'disable all' then 'enable <name>' to view one at a time"
echo "  - Type 'quit' in PyMOL to exit"