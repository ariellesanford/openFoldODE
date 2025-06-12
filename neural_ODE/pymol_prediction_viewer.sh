#!/bin/bash

# =======================================================================================
# MULTI-METHOD PYMOL STRUCTURE VIEWER
# Launches PyMOL instances for all three prediction methods simultaneously
# =======================================================================================

# Base data directory
DATA_DIR="/media/visitor/Extreme SSD/data"

# =======================================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO RUN IN PYCHARM
# =======================================================================================

# CONFIGURATION SECTION - Edit these variables directly:
PDB_ID="7vsw_A"                    # Change this to your protein ID
STRUCTURE_TYPE="relaxed"           # Change to "unrelaxed" if you want unrelaxed structures

# =======================================================================================
# COMMAND LINE SUPPORT (for terminal usage)
# =======================================================================================

show_usage() {
    echo "Usage: $0 <PDB_ID> [relaxed|unrelaxed]"
    echo ""
    echo "Examples:"
    echo "  $0 1fv5_A                # View relaxed structures (default)"
    echo "  $0 1fv5_A relaxed        # View relaxed structures"
    echo "  $0 1fv5_A unrelaxed      # View unrelaxed structures"
    echo ""
    echo "OR edit the script directly for PyCharm usage:"
    echo "  Set PDB_ID=\"your_protein\" and STRUCTURE_TYPE=\"relaxed|unrelaxed\""
    echo ""
    echo "This will launch 3 PyMOL windows showing:"
    echo "  1. Neural ODE prediction"
    echo "  2. OpenFold Deconstructed (48th block)"
    echo "  3. Full OpenFold prediction"
    echo ""
}

# Parse command line arguments (override config if provided)
if [ $# -eq 1 ] || [ $# -eq 2 ]; then
    PDB_ID="$1"
    STRUCTURE_TYPE="${2:-relaxed}"  # Default to relaxed if not provided
elif [ $# -gt 2 ]; then
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
# PATH CONSTRUCTION
# =======================================================================================

# Construct file paths for each method
NEURAL_ODE_PATH="${DATA_DIR}/structure_predictions/${PDB_ID}/neuralODE/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
OPENFOLD_DECON_PATH="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_deconstructed/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"
OPENFOLD_FULL_PATH="${DATA_DIR}/structure_predictions/${PDB_ID}/openfold_0recycles/predictions/${PDB_ID}_model_1_ptm_${STRUCTURE_TYPE}.pdb"

# Method names for display
METHOD_NAMES=("Neural ODE" "OpenFold Deconstructed" "Full OpenFold")
PDB_PATHS=("$NEURAL_ODE_PATH" "$OPENFOLD_DECON_PATH" "$OPENFOLD_FULL_PATH")

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
        "Neural ODE")
            object_name="NeuralODE_${PDB_ID}"
            ;;
        "OpenFold Deconstructed")
            object_name="OpenFoldDecon_${PDB_ID}"
            ;;
        "Full OpenFold")
            object_name="FullOpenFold_${PDB_ID}"
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
#pymol.cmd.color('spectrum', object_name)
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
echo ""
echo "💡 To change settings for PyCharm:"
echo "   Edit PDB_ID and STRUCTURE_TYPE variables at the top of this script"
echo "========================================="

echo "📁 Looking for structures in: $PROTEIN_DIR"
echo ""

# Track successful launches
successful_launches=0
failed_launches=0

# Launch PyMOL for each method
for i in "${!METHOD_NAMES[@]}"; do
    method_name="${METHOD_NAMES[$i]}"
    pdb_path="${PDB_PATHS[$i]}"

    echo "[$((i+1))/3] $method_name"

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
elif [ $successful_launches -lt 3 ]; then
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
echo "👋 Thanks for using the multi-method PyMOL viewer!"