#!/bin/bash
set -e

# =======================================================================================
# COMPLETE EVOFORMER PIPELINE AUTOMATION
# Runs generate_evoformer_inputs.py first, then generate_48th_blocks.py
# =======================================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="/home/visitor/PycharmProjects/openFold"

# Configuration
CONFIG_PRESET="model_1_ptm"
DEVICE="cuda:0"

# Python path
PYTHON_PATH="${CONDA_PREFIX}/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python"
fi

echo "🚀 COMPLETE EVOFORMER PIPELINE"
echo "=============================="
echo "📁 Script directory: $SCRIPT_DIR"
echo "📁 Root directory: $ROOT_DIR"
echo "🔧 Config preset: $CONFIG_PRESET"
echo "💻 Device: $DEVICE"
echo "🐍 Python: $PYTHON_PATH"
echo ""

# =======================================================================================
# STAGE 1: Generate Initial Evoformer Inputs (Block 0)
# =======================================================================================

echo "🎯 STAGE 1: Generating Evoformer Inputs (Block 0)"
echo "================================================="
echo "📜 Running: generate_evoformer_inputs.py"
echo ""

cd "$SCRIPT_DIR"

$PYTHON_PATH generate_evoformer_inputs.py \
    --config-preset "$CONFIG_PRESET" \
    --device "$DEVICE"

STAGE1_EXIT_CODE=$?

if [ $STAGE1_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ STAGE 1 FAILED!"
    echo "   generate_evoformer_inputs.py exited with code $STAGE1_EXIT_CODE"
    echo "   Cannot proceed to Stage 2"
    exit $STAGE1_EXIT_CODE
fi

echo ""
echo "✅ STAGE 1 COMPLETED SUCCESSFULLY!"
echo "   All initial Evoformer blocks (block 0) have been generated"
echo ""

# =======================================================================================
# STAGE 2: Generate 48th Blocks
# =======================================================================================

echo "🎯 STAGE 2: Generating 48th Blocks"
echo "=================================="
echo "📜 Running: generate_48th_blocks.py"
echo ""

cd "$SCRIPT_DIR"

$PYTHON_PATH generate_48th_blocks.py

STAGE2_EXIT_CODE=$?

if [ $STAGE2_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ STAGE 2 FAILED!"
    echo "   generate_48th_blocks.py exited with code $STAGE2_EXIT_CODE"
    echo "   Stage 1 completed successfully, but Stage 2 failed"
    exit $STAGE2_EXIT_CODE
fi

echo ""
echo "✅ STAGE 2 COMPLETED SUCCESSFULLY!"
echo "   All 48th blocks have been generated"
echo ""

# =======================================================================================
# FINAL SUMMARY
# =======================================================================================

echo "🎉 COMPLETE PIPELINE FINISHED!"
echo "=============================="
echo "✅ Stage 1: Initial Evoformer inputs generated"
echo "✅ Stage 2: 48th blocks generated"
echo ""
echo "📁 Results:"
echo "   - Complete blocks: /media/visitor/Extreme SSD/data/complete_blocks/"
echo "   - Endpoint blocks: /media/visitor/Extreme SSD/data/endpoint_blocks/"
echo ""
echo "💡 Next steps:"
echo "   - Use complete_blocks/ for neural ODE training"
echo "   - Use endpoint_blocks/ for structure prediction"
echo ""
echo "🎯 Pipeline completed successfully!"