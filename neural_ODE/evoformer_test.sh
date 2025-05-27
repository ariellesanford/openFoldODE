#!/bin/bash
# Manual test of evoformer iteration

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find a test protein
TEST_PROTEIN_DIR=""
for split in training validation testing; do
    BLOCKS_DIR="$SCRIPT_DIR/mini_data/$split/blocks"
    if [ -d "$BLOCKS_DIR" ]; then
        for protein_dir in "$BLOCKS_DIR"/*_evoformer_blocks; do
            if [ -d "$protein_dir/recycle_0" ]; then
                if [ -f "$protein_dir/recycle_0/m_block_0.pt" ] && [ -f "$protein_dir/recycle_0/z_block_0.pt" ]; then
                    TEST_PROTEIN_DIR="$protein_dir/recycle_0"
                    PROTEIN_NAME=$(basename "$protein_dir" | sed 's/_evoformer_blocks//')
                    break
                fi
            fi
        done
        if [ -n "$TEST_PROTEIN_DIR" ]; then
            break
        fi
    fi
done

if [ -z "$TEST_PROTEIN_DIR" ]; then
    echo "❌ No test protein found in mini_data"
    exit 1
fi

echo "🧪 Testing evoformer iteration manually"
echo "📁 Test protein: $PROTEIN_NAME"
echo "📂 Directory: $TEST_PROTEIN_DIR"

# Check input files
M_PATH="$TEST_PROTEIN_DIR/m_block_0.pt"
Z_PATH="$TEST_PROTEIN_DIR/z_block_0.pt"

echo "📄 Input files:"
echo "   M: $M_PATH ($([ -f "$M_PATH" ] && echo "✅ exists" || echo "❌ missing"))"
echo "   Z: $Z_PATH ($([ -f "$Z_PATH" ] && echo "✅ exists" || echo "❌ missing"))"

# Try to call the evoformer iteration script
# evoformer_iter is at the same level as neural_ODE, not inside it
EVOFORMER_SCRIPT="$(dirname "$SCRIPT_DIR")/evoformer_iter/run_evoformer_iter.py"

if [ ! -f "$EVOFORMER_SCRIPT" ]; then
    echo "❌ Evoformer script not found: $EVOFORMER_SCRIPT"
    echo "🔍 Looking for alternatives..."

    # Try other possible locations
    ALT_LOCATIONS=(
        "$SCRIPT_DIR/evoformer_iter/run_evoformer_iter.py"
        "$SCRIPT_DIR/run_evoformer_iter.py"
        "$(dirname "$SCRIPT_DIR")/evoformer_iter_script.sh"
        "$SCRIPT_DIR/evoformer_iter_script.sh"
    )

    for alt in "${ALT_LOCATIONS[@]}"; do
        echo "   Checking: $alt"
        if [ -f "$alt" ]; then
            EVOFORMER_SCRIPT="$alt"
            echo "   ✅ Found: $alt"
            break
        else
            echo "   ❌ Not found"
        fi
    done

    if [ ! -f "$EVOFORMER_SCRIPT" ]; then
        echo "❌ No evoformer script found in any location"
        exit 1
    fi
fi

echo "📜 Evoformer script: $EVOFORMER_SCRIPT"

# Get Python path
if [ -n "${CONDA_PREFIX}" ]; then
    PYTHON_PATH="${CONDA_PREFIX}/bin/python"
else
    PYTHON_PATH=$(which python)
fi

echo "🐍 Python: $PYTHON_PATH"

# Build command
if [[ "$EVOFORMER_SCRIPT" == *.sh ]]; then
    # Shell script
    CMD="bash $EVOFORMER_SCRIPT \
        --m_path $M_PATH \
        --z_path $Z_PATH \
        --output_dir $TEST_PROTEIN_DIR \
        --config_preset model_1_ptm \
        --device cuda:0"
else
    # Python script
    CMD="$PYTHON_PATH $EVOFORMER_SCRIPT \
        --m_path $M_PATH \
        --z_path $Z_PATH \
        --output_dir $TEST_PROTEIN_DIR \
        --config_preset model_1_ptm \
        --device cuda:0"
fi

echo ""
echo "🔧 Running command:"
echo "$CMD"
echo ""

# Set PYTHONPATH to include the main project directory
export PYTHONPATH="$(dirname "$SCRIPT_DIR"):$PYTHONPATH"

# Run the command
eval "$CMD"

RETURN_CODE=$?

echo ""
echo "📊 Return code: $RETURN_CODE"

# Check output files
M_OUT="$TEST_PROTEIN_DIR/m_block_1.pt"
Z_OUT="$TEST_PROTEIN_DIR/z_block_1.pt"

echo "📄 Output files:"
echo "   M: $M_OUT ($([ -f "$M_OUT" ] && echo "✅ created" || echo "❌ missing"))"
echo "   Z: $Z_OUT ($([ -f "$Z_OUT" ] && echo "✅ created" || echo "❌ missing"))"

if [ $RETURN_CODE -eq 0 ] && [ -f "$M_OUT" ] && [ -f "$Z_OUT" ]; then
    echo "🎉 SUCCESS: Evoformer iteration worked!"
else
    echo "❌ FAILED: Check the error messages above"
fi