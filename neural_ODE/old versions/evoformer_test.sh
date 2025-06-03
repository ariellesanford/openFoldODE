#!/bin/bash
# Improved test of evoformer iteration with better path handling

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find project root by looking for openfold directory
PROJECT_ROOT=""
if [ -d "$SCRIPT_DIR/../openfold" ]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
elif [ -d "$SCRIPT_DIR/openfold" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "❌ Could not find openfold directory. Please run from the correct location."
    exit 1
fi

echo "🔍 Project root: $PROJECT_ROOT"

# Find a test protein
TEST_PROTEIN_DIR=""
PROTEIN_NAME=""

# Look in different possible data directories
DATA_DIRS=(
    "$SCRIPT_DIR/mini_data"
    "$SCRIPT_DIR/data"
    "$PROJECT_ROOT/neural_ODE/data"
    "$PROJECT_ROOT/data"
)

for base_dir in "${DATA_DIRS[@]}"; do
    if [ ! -d "$base_dir" ]; then
        continue
    fi

    echo "🔍 Searching in: $base_dir"

    for split in training validation testing; do
        BLOCKS_DIR="$base_dir/$split/blocks"
        if [ -d "$BLOCKS_DIR" ]; then
            echo "   Checking: $BLOCKS_DIR"
            for protein_dir in "$BLOCKS_DIR"/*_evoformer_blocks; do
                if [ -d "$protein_dir/recycle_0" ]; then
                    if [ -f "$protein_dir/recycle_0/m_block_0.pt" ] && [ -f "$protein_dir/recycle_0/z_block_0.pt" ]; then
                        TEST_PROTEIN_DIR="$protein_dir/recycle_0"
                        PROTEIN_NAME=$(basename "$protein_dir" | sed 's/_evoformer_blocks//')
                        echo "   ✅ Found test protein: $PROTEIN_NAME"
                        break
                    fi
                fi
            done
            if [ -n "$TEST_PROTEIN_DIR" ]; then
                break
            fi
        fi
    done

    # Also check direct structure (no splits)
    if [ -z "$TEST_PROTEIN_DIR" ]; then
        echo "   Checking direct structure in: $base_dir"
        for protein_dir in "$base_dir"/*_evoformer_blocks; do
            if [ -d "$protein_dir/recycle_0" ]; then
                if [ -f "$protein_dir/recycle_0/m_block_0.pt" ] && [ -f "$protein_dir/recycle_0/z_block_0.pt" ]; then
                    TEST_PROTEIN_DIR="$protein_dir/recycle_0"
                    PROTEIN_NAME=$(basename "$protein_dir" | sed 's/_evoformer_blocks//')
                    echo "   ✅ Found test protein: $PROTEIN_NAME"
                    break
                fi
            fi
        done
    fi

    if [ -n "$TEST_PROTEIN_DIR" ]; then
        break
    fi
done

if [ -z "$TEST_PROTEIN_DIR" ]; then
    echo "❌ No test protein found in any data directory"
    echo "📁 Searched directories:"
    for dir in "${DATA_DIRS[@]}"; do
        echo "   - $dir"
    done
    exit 1
fi

echo "🧪 Testing evoformer iteration"
echo "📁 Test protein: $PROTEIN_NAME"
echo "📂 Directory: $TEST_PROTEIN_DIR"

# Check input files
M_PATH="$TEST_PROTEIN_DIR/m_block_0.pt"
Z_PATH="$TEST_PROTEIN_DIR/z_block_0.pt"

echo "📄 Input files:"
echo "   M: $M_PATH ($([ -f "$M_PATH" ] && echo "✅ exists" || echo "❌ missing"))"
echo "   Z: $Z_PATH ($([ -f "$Z_PATH" ] && echo "✅ exists" || echo "❌ missing"))"

if [ ! -f "$M_PATH" ] || [ ! -f "$Z_PATH" ]; then
    echo "❌ Required input files are missing"
    exit 1
fi

# Find the evoformer iteration script
EVOFORMER_SCRIPT=""

# Check multiple possible locations
SCRIPT_LOCATIONS=(
    "$PROJECT_ROOT/evoformer_iter/run_evoformer_iter.py"
    "$SCRIPT_DIR/evoformer_iter/run_evoformer_iter.py"
    "$SCRIPT_DIR/run_evoformer_iter.py"
    "$PROJECT_ROOT/run_evoformer_iter.py"
)

echo "🔍 Looking for evoformer script..."
for script_path in "${SCRIPT_LOCATIONS[@]}"; do
    echo "   Checking: $script_path"
    if [ -f "$script_path" ]; then
        EVOFORMER_SCRIPT="$script_path"
        echo "   ✅ Found: $script_path"
        break
    else
        echo "   ❌ Not found"
    fi
done

if [ -z "$EVOFORMER_SCRIPT" ]; then
    echo "❌ No evoformer script found in any location"
    exit 1
fi

echo "📜 Evoformer script: $EVOFORMER_SCRIPT"

# Get Python path
if [ -n "${CONDA_PREFIX}" ]; then
    PYTHON_PATH="${CONDA_PREFIX}/bin/python"
else
    PYTHON_PATH=$(which python)
fi

echo "🐍 Python: $PYTHON_PATH"

# Check if parameters file exists by running a quick check
echo "🔍 Checking parameter file access..."
PARAM_CHECK_OUTPUT=$("$PYTHON_PATH" -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from evoformer_iter.run_evoformer_iter import find_params_file, find_project_root
    project_root = find_project_root()
    params_file = find_params_file('model_1_ptm', project_root)
    print(f'✅ Parameters file found: {params_file}')
except Exception as e:
    print(f'❌ Error finding parameters: {e}')
    sys.exit(1)
" 2>&1)

echo "$PARAM_CHECK_OUTPUT"

if [[ "$PARAM_CHECK_OUTPUT" == *"❌"* ]]; then
    echo "❌ Parameter file check failed"
    exit 1
fi

# Build command
CMD="$PYTHON_PATH $EVOFORMER_SCRIPT \
    --m_path $M_PATH \
    --z_path $Z_PATH \
    --output_dir $TEST_PROTEIN_DIR \
    --config_preset model_1_ptm \
    --device cuda:0"

echo ""
echo "🔧 Running command:"
echo "$CMD"
echo ""

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "🌍 PYTHONPATH: $PYTHONPATH"

# Change to project root directory for execution
cd "$PROJECT_ROOT"
echo "📁 Working directory: $(pwd)"

# Run the command
echo "⏳ Starting evoformer iteration..."
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
    echo ""
    echo "🎉 SUCCESS: Evoformer iteration worked!"
    echo "✅ The path resolution fix is working correctly"

    # Show file sizes for verification
    echo "📊 Output file info:"
    echo "   $(ls -lh "$M_OUT" | awk '{print "M block: " $5 " " $9}')"
    echo "   $(ls -lh "$Z_OUT" | awk '{print "Z block: " $5 " " $9}')"
else
    echo ""
    echo "❌ FAILED: Check the error messages above"
    echo "🔍 Debug info:"
    echo "   Return code: $RETURN_CODE"
    echo "   M output exists: $([ -f "$M_OUT" ] && echo "yes" || echo "no")"
    echo "   Z output exists: $([ -f "$Z_OUT" ] && echo "yes" || echo "no")"

    # Show what files are actually in the directory
    echo "📁 Contents of output directory:"
    ls -la "$TEST_PROTEIN_DIR/"
fi