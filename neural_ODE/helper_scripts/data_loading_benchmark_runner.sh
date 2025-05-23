#!/bin/bash
set -e

# =======================================================================================
# DATA LOADING STRATEGY BENCHMARK
# This script tests different data loading approaches to find the optimal strategy
# =======================================================================================

# Get the actual directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level from helper_scripts to neural_ODE

# =======================================================================================
# CONFIGURATION
# =======================================================================================

# Data directories
DATA_DIR="${ROOT_DIR}/data/quick_inference_data"
TEMP_DIR="${ROOT_DIR}/benchmark_temp"
TRAINING_SCRIPT="${ROOT_DIR}/train_evoformer_ode.py"
OUTPUT_FILE="${ROOT_DIR}/benchmark_results.json"

# Benchmark settings
QUICK_TEST=true  # Set to false for comprehensive test

# =======================================================================================
# END CONFIGURATION
# =======================================================================================

echo "🧪 DATA LOADING STRATEGY BENCHMARK"
echo "=================================="
echo "Data directory: $DATA_DIR"
echo "Temp directory: $TEMP_DIR"
echo "Training script: $TRAINING_SCRIPT"
echo "Output file: $OUTPUT_FILE"

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please ensure you have generated protein data first."
    exit 1
fi

# Check if training script exists
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "❌ Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

# Count available proteins
PROTEIN_COUNT=$(ls -1 "$DATA_DIR" | grep "_evoformer_blocks" | wc -l)
echo "📊 Found $PROTEIN_COUNT proteins in data directory"

if [[ $PROTEIN_COUNT -eq 0 ]]; then
    echo "❌ No protein data found! Please generate data first."
    exit 1
fi

# Get Python path
if [ -n "${CONDA_PREFIX}" ]; then
    PYTHON_PATH="${CONDA_PREFIX}/bin/python"
else
    PYTHON_PATH=$(which python)
fi
echo "🐍 Using Python: $PYTHON_PATH"

# Create the benchmark script if it doesn't exist
BENCHMARK_SCRIPT="${SCRIPT_DIR}/data_loading_benchmark.py"
if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
    echo "❌ Benchmark script not found: $BENCHMARK_SCRIPT"
    echo "Please ensure data_loading_benchmark.py is in the helper_scripts directory."
    exit 1
fi

# Prepare benchmark command
BENCHMARK_CMD=(
    "$PYTHON_PATH" "$BENCHMARK_SCRIPT"
    "--data_dir" "$DATA_DIR"
    "--temp_dir" "$TEMP_DIR"
    "--training_script" "$TRAINING_SCRIPT"
    "--output" "$OUTPUT_FILE"
)

# Add quick flag if enabled
if [[ "$QUICK_TEST" == "true" ]]; then
    BENCHMARK_CMD+=("--quick")
    echo "⚡ Running quick benchmark (faster but less comprehensive)"
else
    echo "🔬 Running comprehensive benchmark (slower but thorough)"
fi

# Show what will be tested
echo ""
echo "📋 BENCHMARK PLAN:"
echo "1. Strategy 1: Load one protein at a time, train separately"
echo "   - Tests each protein individually"
echo "   - Measures per-protein overhead"
echo ""
echo "2. Strategy 2: Load all proteins at once, train together"
echo "   - Tests your current approach"
echo "   - Measures batch processing efficiency"
echo ""
echo "3. Strategy 3: Load one timestep for all proteins, train, repeat"
echo "   - Tests timestep-by-timestep approach"
echo "   - Measures timestep overhead vs storage efficiency"
echo ""

# Ask for confirmation unless in automated mode
if [[ "${AUTOMATED:-false}" != "true" ]]; then
    echo "⚠️  This benchmark will:"
    echo "   - Use temporary storage (will be cleaned up)"
    echo "   - Run multiple training tests (may take 10-30 minutes)"
    echo "   - Generate detailed performance data"
    echo ""
    read -p "Do you want to proceed? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Benchmark cancelled."
        exit 0
    fi
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run the benchmark
echo ""
echo "🚀 Starting benchmark..."
echo "Command: ${BENCHMARK_CMD[*]}"
echo ""

# Execute benchmark
"${BENCHMARK_CMD[@]}"

# Check if benchmark completed successfully
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    echo "📄 Results saved to: $OUTPUT_FILE"

    # Show quick summary if results file exists
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo ""
        echo "📊 QUICK SUMMARY:"
        echo "Check the full results in $OUTPUT_FILE"
        echo "The benchmark script will have printed the recommendation above."
    fi

    echo ""
    echo "🎯 NEXT STEPS:"
    echo "1. Review the detailed results in $OUTPUT_FILE"
    echo "2. Look for the RECOMMENDATION section in the output above"
    echo "3. Implement the fastest strategy for your training pipeline"

else
    echo ""
    echo "❌ Benchmark failed!"
    echo "Check the error messages above for details."
    exit 1
fi

# Clean up temp directory
if [[ -d "$TEMP_DIR" ]]; then
    echo "🧹 Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
fi

echo ""
echo "🏁 Benchmark complete!"