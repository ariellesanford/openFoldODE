#!/usr/bin/env python3
"""
Script to run basic testing on all test proteins
Automatically finds the latest model and runs testing
"""

import os
import sys
import subprocess
from pathlib import Path
import glob
from datetime import datetime
import re
import argparse


def find_specific_model(outputs_dir: Path, model_name: str) -> Path:
    """Find the specific model file"""
    model_path = outputs_dir / model_name

    if model_path.exists():
        return model_path
    else:
        return None


def extract_full_model_name_from_path(model_path: Path) -> str:
    """Extract full model name from model filename like 20250613_180436_baseline_no_prelim_final_model.pt"""
    model_name = model_path.stem  # Remove .pt extension

    # Remove _final_model suffix if present
    if model_name.endswith('_final_model'):
        model_name = model_name[:-12]  # Remove '_final_model'

    return model_name


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test neural ODE models')
    parser.add_argument('model', nargs='?', default=None, help='Specific model name to test')
    parser.add_argument('--data-dir', type=str, required=True, help='Base data directory')
    parser.add_argument('--splits-dir', type=str, required=True, help='Splits directory')
    args = parser.parse_args()

    # Get script directory and set up paths
    script_dir = Path(__file__).parent

    # Support multiple data directories based on data-dir argument
    data_dirs = [
        f"{args.data_dir}/complete_blocks",
        f"{args.data_dir}/endpoint_blocks",
    ]

    splits_dir = Path(args.splits_dir)
    test_script = script_dir / "test_model.py"
    outputs_dir = script_dir / "trained_models"

    # Use model from command line if provided
    specific_model_name = args.model

    print("🧪 NEURAL ODE MODEL TESTING RUNNER")
    print("=" * 50)

    # Check if test script exists
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return 1

    # Check data directories
    valid_data_dirs = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            valid_data_dirs.append(str(data_path))
            print(f"✅ Found data directory: {data_path}")
        else:
            print(f"⚠️  Data directory not found: {data_path}")

    if not valid_data_dirs:
        print(f"❌ No valid data directories found!")
        return 1

    # Check if splits directory exists
    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        return 1

    # Check if outputs directory exists
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        print(f"   Have you trained a model yet?")
        return 1

    # Find model to test
    if specific_model_name:
        # User specified a model
        model_path = find_specific_model(outputs_dir, specific_model_name)
        if not model_path:
            print(f"❌ Specified model not found: {specific_model_name}")
            print(f"   Looking in: {outputs_dir}")
            return 1
    else:
        # Find the latest model
        print("🔍 Looking for trained models...")
        model_files = list(outputs_dir.glob("*_final_model.pt"))

        if not model_files:
            print(f"❌ No trained models found in {outputs_dir}")
            print(f"   Train a model first using train_evoformer_ode.py")
            return 1

        # Sort by modification time to get the latest
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model_path = latest_model
        print(f"📦 Found latest model: {latest_model.name}")

    # Extract full model name for output directory
    model_full_name = extract_full_model_name_from_path(model_path)

    # Build test command with multiple data directories
    cmd = [
        sys.executable,
        str(test_script),
        "--model_path", str(model_path),
        "--data_dirs", *valid_data_dirs,
        "--splits_dir", str(splits_dir),
        "--save_predictions",
        "--predictions_dir", f"post_evoformer_predictions/{model_full_name}"
    ]

    print(f"\n🚀 Running test with model: {model_path.name}")
    print(f"📁 Data directories: {len(valid_data_dirs)}")
    print(f"📂 Splits directory: {splits_dir}")
    print(f"💾 Predictions will be saved to: post_evoformer_predictions/{model_full_name}")
    print(f"\n🖥️  Command: {' '.join(cmd)}")
    print("=" * 50)

    # Run the test
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())