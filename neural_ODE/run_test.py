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
    # Get script directory and set up paths
    script_dir = Path(__file__).parent
    # Support multiple data directories
    data_dirs = [
        "/media/visitor/Extreme SSD/data/complete_blocks",
        "/media/visitor/Extreme SSD/data/endpoint_blocks",
    ]
    splits_dir = script_dir / "data_splits" / "jumbo"  # Changed to jumbo
    test_script = script_dir / "test_model.py"
    outputs_dir = script_dir / "trained_models"

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

    # Find the specific model
    target_model_name = "20250618_174724_full_ode_with_prelim2_final_model.pt"
    print(f"🔍 Looking for specific model: {target_model_name}")

    model_path = find_specific_model(outputs_dir, target_model_name)

    if not model_path:
        print(f"❌ Model file not found: {target_model_name}")
        print(f"   Looking in: {outputs_dir}")
        # List available models for debugging
        available_models = list(outputs_dir.glob("*.pt"))
        if available_models:
            print(f"   Available models:")
            for model in available_models:
                print(f"     - {model.name}")
        return 1

    print(f"📦 Found model: {model_path.name}")

    # Extract full model name and create predictions directory name
    full_model_name = extract_full_model_name_from_path(model_path)
    predictions_dir_name = f"predictions_{full_model_name}"
    # Use first valid data directory for predictions
    predictions_base_dir = Path(valid_data_dirs[0]).parent / "post_evoformer_predictions"
    predictions_dir = predictions_base_dir / predictions_dir_name

    print(f"🏷️  Full model name: {full_model_name}")
    print(f"📁 Predictions will be saved to: {predictions_dir}")

    # Show model info
    try:
        model_size_mb = model_path.stat().st_size / 1024 / 1024
        model_date = datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"   Size: {model_size_mb:.1f} MB")
        print(f"   Modified: {model_date.strftime('%Y-%m-%d %H:%M:%S')}")
    except:
        pass

    # Parse command line arguments for options
    save_results = '--save-results' in sys.argv or '--save' in sys.argv
    quick_test = '--quick' in sys.argv or '--quick-test' in sys.argv
    small_proteins = '--small-proteins' in sys.argv
    cpu_only = '--cpu' in sys.argv or 'cpu' in sys.argv

    # Build command
    cmd = [
              sys.executable,
              str(test_script),
              "--model_path", str(model_path),
              "--data_dirs"] + valid_data_dirs + [
              "--splits_dir", str(splits_dir)
          ]

    # Add device selection
    if cpu_only:
        cmd.extend(["--device", "cpu"])
        print("💻 Using CPU for testing")
    else:
        cmd.extend(["--device", "cuda"])
        print("🚀 Using CUDA for testing")

    # Add optional filters
    if quick_test:
        cmd.extend(["--max_proteins", "5"])
        print("⚡ Quick test mode: testing first 5 proteins only")

    if small_proteins:
        cmd.extend(["--max_residues", "200"])
        print("📏 Small proteins mode: proteins ≤200 residues only")

    # Add result saving
    if save_results:
        results_file = f"test_results_{full_model_name}.json"
        cmd.extend(["--output_file", results_file])
        print(f"💾 Results will be saved to: {results_file}")

    # Add prediction saving (always save for structure module)
    cmd.append("--save_predictions")
    cmd.extend(["--predictions_dir", str(predictions_dir)])

    print(f"\n🚀 Running test command:")
    print(f"   {' '.join(cmd)}")
    print(f"\n" + "=" * 50)

    try:
        # Run the test
        result = subprocess.run(cmd, cwd=script_dir)

        if result.returncode == 0:
            print(f"\n✅ Testing completed successfully!")

            # Show what was created
            print(f"\n📁 Files created:")
            if save_results:
                results_path = Path(f"test_results_{full_model_name}.json")
                if results_path.exists():
                    size_mb = results_path.stat().st_size / 1024 / 1024
                    print(f"  - {results_path.name} ({size_mb:.1f} MB) [Test results]")

            # Check predictions directory
            if predictions_dir.exists():
                protein_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
                print(f"  - {predictions_dir_name}/ ({len(protein_dirs)} proteins) [Structure module ready]")

                # Show a few examples
                for i, protein_dir in enumerate(protein_dirs[:3]):
                    print(f"    └── {protein_dir.name}/")
                    files = list(protein_dir.glob("*.pt")) + list(protein_dir.glob("*.json"))
                    for file in files:
                        size_kb = file.stat().st_size / 1024
                        print(f"        ├── {file.name} ({size_kb:.1f} KB)")

                if len(protein_dirs) > 3:
                    print(f"        └── ... and {len(protein_dirs) - 3} more proteins")

            print(f"\n🎯 Ready for OpenFold structure module!")
            print(f"   Use predictions in: post_evoformer_predictions/{predictions_dir_name}/")

        else:
            print(f"\n❌ Testing failed with return code: {result.returncode}")
            return result.returncode

    except KeyboardInterrupt:
        print(f"\n⏹️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return 1

    return 0


if __name__ == "__main__":
    print("🧪 Neural ODE Model Testing Runner")
    print("Automatically finds specific model and runs comprehensive testing")
    print("")
    print("Usage:")
    print("  python run_test.py                 # Basic test on all test proteins")
    print("  python run_test.py --quick         # Quick test (first 5 proteins)")
    print("  python run_test.py --save-results  # Save detailed JSON results")
    print("  python run_test.py --small-proteins# Test only small proteins (≤200 residues)")
    print("  python run_test.py --cpu           # Force CPU testing")
    print("")
    print("Features:")
    print("  🎯 Uses specific model: 20250613_180436_baseline_no_prelim_final_model.pt")
    print("  🧪 Tests on all proteins in test split (jumbo)")
    print("  📊 Comprehensive evaluation metrics")
    print("  💾 Saves predictions for OpenFold structure module")
    print("  📁 Creates full-model-name predictions directory")
    print("")

    exit(main())