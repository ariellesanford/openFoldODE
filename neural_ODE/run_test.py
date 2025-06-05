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


def find_latest_model(outputs_dir: Path) -> Path:
    """Find the most recent model file in outputs directory"""

    # Look for model files
    model_patterns = [
        "*_final_model.pt",
        "*_model.pt",
        "*.pt"
    ]

    model_files = []
    for pattern in model_patterns:
        model_files.extend(outputs_dir.glob(pattern))

    if not model_files:
        return None

    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return model_files[0]


def main():
    # Get script directory and set up paths
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    data_dir = Path("/media/visitor/Extreme SSD/data/complete_blocks")
    splits_dir = script_dir / "data_splits" / "mini"
    test_script = script_dir / "test_model.py"

    print("🧪 NEURAL ODE MODEL TESTING RUNNER")
    print("=" * 50)

    # Check if test script exists
    if not test_script.exists():
        print(f"❌ Test script not found: {test_script}")
        return 1

    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
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

    # Find the latest model
    print(f"🔍 Looking for models in: {outputs_dir}")

    latest_model = find_latest_model(outputs_dir)

    if not latest_model:
        print(f"❌ No model files found in {outputs_dir}")
        print(f"   Expected files like: *_final_model.pt, *_model.pt, *.pt")
        return 1

    print(f"📦 Found latest model: {latest_model.name}")

    # Show model info
    try:
        model_size_mb = latest_model.stat().st_size / 1024 / 1024
        model_date = datetime.fromtimestamp(latest_model.stat().st_mtime)
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
        "--model_path", str(latest_model),
        "--data_dir", str(data_dir),
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        cmd.extend(["--output_file", results_file])
        print(f"💾 Results will be saved to: {results_file}")

    # Add prediction saving (always save for structure module)
    cmd.append("--save_predictions")
    predictions_dir = "post_evoformer_predictions"
    cmd.extend(["--predictions_dir", predictions_dir])
    print(f"📦 Predictions will be saved to: {predictions_dir}/")

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
                if Path(results_file).exists():
                    size_mb = Path(results_file).stat().st_size / 1024 / 1024
                    print(f"  - {results_file} ({size_mb:.1f} MB) [Test results]")

            # Check predictions directory
            pred_path = Path(predictions_dir)
            if pred_path.exists():
                protein_dirs = [d for d in pred_path.iterdir() if d.is_dir()]
                print(f"  - {predictions_dir}/ ({len(protein_dirs)} proteins) [Structure module ready]")

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
            print(f"   Use predictions in: {predictions_dir}/")

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
    print("Automatically finds latest model and runs comprehensive testing")
    print("")
    print("Usage:")
    print("  python run_test.py                 # Basic test on all test proteins")
    print("  python run_test.py --quick         # Quick test (first 5 proteins)")
    print("  python run_test.py --save-results  # Save detailed JSON results")
    print("  python run_test.py --small-proteins# Test only small proteins (≤200 residues)")
    print("  python run_test.py --cpu           # Force CPU testing")
    print("")
    print("Features:")
    print("  🔍 Auto-finds latest trained model")
    print("  🧪 Tests on all proteins in test split")
    print("  📊 Comprehensive evaluation metrics")
    print("  💾 Saves predictions for OpenFold structure module")
    print("  📁 Creates post_evoformer_predictions/ directory")
    print("")

    exit(main())