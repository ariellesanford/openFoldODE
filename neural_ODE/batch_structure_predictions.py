#!/usr/bin/env python3
"""
Batch Structure Prediction Script
Runs generate_all_structure_predictions.py for all proteins in testing_chains.txt
"""

import os
import sys
import subprocess
from pathlib import Path
import time
from datetime import datetime


def read_testing_chains(splits_dir: Path) -> list:
    """Read protein IDs from testing_chains.txt"""
    testing_file = splits_dir / "testing_chains.txt"

    if not testing_file.exists():
        print(f"❌ Testing chains file not found: {testing_file}")
        sys.exit(1)

    proteins = []
    with open(testing_file, 'r') as f:
        for line in f:
            protein_id = line.strip()
            if protein_id:
                proteins.append(protein_id)

    return proteins


def run_structure_prediction(protein_id: str, script_path: Path, args: list) -> bool:
    """Run structure prediction for a single protein"""
    cmd = [sys.executable, str(script_path), "--pdb_id", protein_id] + args

    print(f"🚀 Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"💥 Error: {e}")
        return False


def main():
    # Configuration - EDIT THESE FOR PYCHARM EXECUTION
    script_dir = Path(__file__).parent

    # =======================================================================================
    # PYCHARM CONFIGURATION - Edit these variables to run directly in PyCharm
    # =======================================================================================

    SPLITS_FOLDER = "full"  # Change this to your desired splits folder
    ADDITIONAL_ARGS = [
        # "--use_cpu",           # Uncomment to force CPU
        # "--skip_relaxation",   # Uncomment to skip relaxation
        # "--run_metrics", "False"  # Uncomment to disable metrics
    ]

    # =======================================================================================
    # Command line parsing (will override PyCharm config if arguments provided)
    # =======================================================================================

    if len(sys.argv) >= 2:
        # Use command line arguments
        splits_folder = sys.argv[1]
        additional_args = sys.argv[2:]
        print("🖥️  Using command line arguments")
    else:
        # Use PyCharm configuration
        splits_folder = SPLITS_FOLDER
        additional_args = ADDITIONAL_ARGS
        print("🔧 Using PyCharm configuration")
        print(f"   To change settings, edit SPLITS_FOLDER and ADDITIONAL_ARGS variables at the top of main()")

    # Show current configuration
    print(f"📁 Splits folder: {splits_folder}")
    if additional_args:
        print(f"🔧 Additional args: {' '.join(additional_args)}")

    # Validate splits folder exists
    splits_base = script_dir / "data_splits"
    if not splits_base.exists():
        print(f"❌ Data splits directory not found: {splits_base}")
        print("Available options: Create data_splits folder or check path")
        sys.exit(1)

    available_splits = [f.name for f in splits_base.iterdir() if f.is_dir()]
    if splits_folder not in available_splits:
        print(f"❌ Splits folder '{splits_folder}' not found")
        print(f"Available splits folders: {', '.join(available_splits)}")
        print("Edit SPLITS_FOLDER variable or provide as command line argument")
        sys.exit(1)

    # Paths
    splits_dir = script_dir / "data_splits" / splits_folder
    prediction_script = script_dir / "generate_all_structure_predictions.py"

    # Validate paths
    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        sys.exit(1)

    if not prediction_script.exists():
        print(f"❌ Prediction script not found: {prediction_script}")
        sys.exit(1)

    # Read testing proteins
    proteins = read_testing_chains(splits_dir)

    print("🧬 BATCH STRUCTURE PREDICTION")
    print("=" * 50)
    print(f"📁 Splits folder: {splits_folder}")
    print(f"📋 Testing proteins: {len(proteins)}")
    print(f"📜 Script: {prediction_script.name}")
    if additional_args:
        print(f"🔧 Additional args: {' '.join(additional_args)}")
    print("")

    # Show first few proteins
    print("🧬 Proteins to process:")
    for i, protein in enumerate(proteins[:10]):
        print(f"  {i + 1:2d}. {protein}")
    if len(proteins) > 10:
        print(f"  ... and {len(proteins) - 10} more")
    print("")

    # Confirm before starting
    response = input(f"🚀 Process all {len(proteins)} proteins? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Aborted by user")
        sys.exit(0)

    # Run predictions
    successful = 0
    failed = 0
    start_time = time.time()

    for i, protein_id in enumerate(proteins, 1):
        print(f"\n[{i}/{len(proteins)}] Processing {protein_id}")
        print("-" * 40)

        protein_start = time.time()

        if run_structure_prediction(protein_id, prediction_script, additional_args):
            successful += 1
            print(f"✅ {protein_id} completed successfully")
        else:
            failed += 1
            print(f"❌ {protein_id} failed")

            # Ask whether to continue on failure
            if i < len(proteins):
                response = input(f"\n⚠️  Continue with remaining proteins? [y/N]: ")
                if response.lower() != 'y':
                    print("❌ Stopping batch processing")
                    break

        protein_time = time.time() - protein_start
        print(f"⏱️  Time: {protein_time / 60:.1f} minutes")

    total_time = time.time() - start_time

    # Final summary
    print("\n" + "=" * 50)
    print("📊 BATCH PROCESSING COMPLETE")
    print("=" * 50)
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {successful / (successful + failed) * 100:.1f}%")

    print("\n🎯 Structure predictions completed!")
    print("📁 Check individual protein directories for results:")
    print("   /media/visitor/Extreme SSD/data/structure_predictions/")


if __name__ == "__main__":
    main()