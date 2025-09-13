#!/usr/bin/env python3
"""
Generate 48 evoformer blocks for all proteins and move to complete_blocks when done
Modified to accept data_dir as command line argument
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import time
import argparse


def find_python():
    """Find the appropriate Python interpreter"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_python = Path(conda_prefix) / "bin" / "python"
        if conda_python.exists():
            return str(conda_python)
    return sys.executable


def find_evoformer_script():
    """Find the evoformer iteration script"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    possible_locations = [
        project_root / "save_intermediates" / "run_evoformer_iter.py",
        script_dir / "save_intermediates" / "run_evoformer_iter.py",
        script_dir / "run_evoformer_iter.py"
    ]

    for script_path in possible_locations:
        if script_path.exists():
            return script_path

    raise FileNotFoundError("Could not find evoformer iteration script")


def get_last_block_index(protein_dir):
    """Get the highest existing block index for a protein"""
    recycle_dir = protein_dir / "recycle_0"
    if not recycle_dir.exists():
        return -1

    m_blocks = list(recycle_dir.glob("m_block_*.pt"))
    z_blocks = list(recycle_dir.glob("z_block_*.pt"))

    if not m_blocks or not z_blocks:
        return -1

    def extract_index(filename):
        try:
            return int(filename.stem.split('_')[-1])
        except:
            return -1

    m_indices = [extract_index(f) for f in m_blocks if extract_index(f) >= 0]
    z_indices = [extract_index(f) for f in z_blocks if extract_index(f) >= 0]

    if not m_indices or not z_indices:
        return -1

    return min(max(m_indices), max(z_indices))


def run_single_iteration(protein_dir, current_index, evoformer_script, python_path, project_root, device="cuda:0"):
    """Run a single evoformer iteration"""
    recycle_dir = protein_dir / "recycle_0"

    m_path = recycle_dir / f"m_block_{current_index}.pt"
    z_path = recycle_dir / f"z_block_{current_index}.pt"

    if not m_path.exists() or not z_path.exists():
        return False

    # Build command
    cmd = [
        python_path,
        str(evoformer_script),
        "--m_path", str(m_path),
        "--z_path", str(z_path),
        "--output_dir", str(recycle_dir),
        "--config_preset", "model_1_ptm",
        "--device", device
    ]

    # Set up environment
    env = os.environ.copy()
    pythonpath = str(project_root)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{pythonpath}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = pythonpath

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(project_root),
            env=env
        )

        # Check if output files were created
        next_m = recycle_dir / f"m_block_{current_index + 1}.pt"
        next_z = recycle_dir / f"z_block_{current_index + 1}.pt"

        return result.returncode == 0 and next_m.exists() and next_z.exists()

    except subprocess.TimeoutExpired:
        print(f"    ⏰ Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"    💥 Error: {e}")
        return False


def generate_blocks_for_protein(protein_dir, target_blocks, evoformer_script, python_path, project_root):
    """Generate all blocks for a single protein up to target_blocks"""
    protein_name = protein_dir.name.replace('_evoformer_blocks', '')

    # Check current state
    last_block = get_last_block_index(protein_dir)

    if last_block >= target_blocks:
        print(f"✅ {protein_name}: Already complete (block {last_block})")
        return True

    if last_block < 0:
        print(f"❌ {protein_name}: No valid starting blocks found")
        return False

    print(f"🔄 {protein_name}: Generating blocks {last_block + 1} to {target_blocks}")

    current_block = last_block
    failures = 0
    max_failures = 3

    while current_block < target_blocks:
        iteration_num = current_block + 1
        print(f"  [{iteration_num}/{target_blocks}] Block {current_block} → {iteration_num}... ", end='', flush=True)

        if run_single_iteration(protein_dir, current_block, evoformer_script, python_path, project_root):
            print("✅")
            current_block += 1
            failures = 0
        else:
            print("❌")
            failures += 1
            if failures >= max_failures:
                print(f"  💀 Failed {failures} times on block {iteration_num}, stopping")
                return False
            # Don't increment current_block - retry the same iteration

    print(f"🎉 {protein_name}: All {target_blocks} blocks generated!")
    return True


def move_protein_to_complete(protein_dir, complete_dir):
    """Move completed protein to complete_blocks directory"""
    protein_name = protein_dir.name
    target_path = complete_dir / protein_name

    if target_path.exists():
        print(f"  ⚠️  Target already exists: {target_path}")
        return False

    try:
        shutil.move(str(protein_dir), str(target_path))
        print(f"  📦 Moved to: {target_path}")
        return True
    except Exception as e:
        print(f"  ❌ Move failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate all 48 Evoformer blocks for proteins',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Base data directory containing incomplete_blocks folder'
    )

    parser.add_argument(
        '--target-blocks',
        type=int,
        default=48,
        help='Number of blocks to generate (default: 48)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for computation (e.g., cuda:0, cpu)'
    )

    parser.add_argument(
        '--max-proteins',
        type=int,
        default=None,
        help='Maximum number of proteins to process (default: all)'
    )

    parser.add_argument(
        '--protein-id',
        type=str,
        default=None,
        help='Process only a specific protein ID'
    )

    args = parser.parse_args()

    # Set up paths based on data_dir
    data_dir = Path(args.data_dir)
    incomplete_dir = data_dir / "incomplete_blocks"
    complete_dir = data_dir / "complete_blocks"

    # Create directories if they don't exist
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    complete_dir.mkdir(parents=True, exist_ok=True)

    # Find dependencies
    python_path = find_python()
    evoformer_script = find_evoformer_script()
    project_root = Path(__file__).parent.parent

    print("🚀 48-Block Evoformer Generator")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Source: {incomplete_dir}")
    print(f"📁 Target: {complete_dir}")
    print(f"🎯 Target blocks: {args.target_blocks}")
    print(f"💻 Device: {args.device}")
    print(f"🐍 Python: {python_path}")
    print(f"📜 Evoformer script: {evoformer_script}")
    print("=" * 50)

    # Check if incomplete_dir exists and has content
    if not incomplete_dir.exists():
        print(f"❌ Incomplete blocks directory not found: {incomplete_dir}")
        print("   Please run generate_evoformer_inputs.py first")
        return

    # Get all protein directories
    if args.protein_id:
        # Process specific protein
        protein_dirs = [d for d in incomplete_dir.iterdir()
                        if d.is_dir() and args.protein_id in d.name]
        if not protein_dirs:
            print(f"❌ Protein {args.protein_id} not found in {incomplete_dir}")
            return
    else:
        # Process all proteins
        protein_dirs = [d for d in incomplete_dir.iterdir() if d.is_dir() and d.name.endswith('_evoformer_blocks')]

    total_proteins = len(protein_dirs)
    if total_proteins == 0:
        print("❌ No protein directories found in incomplete_blocks/")
        print("   Please run generate_evoformer_inputs.py first")
        return

    print(f"📊 Found {total_proteins} proteins to process")

    # Limit number of proteins if requested
    if args.max_proteins:
        protein_dirs = protein_dirs[:args.max_proteins]
        print(f"🔢 Processing only first {args.max_proteins} proteins")

    # Process each protein
    successful = 0
    failed = 0

    for i, protein_dir in enumerate(protein_dirs, 1):
        print(f"\n[{i}/{len(protein_dirs)}] Processing {protein_dir.name}")

        if generate_blocks_for_protein(protein_dir, args.target_blocks,
                                       evoformer_script, python_path, project_root, args.device):
            if move_protein_to_complete(protein_dir, complete_dir):
                successful += 1
            else:
                print(f"  ⚠️  Generated but couldn't move {protein_dir.name}")
                failed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Completed proteins in: {complete_dir}")


if __name__ == "__main__":
    main()