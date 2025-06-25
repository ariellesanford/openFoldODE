#!/usr/bin/env python3
"""
Generate 48 evoformer blocks for all proteins and move to complete_blocks when done
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import time


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


def run_single_iteration(protein_dir, current_index, evoformer_script, python_path, project_root):
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
        "--device", "cuda:0"
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
    # Paths
    incomplete_dir = Path("/media/visitor/Extreme SSD/data/incomplete_blocks")
    complete_dir = Path("/media/visitor/Extreme SSD/data/complete_blocks")
    target_blocks = 48

    # Find dependencies
    python_path = find_python()
    evoformer_script = find_evoformer_script()
    project_root = Path(__file__).parent.parent

    print("🚀 48-Block Evoformer Generator")
    print(f"📁 Source: {incomplete_dir}")
    print(f"📁 Target: {complete_dir}")
    print(f"🎯 Target blocks: {target_blocks}")
    print(f"📜 Script: {evoformer_script}")
    print("")

    # Create complete directory if it doesn't exist
    complete_dir.mkdir(exist_ok=True)

    # Find all protein directories
    protein_dirs = []
    for item in incomplete_dir.iterdir():
        if item.is_dir() and item.name.endswith('_evoformer_blocks'):
            protein_dirs.append(item)

    if not protein_dirs:
        print(f"❌ No protein directories found in {incomplete_dir}")
        return

    print(f"🧬 Found {len(protein_dirs)} proteins to process")

    # Process each protein
    completed_proteins = 0
    failed_proteins = 0
    start_time = time.time()

    for i, protein_dir in enumerate(protein_dirs, 1):
        protein_name = protein_dir.name.replace('_evoformer_blocks', '')

        print(f"\n[{i}/{len(protein_dirs)}] Processing {protein_name}")
        print("=" * 60)

        # Generate blocks
        if generate_blocks_for_protein(protein_dir, target_blocks, evoformer_script, python_path, project_root):
            # Move to complete directory
            print(f"📦 Moving {protein_name} to complete_blocks...")
            if move_protein_to_complete(protein_dir, complete_dir):
                completed_proteins += 1
                print(f"✅ {protein_name}: COMPLETED and moved")
            else:
                failed_proteins += 1
                print(f"❌ {protein_name}: Generated but move failed")
        else:
            failed_proteins += 1
            print(f"❌ {protein_name}: Block generation failed")

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Completed and moved: {completed_proteins}")
    print(f"❌ Failed: {failed_proteins}")
    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")

    if completed_proteins > 0:
        avg_time = total_time / completed_proteins
        print(f"📈 Average time per protein: {avg_time / 60:.1f} minutes")

    print(f"\n📁 Results:")
    print(f"  Complete: {complete_dir}")
    print(f"  Remaining: {incomplete_dir}")


if __name__ == "__main__":
    main()