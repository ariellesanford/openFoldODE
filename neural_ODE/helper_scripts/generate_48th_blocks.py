#!/usr/bin/env python3
"""
Generate 48th evoformer blocks for all proteins and move to endpoint_blocks when done
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
    """Find the evoformer 48th iteration script"""
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent

    possible_locations = [
        project_root / "save_intermediates" / "run_evoformer_48th_iter.py",
        script_dir / "run_evoformer_48th_iter.py"
    ]

    for script_path in possible_locations:
        if script_path.exists():
            return script_path

    raise FileNotFoundError("Could not find run_evoformer_48th_iter.py script")


def check_has_block_0(protein_dir):
    """Check if protein has valid block 0 files"""
    recycle_dir = protein_dir / "recycle_0"
    if not recycle_dir.exists():
        return False

    m_block_0 = recycle_dir / "m_block_0.pt"
    z_block_0 = recycle_dir / "z_block_0.pt"
    msa_mask = recycle_dir / "msa_mask.pt"
    pair_mask = recycle_dir / "pair_mask.pt"

    return all(f.exists() for f in [m_block_0, z_block_0, msa_mask, pair_mask])


def check_has_block_48(protein_dir):
    """Check if protein already has block 48 files"""
    recycle_dir = protein_dir / "recycle_0"
    if not recycle_dir.exists():
        return False

    m_block_48 = recycle_dir / "m_block_48.pt"
    z_block_48 = recycle_dir / "z_block_48.pt"

    return m_block_48.exists() and z_block_48.exists()


def run_48th_iteration(protein_dir, evoformer_script, python_path, project_root):
    """Run the 48th evoformer iteration for a protein"""
    recycle_dir = protein_dir / "recycle_0"

    m_path = recycle_dir / "m_block_0.pt"
    z_path = recycle_dir / "z_block_0.pt"

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
            timeout=1800,  # 30 minute timeout
            cwd=str(project_root),
            env=env
        )

        # Check if output files were created
        m_block_48 = recycle_dir / "m_block_48.pt"
        z_block_48 = recycle_dir / "z_block_48.pt"

        return result.returncode == 0 and m_block_48.exists() and z_block_48.exists()

    except subprocess.TimeoutExpired:
        print(f"    ⏰ Timeout after 30 minutes")
        return False
    except Exception as e:
        print(f"    💥 Error: {e}")
        return False


def generate_48th_block_for_protein(protein_dir, evoformer_script, python_path, project_root):
    """Generate 48th block for a single protein"""
    protein_name = protein_dir.name.replace('_evoformer_blocks', '')

    # Check if already has block 48
    if check_has_block_48(protein_dir):
        print(f"✅ {protein_name}: Already has 48th block")
        return True

    # Check if has block 0
    if not check_has_block_0(protein_dir):
        print(f"❌ {protein_name}: No valid block 0 found")
        return False

    print(f"🔄 {protein_name}: Generating 48th block...")

    if run_48th_iteration(protein_dir, evoformer_script, python_path, project_root):
        print(f"🎉 {protein_name}: 48th block generated!")
        return True
    else:
        print(f"💀 {protein_name}: Failed to generate 48th block")
        return False


def move_protein_to_endpoint(protein_dir, endpoint_dir):
    """Move completed protein to endpoint_blocks directory"""
    protein_name = protein_dir.name
    target_path = endpoint_dir / protein_name

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
    endpoint_dir = Path("/media/visitor/Extreme SSD/data/endpoint_blocks")

    # Find dependencies
    python_path = find_python()
    evoformer_script = find_evoformer_script()
    project_root = Path(__file__).parent.parent

    print("🚀 48th Block Generator")
    print(f"📁 Source: {incomplete_dir}")
    print(f"📁 Target: {endpoint_dir}")
    print(f"📜 Script: {evoformer_script}")
    print("")

    # Create endpoint directory if it doesn't exist
    endpoint_dir.mkdir(exist_ok=True)

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

        # Generate 48th block
        if generate_48th_block_for_protein(protein_dir, evoformer_script, python_path, project_root):
            # Move to endpoint directory
            print(f"📦 Moving {protein_name} to endpoint_blocks...")
            if move_protein_to_endpoint(protein_dir, endpoint_dir):
                completed_proteins += 1
                print(f"✅ {protein_name}: COMPLETED and moved")
            else:
                failed_proteins += 1
                print(f"❌ {protein_name}: Generated but move failed")
        else:
            failed_proteins += 1
            print(f"❌ {protein_name}: Generation failed")

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
    print(f"  Endpoint blocks: {endpoint_dir}")
    print(f"  Remaining incomplete: {incomplete_dir}")


if __name__ == "__main__":
    main()