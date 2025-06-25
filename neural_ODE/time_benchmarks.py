#!/usr/bin/env python3
"""
Benchmark script to time run_evoformer_48th_iter.py and test_model.py
on all test proteins using 20250616_180845_full_ode_with_prelim_final_model.pt
Neural ODE and openfold are swapped
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict
import json


def load_test_proteins(splits_dir: str) -> List[str]:
    """Load test protein IDs from testing_chains.txt"""
    test_file = Path(splits_dir) / "testing_chains.txt"
    with open(test_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def find_protein_data(protein_id: str, data_dirs: List[str]) -> str:
    """Find which data directory contains the protein"""
    for data_dir in data_dirs:
        protein_dir = Path(data_dir) / f"{protein_id}_evoformer_blocks" / "recycle_0"
        if protein_dir.exists():
            return data_dir
    return None


def get_protein_residue_count(protein_id: str, data_dir: str) -> int:
    """Get number of residues for a protein"""
    protein_dir = Path(data_dir) / f"{protein_id}_evoformer_blocks" / "recycle_0"
    m_path = protein_dir / "m_block_0.pt"

    try:
        import torch
        m = torch.load(m_path, map_location='cpu')
        if m.dim() == 4:
            m = m.squeeze(0)
        return m.shape[-2]
    except:
        return -1


def benchmark_evoformer_48th(protein_id: str, data_dir: str, script_dir: Path) -> Dict:
    """Benchmark run_evoformer_48th_iter.py for a protein"""
    protein_dir = Path(data_dir) / f"{protein_id}_evoformer_blocks" / "recycle_0"
    m_path = protein_dir / "m_block_0.pt"
    z_path = protein_dir / "z_block_0.pt"

    # Get residue count
    num_residues = get_protein_residue_count(protein_id, data_dir)

    if not (m_path.exists() and z_path.exists()):
        return {"success": False, "neural ODE time": 0, "num_residues": num_residues}

    output_dir = script_dir / "temp_evoformer_output" / protein_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_dir.parent / "save_intermediates" / "run_evoformer_48th_iter.py"),
        "--m_path", str(m_path),
        "--z_path", str(z_path),
        "--output_dir", str(output_dir),
        "--device", "cuda:0"
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
    end_time = time.time()

    # Clean up output
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    return {
        "success": result.returncode == 0,
        "neural ODE time": end_time - start_time,
        "num_residues": num_residues
    }


def benchmark_test_model_individual(protein_id: str, data_dir: str, model_path: str,
                                    splits_dir: str, script_dir: Path) -> Dict:
    """Benchmark test_model.py for a single protein"""
    cmd = [
        sys.executable,
        str(script_dir / "test_model.py"),
        "--model_path", model_path,
        "--data_dirs", data_dir,
        "--splits_dir", splits_dir,
        "--device", "cuda",
        "--protein_list", protein_id
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
    end_time = time.time()

    return {
        "success": result.returncode == 0,
        "openfold time": end_time - start_time
    }


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    data_dirs = [
        "/media/visitor/Extreme SSD/data/complete_blocks",
        "/media/visitor/Extreme SSD/data/endpoint_blocks",
    ]
    splits_dir = script_dir / "data_splits" / "jumbo"
    model_name = "20250616_180845_full_ode_with_prelim_final_model.pt"
    model_path = script_dir / "trained_models" / model_name

    print("🚀 BENCHMARKING EVOFORMER SCRIPTS")
    print("=" * 50)
    print(f"📁 Model: {model_name}")
    print(f"📂 Data dirs: {len(data_dirs)} directories")
    print(f"📊 Splits: {splits_dir}")

    # Validate setup
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        sys.exit(1)

    # Load test proteins
    test_proteins = load_test_proteins(str(splits_dir))
    print(f"📋 Found {len(test_proteins)} test proteins")

    # Find available proteins in data directories
    available_proteins = []
    for protein_id in test_proteins:
        data_dir = find_protein_data(protein_id, data_dirs)
        if data_dir:
            available_proteins.append(protein_id)

    print(f"✅ Available proteins: {len(available_proteins)}/{len(test_proteins)}")

    if not available_proteins:
        print("❌ No proteins found in data directories!")
        sys.exit(1)

    results = {
        "model": model_name,
        "total_proteins": len(available_proteins),
        "proteins": {}
    }

    # Benchmark both methods for each protein
    print(f"\n🔬 Benchmarking both methods for each protein...")

    for i, protein_id in enumerate(available_proteins):
        print(f"   [{i + 1}/{len(available_proteins)}] {protein_id}... ", end="", flush=True)

        data_dir = find_protein_data(protein_id, data_dirs)

        # Initialize protein result structure
        protein_result = {
            "success": False,
            "openfold time": 0,
            "neural ODE time": 0,
            "num_residues": get_protein_residue_count(protein_id, data_dir)
        }

        # Run neural ODE benchmark
        neural_ode_result = benchmark_evoformer_48th(protein_id, data_dir, script_dir)
        protein_result["neural ODE time"] = neural_ode_result["neural ODE time"]
        neural_ode_success = neural_ode_result["success"]

        # Run openfold benchmark
        openfold_result = benchmark_test_model_individual(
            protein_id, data_dir, str(model_path), str(splits_dir), script_dir
        )
        protein_result["openfold time"] = openfold_result["openfold time"]
        openfold_success = openfold_result["success"]

        # Overall success if both methods succeed
        protein_result["success"] = neural_ode_success and openfold_success

        results["proteins"][protein_id] = protein_result

        # Print status
        if protein_result["success"]:
            print(
                f"✅ Neural: {protein_result['neural ODE time']:.2f}s, Openfold: {protein_result['openfold time']:.2f}s ({protein_result['num_residues']} res)")
        else:
            failed_methods = []
            if not neural_ode_success:
                failed_methods.append("Neural")
            if not openfold_success:
                failed_methods.append("Openfold")
            print(f"❌ Failed: {', '.join(failed_methods)} ({protein_result['num_residues']} res)")

    # Calculate summary statistics
    successful_proteins = [p for p in results["proteins"].values() if p["success"]]
    neural_times = [p["neural ODE time"] for p in successful_proteins]
    openfold_times = [p["openfold time"] for p in successful_proteins]

    print(f"\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    print(f"🔬 Neural ODE Evoformer:")
    print(f"   Successful: {len(successful_proteins)}/{len(available_proteins)}")
    if neural_times:
        print(f"   Total time: {sum(neural_times):.2f}s")
        print(f"   Average per protein: {sum(neural_times) / len(neural_times):.2f}s")
        print(f"   Range: [{min(neural_times):.2f}s, {max(neural_times):.2f}s]")

    print(f"\n🧪 Openfold Evoformer:")
    print(f"   Successful: {len(successful_proteins)}/{len(available_proteins)}")
    if openfold_times:
        print(f"   Total time: {sum(openfold_times):.2f}s")
        print(f"   Average per protein: {sum(openfold_times) / len(openfold_times):.2f}s")
        print(f"   Range: [{min(openfold_times):.2f}s, {max(openfold_times):.2f}s]")

    # Save detailed results
    results_file = script_dir / f"benchmark_results_{model_name.replace('.pt', '')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_file.name}")
    print(f"🎯 Benchmark complete!")


if __name__ == "__main__":
    main()