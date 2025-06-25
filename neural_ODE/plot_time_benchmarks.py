#!/usr/bin/env python3
"""
Plot residues vs time for both evoformer_48th and test_model methods
from benchmark results JSON
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_benchmark_data(json_file: str):
    """Load and parse benchmark results"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract data from new structure
    neural_ode_residues = []
    neural_ode_times = []
    openfold_residues = []
    openfold_times = []

    for protein_id, result in data["proteins"].items():
        if result["success"] and result["num_residues"] > 0:
            # Neural ODE data
            neural_ode_residues.append(result["num_residues"])
            neural_ode_times.append(result["neural ODE time"])

            # Openfold data
            openfold_residues.append(result["num_residues"])
            openfold_times.append(result["openfold time"])

    return {
        "neural_ode": {"residues": neural_ode_residues, "times": neural_ode_times},
        "openfold": {"residues": openfold_residues, "times": openfold_times},
        "model_name": data.get("model", "Unknown Model")
    }


def create_plot(data, output_file=None):
    """Create residues vs time scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot neural ODE data
    neural_residues = data["neural_ode"]["residues"]
    neural_times = data["neural_ode"]["times"]

    if neural_residues:
        ax.scatter(neural_residues, neural_times,
                   alpha=0.7, s=50, color='blue',
                   label=f'Openfold Evoformer (n={len(neural_residues)})')

    # Plot openfold data
    openfold_residues = data["openfold"]["residues"]
    openfold_times = data["openfold"]["times"]

    if openfold_residues:
        ax.scatter(openfold_residues, openfold_times,
                   alpha=0.7, s=50, color='red', marker='s',
                   label=f'Neural ODE Evoformer (n={len(openfold_residues)})')

    # Formatting
    ax.set_xlabel('Number of Residues', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title(f'Execution Time vs Protein Size\nModel: {data["model_name"]}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_file}")

    plt.show()


def print_summary(data):
    """Print summary statistics"""
    print("\n📊 BENCHMARK ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Model: {data['model_name']}")

    neural_data = data["neural_ode"]
    openfold_data = data["openfold"]

    if neural_data["residues"]:
        print(f"\n🔬 NeuralODE Evoformer:")
        print(f"   Proteins tested: {len(neural_data['residues'])}")
        print(f"   Avg time: {np.mean(neural_data['times']):.2f}s")
        print(f"   Time range: [{min(neural_data['times']):.2f}s, {max(neural_data['times']):.2f}s]")
        print(f"   Residue range: [{min(neural_data['residues'])}, {max(neural_data['residues'])}]")

        # Time per residue analysis
        time_per_residue = [t / r for t, r in zip(neural_data['times'], neural_data['residues'])]
        print(f"   Avg time per residue: {np.mean(time_per_residue):.4f}s/residue")

    if openfold_data["residues"]:
        print(f"\n🧪 OpenfoldEvoformer:")
        print(f"   Proteins tested: {len(openfold_data['residues'])}")
        print(f"   Avg time: {np.mean(openfold_data['times']):.2f}s")
        print(f"   Time range: [{min(openfold_data['times']):.2f}s, {max(openfold_data['times']):.2f}s]")
        print(f"   Residue range: [{min(openfold_data['residues'])}, {max(openfold_data['residues'])}]")

        # Time per residue analysis
        time_per_residue = [t / r for t, r in zip(openfold_data['times'], openfold_data['residues'])]
        print(f"   Avg time per residue: {np.mean(time_per_residue):.4f}s/residue")


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results: residues vs time')
    parser.add_argument('json_file', nargs='?', help='Benchmark results JSON file')
    parser.add_argument('--output', '-o', help='Output plot file (PNG/PDF)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t show plot interactively')

    args = parser.parse_args()

    # If no JSON file provided (e.g., running in PyCharm), look for one automatically
    if not args.json_file:
        # Look for benchmark JSON files in current directory
        json_files = list(Path('.').glob('benchmark_results_*.json'))
        if json_files:
            args.json_file = str(json_files[0])  # Use first found
            print(f"🔍 Auto-detected JSON file: {args.json_file}")
        else:
            print("❌ No JSON file provided and none found matching 'benchmark_results_*.json'")
            print("Usage: python plot_benchmark_results.py <json_file>")
            return 1

    # Validate input file
    if not Path(args.json_file).exists():
        print(f"❌ File not found: {args.json_file}")
        return 1

    # Load and analyze data
    print(f"📁 Loading benchmark data from: {args.json_file}")
    data = load_benchmark_data(args.json_file)

    # Print summary
    print_summary(data)

    # Generate output filename if not provided
    output_file = args.output
    if not output_file:
        json_path = Path(args.json_file)
        output_file = json_path.parent / f"{json_path.stem}_plot.png"

    # Create plot
    print(f"\n📈 Creating plot...")
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

    create_plot(data, output_file)

    print(f"✅ Analysis complete!")


if __name__ == "__main__":
    main()