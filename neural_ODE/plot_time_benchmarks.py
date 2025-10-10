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


def fit_and_print_equations(data):
    """Fit functions to the data and print equations"""
    equations = {}

    # Fit linear for Neural ODE
    x1 = np.array(data["neural_ode"]["residues"])
    y1 = np.array(data["neural_ode"]["times"])
    if len(x1) >= 2:
        linear_coeffs = np.polyfit(x1, y1, 1)
        equations["neural_ode"] = linear_coeffs
        print(f"\n📈 Linear Fit (NeuralODE): time = {linear_coeffs[0]:.6f} * residues + {linear_coeffs[1]:.2f}")

    # Fit quadratic for Openfold
    x2 = np.array(data["openfold"]["residues"])
    y2 = np.array(data["openfold"]["times"])
    if len(x2) >= 3:
        quad_coeffs = np.polyfit(x2, y2, 2)
        equations["openfold"] = quad_coeffs
        print(f"📉 Quadratic Fit (Openfold): time = {quad_coeffs[0]:.6f} * residues^2 + {quad_coeffs[1]:.4f} * residues + {quad_coeffs[2]:.2f}")

    return equations


def create_plot(data, equations, output_file=None):
    """Create residues vs time scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Neural ODE
    neural_residues = np.array(data["neural_ode"]["residues"])
    neural_times = np.array(data["neural_ode"]["times"])
    if len(neural_residues) > 0:
        ax.scatter(neural_residues, neural_times,
                   alpha=0.7, s=50, color='blue',
                   label=f'Neural ODE Evoformer (n={len(neural_residues)})')
        if "neural_ode" in equations:
            a, b = equations["neural_ode"]
            xfit = np.linspace(min(neural_residues), max(neural_residues), 100)
            yfit = a * xfit + b
            ax.plot(xfit, yfit, color='blue', linestyle='--', label='Linear Fit (Neural ODE)')

    # Openfold
    openfold_residues = np.array(data["openfold"]["residues"])
    openfold_times = np.array(data["openfold"]["times"])
    if len(openfold_residues) > 0:
        ax.scatter(openfold_residues, openfold_times,
                   alpha=0.7, s=50, color='red', marker='s',
                   label=f'Openfold Evoformer (n={len(openfold_residues)})')
        if "openfold" in equations:
            a, b, c = equations["openfold"]
            xfit = np.linspace(min(openfold_residues), max(openfold_residues), 100)
            yfit = a * xfit ** 2 + b * xfit + c
            ax.plot(xfit, yfit, color='red', linestyle='--', label='Quadratic Fit (Openfold)')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Number of Residues', fontsize=18)
    ax.set_ylabel('Execution Time (seconds)', fontsize=18)
    ax.set_title(f'Execution Time vs Protein Size', fontsize=20)
    ax.legend(fontsize=18)
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

    for key, label in [("neural_ode", "🔬 NeuralODE Evoformer"), ("openfold", "🧪 Openfold Evoformer")]:
        d = data[key]
        if d["residues"]:
            print(f"\n{label}:")
            print(f"   Proteins tested: {len(d['residues'])}")
            print(f"   Avg time: {np.mean(d['times']):.2f}s")
            print(f"   Time range: [{min(d['times']):.2f}s, {max(d['times']):.2f}s]")
            print(f"   Residue range: [{min(d['residues'])}, {max(d['residues'])}]")
            time_per_residue = [t / r for t, r in zip(d['times'], d['residues'])]
            print(f"   Avg time per residue: {np.mean(time_per_residue):.4f}s/residue")


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results: residues vs time')
    parser.add_argument('json_file', nargs='?', help='Benchmark results JSON file')
    parser.add_argument('--output', '-o', help='Output plot file (PNG/PDF)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t show plot interactively')

    args = parser.parse_args()

    if not args.json_file:
        json_files = list(Path('.').glob('benchmark_results_*.json'))
        if json_files:
            args.json_file = str(json_files[0])
            print(f"🔍 Auto-detected JSON file: {args.json_file}")
        else:
            print("❌ No JSON file provided and none found matching 'benchmark_results_*.json'")
            print("Usage: python plot_benchmark_results.py <json_file>")
            return 1

    if not Path(args.json_file).exists():
        print(f"❌ File not found: {args.json_file}")
        return 1

    print(f"📁 Loading benchmark data from: {args.json_file}")
    data = load_benchmark_data(args.json_file)

    print_summary(data)

    print(f"\n🔧 Fitting trend lines...")
    equations = fit_and_print_equations(data)

    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')

    output_file = args.output or str(Path(args.json_file).with_name(Path(args.json_file).stem + "_plot.pdf"))

    create_plot(data, equations, output_file)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
