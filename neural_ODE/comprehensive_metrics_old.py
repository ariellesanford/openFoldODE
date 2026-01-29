#!/usr/bin/env python3
"""
Comprehensive metrics collection for protein structure prediction
Processes all proteins and generates summary statistics + scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List
from collections import defaultdict

try:
    from tmtools import tm_align
    TMTOOLS_AVAILABLE = True
except ImportError:
    TMTOOLS_AVAILABLE = False
    print("⚠️  tmtools not installed. Run: pip install tmtools")


def parse_pdb_ca_coords(pdb_path: Path) -> Optional[np.ndarray]:
    """Parse CA coordinates from PDB file"""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[13:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else None


def compute_tm_metrics(pred_coords: np.ndarray, ref_coords: np.ndarray) -> Dict:
    """Compute TM-score and RMSD using tmtools"""
    pred_seq = 'A' * len(pred_coords)
    ref_seq = 'A' * len(ref_coords)
    result = tm_align(pred_coords, ref_coords, pred_seq, ref_seq)
    return {
        "tm_score": float(result.tm_norm_chain2),
        "rmsd": float(result.rmsd),
    }


def discover_proteins(base_path: Path) -> List[str]:
    """Find all protein IDs in structure_predictions directory"""
    proteins = []
    for d in base_path.iterdir():
        if d.is_dir() and (d / "openfold_deconstructed").exists():
            proteins.append(d.name)
    return sorted(proteins)


def get_pdb_path(base_path: Path, pdb_id: str, method: str) -> Optional[Path]:
    """Get PDB path for a method, checking relaxed then unrelaxed"""
    if method == "openfold_deconstructed":
        dir_path = base_path / pdb_id / "openfold_deconstructed"
    elif method == "openfold_0recycles":
        dir_path = base_path / pdb_id / "openfold_0recycles" / "predictions"
    elif method == "half_evoformer":
        dir_path = base_path / pdb_id / "half_evoformer"
    elif method.startswith("neural_ode:"):
        pred_name = method.split(":")[1]
        dir_path = base_path / pdb_id / "neuralODE" / pred_name
    else:
        return None
    
    for suffix in ["_relaxed.pdb", "_unrelaxed.pdb"]:
        p = dir_path / f"{pdb_id}_model_1_ptm{suffix}"
        if p.exists():
            return p
    return None


def process_all_proteins(base_path: Path):
    """Process all proteins and collect metrics"""
    if not TMTOOLS_AVAILABLE:
        print("❌ tmtools not available")
        return

    proteins = discover_proteins(base_path)
    print(f"Found {len(proteins)} proteins")

    # Results storage: {method: {reference: [(pdb_id, length, tm_score, rmsd), ...]}}
    results = defaultdict(lambda: defaultdict(list))
    
    references = ["openfold_deconstructed", "openfold_0recycles"]
    
    # Only these two Neural ODE runs
    neural_ode_preds = [
        "predictions_20250616_180845_full_ode_with_prelim",
        "predictions_20250618_174724_full_ode_with_prelim2",
    ]
    
    for pdb_id in proteins:
        # Check if both Neural ODE predictions exist for this protein
        neural_ode_paths = [get_pdb_path(base_path, pdb_id, f"neural_ode:{pred}") for pred in neural_ode_preds]
        if not all(neural_ode_paths):
            continue

        # Get reference coordinates
        ref_coords = {}
        for ref_name in references:
            ref_path = get_pdb_path(base_path, pdb_id, ref_name)
            if ref_path:
                coords = parse_pdb_ca_coords(ref_path)
                if coords is not None:
                    ref_coords[ref_name] = coords

        if not ref_coords:
            continue

        protein_length = len(list(ref_coords.values())[0])

        # Methods to evaluate
        methods = ["openfold_deconstructed", "openfold_0recycles", "half_evoformer"]
        methods += [f"neural_ode:{pred}" for pred in neural_ode_preds]

        for method in methods:
            pred_path = get_pdb_path(base_path, pdb_id, method)
            if not pred_path:
                continue
            
            pred_coords = parse_pdb_ca_coords(pred_path)
            if pred_coords is None:
                continue

            for ref_name, ref_c in ref_coords.items():
                metrics = compute_tm_metrics(pred_coords, ref_c)
                # Simplify method name for display
                display_name = (method
                    .replace("neural_ode:predictions_", "NeuralODE_")
                    .replace("_full_ode_with_prelim2", "_v2")
                    .replace("_full_ode_with_prelim", "")
                )
                results[display_name][ref_name].append((pdb_id, protein_length, metrics["tm_score"], metrics["rmsd"]))

    return results


def print_summary_stats(results):
    """Print average and STD for each method against each reference"""
    references = ["openfold_deconstructed", "openfold_0recycles"]
    
    for ref_name in references:
        print(f"\n{'='*80}")
        print(f"Summary Statistics vs {ref_name}")
        print(f"{'='*80}")
        print(f"\n{'Method':<35} {'N':<6} {'TM-score':<20} {'RMSD':<20}")
        print(f"{'-'*80}")
        
        for method in sorted(results.keys()):
            data = results[method].get(ref_name, [])
            if not data:
                continue
            
            tm_scores = [d[2] for d in data]
            rmsds = [d[3] for d in data]
            n = len(data)
            
            tm_mean, tm_std = np.mean(tm_scores), np.std(tm_scores)
            rmsd_mean, rmsd_std = np.mean(rmsds), np.std(rmsds)
            
            print(f"{method:<35} {n:<6} {tm_mean:.3f} ± {tm_std:.3f}        {rmsd_mean:.2f} ± {rmsd_std:.2f}")


def plot_scatter(results, output_dir: Path):
    """Generate scatter plots: TM-score and RMSD vs protein length"""
    references = ["openfold_deconstructed", "openfold_0recycles"]
    
    for ref_name in references:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Collect data for plotting
        for method in sorted(results.keys()):
            data = results[method].get(ref_name, [])
            if not data:
                continue
            
            lengths = [d[1] for d in data]
            tm_scores = [d[2] for d in data]
            rmsds = [d[3] for d in data]
            
            # Skip self-comparisons (TM=1.0)
            if method == ref_name:
                continue
            
            axes[0].scatter(lengths, tm_scores, label=method, alpha=0.7, s=30)
            axes[1].scatter(lengths, rmsds, label=method, alpha=0.7, s=30)
        
        axes[0].set_xlabel("Protein Length (residues)")
        axes[0].set_ylabel("TM-score")
        axes[0].set_title(f"TM-score vs Protein Length (ref: {ref_name})")
        axes[0].axhline(y=0.5, color='r', linestyle='--', label='Same fold threshold')
        axes[0].legend(fontsize=8, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Protein Length (residues)")
        axes[1].set_ylabel("RMSD (Å)")
        axes[1].set_title(f"RMSD vs Protein Length (ref: {ref_name})")
        axes[1].legend(fontsize=8, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f"scatter_vs_{ref_name}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved: {plot_path}")
        plt.close()


if __name__ == "__main__":
    # === CONFIGURATION ===
    base_path = Path('/Volumes/Extreme SSD/data/structure_predictions')
    output_dir = Path('/Volumes/Extreme SSD/data/structure_predictions')
    
    results = process_all_proteins(base_path)
    print_summary_stats(results)
    plot_scatter(results, output_dir)