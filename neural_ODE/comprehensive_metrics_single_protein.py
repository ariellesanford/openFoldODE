#!/usr/bin/env python3
"""
Single protein structure comparison using tmtools
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

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


def compute_metrics(pred_coords: np.ndarray, ref_coords: np.ndarray) -> Dict:
    """Compute TM-score, RMSD, GDT-TS and other metrics using tmtools"""
    pred_seq = 'A' * len(pred_coords)
    ref_seq = 'A' * len(ref_coords)
    result = tm_align(pred_coords, ref_coords, pred_seq, ref_seq)
    
    # Apply alignment to get per-residue distances
    aligned_pred = pred_coords @ result.u + result.t
    distances = np.sqrt(np.sum((aligned_pred - ref_coords) ** 2, axis=1))
    
    return {
        "tm_score": float(result.tm_norm_chain2),
        "rmsd": float(result.rmsd),
        "gdt_ts": float(np.mean([np.mean(distances <= t) for t in [1, 2, 4, 8]])),
        "gdt_ha": float(np.mean([np.mean(distances <= t) for t in [0.5, 1, 2, 4]])),
        "frac_within_2A": float(np.mean(distances <= 2.0)),
        "frac_within_4A": float(np.mean(distances <= 4.0)),
        "max_deviation": float(np.max(distances)),
        "median_distance": float(np.median(distances)),
    }


def analyze_protein(pdb_id: str, base_path: Path):
    """Analyze a single protein against references"""
    if not TMTOOLS_AVAILABLE:
        print("❌ tmtools not available")
        return

    # Define methods and their paths
    methods = {
        "OpenFold Deconstructed": base_path / pdb_id / "openfold_deconstructed",
        "OpenFold 0recycles": base_path / pdb_id / "openfold_0recycles" / "predictions",
        "Half Evoformer": base_path / pdb_id / "half_evoformer",
        "NeuralODE_20250616_180845": base_path / pdb_id / "neuralODE" / "predictions_20250616_180845_full_ode_with_prelim",
        "NeuralODE_20250618_174724": base_path / pdb_id / "neuralODE" / "predictions_20250618_174724_full_ode_with_prelim2",
    }

    references = {
        "openfold_deconstructed": base_path / pdb_id / "openfold_deconstructed",
        "openfold_0recycles": base_path / pdb_id / "openfold_0recycles" / "predictions",
    }

    # Load method coordinates
    method_coords = {}
    for name, dir_path in methods.items():
        for suffix in ["_relaxed.pdb", "_unrelaxed.pdb"]:
            pdb_path = dir_path / f"{pdb_id}_model_1_ptm{suffix}"
            if pdb_path.exists():
                coords = parse_pdb_ca_coords(pdb_path)
                if coords is not None:
                    method_coords[name] = coords
                break

    if not method_coords:
        print(f"❌ No structures found for {pdb_id}")
        return

    # Compare against each reference
    for ref_name, ref_dir in references.items():
        ref_path = None
        for suffix in ["_relaxed.pdb", "_unrelaxed.pdb"]:
            p = ref_dir / f"{pdb_id}_model_1_ptm{suffix}"
            if p.exists():
                ref_path = p
                break

        if not ref_path:
            print(f"Reference not found: {ref_name}")
            continue

        ref_coords = parse_pdb_ca_coords(ref_path)
        if ref_coords is None:
            continue

        print(f"\n{'='*80}")
        print(f"TM-scores vs {ref_name}")
        print(f"{'='*80}")
        print(f"Reference CA atoms: {len(ref_coords)}")
        print(f"\n{'Method':<30} {'TM-score':<12} {'GDT-TS':<12} {'RMSD':<12}")
        print(f"{'-'*70}")

        for method_name, pred_coords in method_coords.items():
            metrics = compute_metrics(pred_coords, ref_coords)
            tm = metrics['tm_score']
            gdt_ts = metrics['gdt_ts']
            rmsd = metrics['rmsd']
            
            # Add quality indicator
            if tm >= 0.5:
                quality = "✓ same fold"
            elif tm >= 0.3:
                quality = "~ partial"
            else:
                quality = "✗ different"
            
            print(f"{method_name:<30} {tm:<12.4f} {gdt_ts:<12.4f} {rmsd:<12.2f} {quality}")

    print(f"\n{'='*80}")
    print("Metric interpretation:")
    print("  TM-score: ≥0.50 = same fold | 0.30-0.50 = partial | <0.30 = different")
    print("  GDT-TS:   avg % residues within 1/2/4/8 Å (rewards partial correctness)")
    print("  RMSD:     root mean square deviation after alignment (lower = better)")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    
    base_path = Path('/Volumes/Extreme SSD/data/structure_predictions')
    
    if len(sys.argv) > 1:
        pdb_id = sys.argv[1]
    else:
        pdb_id = "1o70_A"  # default
    
    print(f"Analyzing: {pdb_id}")
    analyze_protein(pdb_id, base_path)