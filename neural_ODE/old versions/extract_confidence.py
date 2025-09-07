#!/usr/bin/env python3
"""
Extract pTM and pLDDT confidence scores from OpenFold pickle output files
Usage: python extract_confidence_scores.py <path_to_output_dict.pkl>
"""

import pickle
import sys
import numpy as np
from pathlib import Path


def extract_scores(pkl_path):
    """Extract pTM and pLDDT scores from OpenFold output pickle file"""

    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        output_dict = pickle.load(f)

    # Extract scores
    ptm_score = None
    plddt_scores = None
    mean_plddt = None

    # Get pTM score (single value for whole structure) - try multiple possible keys
    ptm_keys = ['ptm', 'predicted_tm_score', 'predicted_tm', 'tm_score', 'ranking_confidence']
    for key in ptm_keys:
        if key in output_dict:
            ptm_score = float(output_dict[key])
            break

    # Get pLDDT scores (per-residue confidence) - try multiple possible keys
    plddt_keys = ['plddt', 'predicted_lddt', 'lddt', 'confidence', 'per_residue_confidence']
    for key in plddt_keys:
        if key in output_dict:
            plddt_scores = output_dict[key]
            if isinstance(plddt_scores, np.ndarray):
                plddt_scores = plddt_scores.flatten()
                mean_plddt = float(np.mean(plddt_scores))
            break

    return ptm_score, plddt_scores, mean_plddt


def print_scores(pkl_path, ptm_score, plddt_scores, mean_plddt):
    """Print the extracted scores in a nice format"""

    print(f"📊 Confidence Scores for: {Path(pkl_path).name}")
    print("=" * 50)

    # pTM score
    if ptm_score is not None:
        print(f"🎯 pTM Score: {ptm_score:.3f}")
        if ptm_score >= 0.8:
            print("   → Very High Confidence")
        elif ptm_score >= 0.7:
            print("   → High Confidence")
        elif ptm_score >= 0.6:
            print("   → Medium Confidence")
        else:
            print("   → Low Confidence")
    else:
        print("❌ pTM Score: Not found")

    print()

    # pLDDT scores
    if plddt_scores is not None and mean_plddt is not None:
        print(f"📈 pLDDT Scores:")
        print(f"   Mean pLDDT: {mean_plddt:.2f}")
        print(f"   Min pLDDT:  {np.min(plddt_scores):.2f}")
        print(f"   Max pLDDT:  {np.max(plddt_scores):.2f}")
        print(f"   Residues:   {len(plddt_scores)}")

        # Confidence breakdown
        very_high = np.sum(plddt_scores >= 90)
        high = np.sum((plddt_scores >= 70) & (plddt_scores < 90))
        medium = np.sum((plddt_scores >= 50) & (plddt_scores < 70))
        low = np.sum(plddt_scores < 50)

        print(f"\n   Confidence Breakdown:")
        print(f"   Very High (≥90): {very_high:3d} residues ({very_high / len(plddt_scores) * 100:.1f}%)")
        print(f"   High (70-89):    {high:3d} residues ({high / len(plddt_scores) * 100:.1f}%)")
        print(f"   Medium (50-69):  {medium:3d} residues ({medium / len(plddt_scores) * 100:.1f}%)")
        print(f"   Low (<50):       {low:3d} residues ({low / len(plddt_scores) * 100:.1f}%)")

        # Overall assessment
        if mean_plddt >= 80:
            assessment = "Very High Confidence"
        elif mean_plddt >= 70:
            assessment = "High Confidence"
        elif mean_plddt >= 60:
            assessment = "Medium Confidence"
        else:
            assessment = "Low Confidence"

        print(f"\n   → Overall: {assessment}")

    else:
        print("❌ pLDDT Scores: Not found")


def main():
    # =====================================================
    # CONFIGURATION - SET YOUR PKL FILE PATH HERE
    # =====================================================
    pkl_path = "/media/visitor/Extreme SSD/data/structure_predictions/1fv5_A/openfold_0recycles/predictions/1fv5_A_model_1_ptm_output_dict.pkl"

    # =====================================================
    # You can also uncomment this section to use command line arguments instead:
    # =====================================================
    # if len(sys.argv) == 2:
    #     pkl_path = sys.argv[1]  # Override with command line argument if provided

    # Check if file exists
    if not Path(pkl_path).exists():
        print(f"❌ Error: File not found: {pkl_path}")
        sys.exit(1)

    # Check if file is a pickle file
    if not pkl_path.endswith('.pkl'):
        print(f"⚠️  Warning: File doesn't end with .pkl: {pkl_path}")

    try:
        # Extract scores
        ptm_score, plddt_scores, mean_plddt = extract_scores(pkl_path)

        # Print results
        print_scores(pkl_path, ptm_score, plddt_scores, mean_plddt)

        # Also show available keys for debugging
        with open(pkl_path, 'rb') as f:
            output_dict = pickle.load(f)

        print(f"\n🔍 Available keys in pickle file:")
        for key in sorted(output_dict.keys()):
            value = output_dict[key]
            if isinstance(value, np.ndarray):
                print(f"   {key}: {value.shape} {value.dtype}")
            else:
                print(f"   {key}: {type(value).__name__}")

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
