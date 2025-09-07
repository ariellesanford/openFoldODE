#!/usr/bin/env python3
"""
Debug script to analyze the Neural ODE → Structure Module pipeline
This will identify where the structure prediction is failing
"""

import torch
import numpy as np
import os
from pathlib import Path
import json
from torchdiffeq import odeint
import torch.nn.functional as F


def load_trained_model(model_path: str, device: str = 'cuda'):
    """Load your trained Neural ODE model"""
    print(f"📦 Loading trained model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})

    # Determine model type from config
    use_fast_ode = config.get('use_fast_ode', True)
    hidden_dim = config.get('hidden_dim', 64)

    from neural_ODE.evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast

    if use_fast_ode:
        model = EvoformerODEFuncFast(256, 128, hidden_dim).to(device)
    else:
        model = EvoformerODEFunc(256, 128, hidden_dim).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded {'Fast' if use_fast_ode else 'Full'} ODE model with {hidden_dim} hidden dims")
    return model, config


def compare_representations(protein_id: str, model, device: str = 'cuda'):
    """Compare Neural ODE predictions vs actual OpenFold representations"""

    print(f"\n🔍 COMPARING REPRESENTATIONS FOR {protein_id}")
    print("=" * 60)

    # Load original data
    from neural_ODE.train_evoformer_ode import DataManager
    data_dir = "/media/visitor/Extreme SSD/data/complete_blocks"

    m0, z0, m48_actual, z48_actual = DataManager.load_protein_blocks(
        protein_id, data_dir, device, max_cluster_size=256, blocks=[0, 48]
    )

    print(f"📊 Original data shapes:")
    print(f"   Input: M{m0.shape}, Z{z0.shape}")
    print(f"   Target: M{m48_actual.shape}, Z{z48_actual.shape}")

    # Generate Neural ODE prediction
    print(f"\n🚀 Running Neural ODE prediction...")

    with torch.no_grad():
        trajectory = odeint(
            model, (m0, z0),
            torch.tensor([0.0, 1.0]).to(device),
            method='rk4', rtol=1e-4, atol=1e-5
        )

        m48_pred, z48_pred = trajectory[0][-1], trajectory[1][-1]

    print(f"   Predicted: M{m48_pred.shape}, Z{z48_pred.shape}")

    # Compute losses (using your actual loss function)
    def compute_your_loss(pred_m, target_m, pred_z, target_z):
        num_residues = target_m.shape[1] if target_m.dim() == 3 else target_m.shape[0]

        msa_loss = F.mse_loss(pred_m, target_m)
        pair_loss = F.mse_loss(pred_z, target_z)

        msa_variance = target_m.var() + 1e-8
        pair_variance = target_z.var() + 1e-8

        msa_scaled = msa_loss / msa_variance
        pair_scaled = pair_loss / pair_variance

        base_loss = msa_scaled + pair_scaled
        return base_loss * float(num_residues)

    loss = compute_your_loss(m48_pred, m48_actual, z48_pred, z48_actual)
    print(f"   Prediction loss: {loss.item():.5f}")

    # Analyze prediction quality
    print(f"\n📊 PREDICTION QUALITY ANALYSIS:")

    # 1. Value ranges
    print(f"   Value ranges:")
    print(f"     M_actual: [{m48_actual.min():.3f}, {m48_actual.max():.3f}]")
    print(f"     M_pred:   [{m48_pred.min():.3f}, {m48_pred.max():.3f}]")
    print(f"     Z_actual: [{z48_actual.min():.3f}, {z48_actual.max():.3f}]")
    print(f"     Z_pred:   [{z48_pred.min():.3f}, {z48_pred.max():.3f}]")

    # 2. Distribution comparison
    print(f"   Distributions:")
    print(f"     M_actual: mean={m48_actual.mean():.3f}, std={m48_actual.std():.3f}")
    print(f"     M_pred:   mean={m48_pred.mean():.3f}, std={m48_pred.std():.3f}")
    print(f"     Z_actual: mean={z48_actual.mean():.3f}, std={z48_actual.std():.3f}")
    print(f"     Z_pred:   mean={z48_pred.mean():.3f}, std={z48_pred.std():.3f}")

    # 3. Check for NaN/Inf
    nan_check = {
        'M_pred_nan': torch.isnan(m48_pred).any().item(),
        'Z_pred_nan': torch.isnan(z48_pred).any().item(),
        'M_pred_inf': torch.isinf(m48_pred).any().item(),
        'Z_pred_inf': torch.isinf(z48_pred).any().item(),
    }

    print(f"   NaN/Inf check: {nan_check}")

    # 4. Correlation analysis
    m_correlation = F.cosine_similarity(m48_pred.flatten().unsqueeze(0),
                                        m48_actual.flatten().unsqueeze(0)).item()
    z_correlation = F.cosine_similarity(z48_pred.flatten().unsqueeze(0),
                                        z48_actual.flatten().unsqueeze(0)).item()

    print(f"   Cosine similarities:")
    print(f"     MSA: {m_correlation:.4f}")
    print(f"     Pair: {z_correlation:.4f}")

    # 5. Check if prediction is reasonable
    issues = []

    if nan_check['M_pred_nan'] or nan_check['Z_pred_nan']:
        issues.append("CRITICAL: Predictions contain NaN values")

    if nan_check['M_pred_inf'] or nan_check['Z_pred_inf']:
        issues.append("CRITICAL: Predictions contain infinite values")

    if abs(m48_pred.mean() - m48_actual.mean()) > 10 * m48_actual.std():
        issues.append("WARNING: MSA prediction mean is way off")

    if abs(z48_pred.mean() - z48_actual.mean()) > 10 * z48_actual.std():
        issues.append("WARNING: Pair prediction mean is way off")

    if m_correlation < 0.1:
        issues.append("WARNING: MSA prediction has very low correlation")

    if z_correlation < 0.1:
        issues.append("WARNING: Pair prediction has very low correlation")

    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"     {issue}")
    else:
        print(f"\n✅ Predictions look numerically reasonable")

    return {
        'loss': loss.item(),
        'm_pred': m48_pred,
        'z_pred': z48_pred,
        'm_actual': m48_actual,
        'z_actual': z48_actual,
        'issues': issues,
        'correlations': {'msa': m_correlation, 'pair': z_correlation}
    }


def analyze_saved_predictions(protein_id: str, predictions_dir: str):
    """Analyze the saved prediction files for structure module"""

    print(f"\n📁 ANALYZING SAVED PREDICTIONS")
    print("=" * 40)

    pred_dir = Path(predictions_dir) / protein_id

    if not pred_dir.exists():
        print(f"❌ Predictions directory not found: {pred_dir}")
        return None

    # Check files
    msa_file = pred_dir / "msa_representation.pt"
    pair_file = pred_dir / "pair_representation.pt"
    metadata_file = pred_dir / "metadata.json"

    print(f"   Checking files in: {pred_dir}")
    print(f"     msa_representation.pt: {'✅' if msa_file.exists() else '❌'}")
    print(f"     pair_representation.pt: {'✅' if pair_file.exists() else '❌'}")
    print(f"     metadata.json: {'✅' if metadata_file.exists() else '❌'}")

    if not (msa_file.exists() and pair_file.exists()):
        print(f"❌ Missing prediction files")
        return None

    # Load and analyze saved predictions
    msa_saved = torch.load(msa_file, map_location='cpu')
    pair_saved = torch.load(pair_file, map_location='cpu')

    print(f"\n📊 Saved prediction analysis:")
    print(f"   MSA shape: {msa_saved.shape}")
    print(f"   Pair shape: {pair_saved.shape}")
    print(f"   MSA range: [{msa_saved.min():.3f}, {msa_saved.max():.3f}]")
    print(f"   Pair range: [{pair_saved.min():.3f}, {pair_saved.max():.3f}]")

    # Check if batch dimension was added correctly
    expected_msa_shape = (1, 256, None, 256)  # (batch, seq, res, channels)
    expected_pair_shape = (1, None, None, 128)  # (batch, res, res, channels)

    batch_ok = (msa_saved.dim() == 4 and msa_saved.shape[0] == 1 and
                pair_saved.dim() == 4 and pair_saved.shape[0] == 1)

    print(f"   Batch dimension correct: {'✅' if batch_ok else '❌'}")

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"   Metadata: {metadata}")

    return {
        'msa_shape': msa_saved.shape,
        'pair_shape': pair_saved.shape,
        'msa_range': (msa_saved.min().item(), msa_saved.max().item()),
        'pair_range': (pair_saved.min().item(), pair_saved.max().item()),
        'batch_ok': batch_ok
    }


def compare_with_openfold_reference(protein_id: str, predictions_dir: str):
    """Compare Neural ODE predictions with OpenFold reference"""

    print(f"\n⚖️  COMPARING WITH OPENFOLD REFERENCE")
    print("=" * 45)

    # Load Neural ODE predictions
    neural_ode_dir = Path(predictions_dir) / protein_id
    if not neural_ode_dir.exists():
        print(f"❌ Neural ODE predictions not found")
        return

    msa_neural_ode = torch.load(neural_ode_dir / "msa_representation.pt", map_location='cpu')
    pair_neural_ode = torch.load(neural_ode_dir / "pair_representation.pt", map_location='cpu')

    # Load OpenFold reference (from complete_blocks)
    from neural_ODE.train_evoformer_ode import DataManager
    data_dir = "/media/visitor/Extreme SSD/data/complete_blocks"

    m48_ref, z48_ref = DataManager.load_protein_blocks(
        protein_id, data_dir, 'cpu', max_cluster_size=256, target_block=48
    )

    # Add batch dimension to reference to match saved format
    m48_ref_batch = m48_ref.unsqueeze(0)
    z48_ref_batch = z48_ref.unsqueeze(0)

    print(f"   Neural ODE: M{msa_neural_ode.shape}, Z{pair_neural_ode.shape}")
    print(f"   OpenFold:   M{m48_ref_batch.shape}, Z{z48_ref_batch.shape}")

    # Check shapes match
    shapes_match = (msa_neural_ode.shape == m48_ref_batch.shape and
                    pair_neural_ode.shape == z48_ref_batch.shape)

    print(f"   Shapes match: {'✅' if shapes_match else '❌'}")

    if shapes_match:
        # Compute differences
        msa_diff = F.mse_loss(msa_neural_ode, m48_ref_batch).item()
        pair_diff = F.mse_loss(pair_neural_ode, z48_ref_batch).item()

        msa_corr = F.cosine_similarity(msa_neural_ode.flatten().unsqueeze(0),
                                       m48_ref_batch.flatten().unsqueeze(0)).item()
        pair_corr = F.cosine_similarity(pair_neural_ode.flatten().unsqueeze(0),
                                        z48_ref_batch.flatten().unsqueeze(0)).item()

        print(f"   MSE differences:")
        print(f"     MSA: {msa_diff:.6f}")
        print(f"     Pair: {pair_diff:.6f}")
        print(f"   Correlations:")
        print(f"     MSA: {msa_corr:.4f}")
        print(f"     Pair: {pair_corr:.4f}")

        # Assessment
        if msa_corr > 0.9 and pair_corr > 0.9:
            print(f"   ✅ Excellent match - predictions should work well")
        elif msa_corr > 0.7 and pair_corr > 0.7:
            print(f"   ⚠️  Good match - predictions might work")
        else:
            print(f"   ❌ Poor match - this explains bad structures")


def check_structure_module_compatibility(protein_id: str, predictions_dir: str):
    """Check if predictions are compatible with structure module"""

    print(f"\n🏗️  STRUCTURE MODULE COMPATIBILITY CHECK")
    print("=" * 45)

    pred_dir = Path(predictions_dir) / protein_id

    if not pred_dir.exists():
        print(f"❌ Predictions directory not found")
        return

    # Load predictions
    msa_pred = torch.load(pred_dir / "msa_representation.pt", map_location='cpu')
    pair_pred = torch.load(pred_dir / "pair_representation.pt", map_location='cpu')

    # Check OpenFold requirements
    checks = {
        'msa_batch_dim': msa_pred.dim() == 4 and msa_pred.shape[0] == 1,
        'pair_batch_dim': pair_pred.dim() == 4 and pair_pred.shape[0] == 1,
        'msa_channels': msa_pred.shape[-1] == 256,
        'pair_channels': pair_pred.shape[-1] == 128,
        'sequence_match': msa_pred.shape[2] == pair_pred.shape[1] == pair_pred.shape[2],
        'no_nan': not (torch.isnan(msa_pred).any() or torch.isnan(pair_pred).any()),
        'no_inf': not (torch.isinf(msa_pred).any() or torch.isinf(pair_pred).any()),
        'reasonable_values': (abs(msa_pred.mean()) < 100 and abs(pair_pred.mean()) < 100)
    }

    print(f"   Structure module requirements:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"     {check}: {status}")

    all_passed = all(checks.values())

    if all_passed:
        print(f"\n   ✅ Predictions should be compatible with structure module")
    else:
        print(f"\n   ❌ Compatibility issues found - this explains structure problems")

        # Provide specific fixes
        if not checks['msa_batch_dim'] or not checks['pair_batch_dim']:
            print(f"     FIX: Add batch dimension with .unsqueeze(0)")

        if not checks['reasonable_values']:
            print(f"     FIX: Values too extreme - check model training")

        if not checks['no_nan'] or not checks['no_inf']:
            print(f"     FIX: Remove NaN/Inf values from predictions")

    return checks


def main_analysis(protein_id: str = "1fv5_A", model_path: str = None, predictions_dir: str = None):
    """Run complete pipeline analysis"""

    print(f"🔍 NEURAL ODE → STRUCTURE PIPELINE ANALYSIS")
    print(f"Protein: {protein_id}")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find model and predictions if not specified
    if model_path is None:
        model_files = list(Path("../trained_models").glob("*1fv*final_model.pt"))
        if model_files:
            model_path = str(model_files[0])
            print(f"📦 Auto-found model: {model_path}")
        else:
            print(f"❌ No model found. Please specify --model_path")
            return

    if predictions_dir is None:
        pred_base = Path("/media/visitor/Extreme SSD/data/post_evoformer_predictions")
        pred_dirs = [d for d in pred_base.glob("predictions_*") if d.is_dir()]
        if pred_dirs:
            predictions_dir = str(pred_dirs[-1])  # Most recent
            print(f"📁 Auto-found predictions: {predictions_dir}")
        else:
            print(f"❌ No predictions found. Run test_model.py first")
            return

    # Run all analyses
    model, config = load_trained_model(model_path, device)

    # 1. Compare predictions vs actual
    pred_analysis = compare_representations(protein_id, model, device)

    # 2. Analyze saved files
    saved_analysis = analyze_saved_predictions(protein_id, predictions_dir)

    # 3. Compare with reference
    compare_with_openfold_reference(protein_id, predictions_dir)

    # 4. Check structure compatibility
    compatibility = check_structure_module_compatibility(protein_id, predictions_dir)

    # Final diagnosis
    print(f"\n🎯 FINAL DIAGNOSIS")
    print("=" * 20)

    if pred_analysis['correlations']['msa'] > 0.9 and pred_analysis['correlations']['pair'] > 0.9:
        print(f"✅ Neural ODE predictions are high quality")
    else:
        print(f"❌ Neural ODE predictions are poor quality")
        print(f"   MSA correlation: {pred_analysis['correlations']['msa']:.3f}")
        print(f"   Pair correlation: {pred_analysis['correlations']['pair']:.3f}")

    if compatibility and all(compatibility.values()):
        print(f"✅ Predictions are compatible with structure module")
    else:
        print(f"❌ Predictions have structure module compatibility issues")

    if pred_analysis['issues']:
        print(f"⚠️  Issues found:")
        for issue in pred_analysis['issues']:
            print(f"     {issue}")


if __name__ == "__main__":
    print("🔍 NEURAL ODE → STRUCTURE PIPELINE DEBUGGER")
    print("This will identify exactly where structure prediction fails")
    print("")

    # =======================================================================================
    # MANUAL CONFIGURATION - EDIT THESE PATHS DIRECTLY
    # =======================================================================================

    # Protein to analyze
    PROTEIN_ID = "1fv5_A"

    # Path to your trained model (.pt file)
    MODEL_PATH = "/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250617_144628_1fv5_final_model.pt"

    # Directory containing predictions (the folder with protein_id subdirs)
    PREDICTIONS_DIR = "/media/visitor/Extreme SSD/data/post_evoformer_predictions/predictions_20250617_144628_1fv5"

    # =======================================================================================
    # END CONFIGURATION - No need to edit below
    # =======================================================================================

    print(f"Configuration:")
    print(f"  Protein ID: {PROTEIN_ID}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print("")

    # Check if paths exist
    if MODEL_PATH and not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Please update MODEL_PATH in the script")
        exit(1)

    if PREDICTIONS_DIR and not os.path.exists(PREDICTIONS_DIR):
        print(f"❌ Predictions directory not found: {PREDICTIONS_DIR}")
        print("   Please update PREDICTIONS_DIR in the script")
        exit(1)

    # Run analysis
    main_analysis(PROTEIN_ID, MODEL_PATH, PREDICTIONS_DIR)