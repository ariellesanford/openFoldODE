#!/usr/bin/env python3
"""
Test a trained Neural ODE model on test data
Loads a model from outputs folder and evaluates on test proteins
"""

import os
import torch
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime
from torchdiffeq import odeint
import torch.nn.functional as F
from evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast


def load_split_proteins(splits_dir: str, mode: str) -> List[str]:
    """Load protein IDs from the appropriate split file"""
    split_files = {
        'training': 'training_chains.txt',
        'validation': 'validation_chains.txt',
        'testing': 'testing_chains.txt'
    }

    split_file = os.path.join(splits_dir, split_files[mode])

    proteins = []
    with open(split_file, 'r') as f:
        for line in f:
            protein_id = line.strip()
            if protein_id:
                proteins.append(protein_id)

    return proteins


def get_available_proteins(data_dir: str, splits_dir: str, mode: str) -> List[str]:
    """Get list of available protein IDs for the specified mode"""
    split_proteins = load_split_proteins(splits_dir, mode)

    available_proteins = []
    for protein_id in split_proteins:
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
        if os.path.isdir(protein_dir):
            available_proteins.append(protein_id)

    return available_proteins


def get_protein_info(protein_id: str, data_dir: str) -> Dict:
    """Get basic info about a protein"""
    try:
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
        m_path = os.path.join(protein_dir, "m_block_0.pt")

        if os.path.exists(m_path):
            m = torch.load(m_path, map_location='cpu')
            if m.dim() == 4:
                m = m.squeeze(0)

            return {
                'protein_id': protein_id,
                'num_residues': m.shape[-2],
                'num_sequences': m.shape[0],
                'msa_channels': m.shape[2]
            }
    except Exception as e:
        return None


def load_protein_blocks(protein_id: str, data_dir: str, device: str, max_cluster_size: int = None) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load M and Z tensors for blocks 0 and 48"""
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

    # Load block 0
    m0 = torch.load(os.path.join(protein_dir, "m_block_0.pt"), map_location=device)
    z0 = torch.load(os.path.join(protein_dir, "z_block_0.pt"), map_location=device)

    # Load block 48
    m48 = torch.load(os.path.join(protein_dir, "m_block_48.pt"), map_location=device)
    z48 = torch.load(os.path.join(protein_dir, "z_block_48.pt"), map_location=device)

    # Remove batch dimension
    if m0.dim() == 4:
        m0 = m0.squeeze(0)
    if z0.dim() == 4:
        z0 = z0.squeeze(0)
    if m48.dim() == 4:
        m48 = m48.squeeze(0)
    if z48.dim() == 4:
        z48 = z48.squeeze(0)

    # Limit cluster size for memory efficiency
    if max_cluster_size and m0.shape[0] > max_cluster_size:
        m0 = m0[:max_cluster_size]
        m48 = m48[:max_cluster_size]

    return m0, z0, m48, z48


def compute_adaptive_loss(pred_m: torch.Tensor, target_m: torch.Tensor,
                          pred_z: torch.Tensor, target_z: torch.Tensor) -> Dict[str, float]:
    """Compute the same adaptive loss as in training"""
    # Standard MSE losses
    msa_loss = F.mse_loss(pred_m, target_m)
    pair_loss = F.mse_loss(pred_z, target_z)

    # Scale-aware losses (normalized by variance to be scale-invariant)
    msa_variance = target_m.var() + 1e-8
    pair_variance = target_z.var() + 1e-8

    msa_scaled = msa_loss / msa_variance
    pair_scaled = pair_loss / pair_variance

    # Balanced loss (equal contribution from MSA and pair)
    total_loss = msa_scaled + pair_scaled

    return {
        'total': total_loss.item(),
        'msa_raw': msa_loss.item(),
        'pair_raw': pair_loss.item(),
        'msa_scaled': msa_scaled.item(),
        'pair_scaled': pair_scaled.item(),
        'msa_variance': msa_variance.item(),
        'pair_variance': pair_variance.item()
    }


def test_single_protein(protein_id: str, model: torch.nn.Module, config: Dict, device: str,
                        save_predictions: bool = False, predictions_dir: str = None) -> Dict:
    """Test model on a single protein"""

    # Load protein data
    m0, z0, m48, z48 = load_protein_blocks(
        protein_id,
        config['data_dir'],
        device,
        config.get('reduced_cluster_size')
    )

    with torch.no_grad():
        # Run model prediction (0 → 48 transformation)
        start_time = time.time()

        trajectory = odeint(
            model,
            (m0, z0),
            torch.tensor([0.0, 1.0]).to(device),
            method=config.get('integrator', 'rk4'),
            rtol=1e-4,
            atol=1e-5
        )

        inference_time = time.time() - start_time

        # Get final prediction
        m_pred = trajectory[0][-1]
        z_pred = trajectory[1][-1]

        # Save predictions for structure module if requested
        if save_predictions and predictions_dir:
            protein_pred_dir = os.path.join(predictions_dir, protein_id)
            os.makedirs(protein_pred_dir, exist_ok=True)

            # Add batch dimension back for compatibility with OpenFold structure module
            m_pred_batch = m_pred.unsqueeze(0)  # [1, N_seq, N_res, c_m]
            z_pred_batch = z_pred.unsqueeze(0)  # [1, N_res, N_res, c_z]

            # Save tensors
            torch.save(m_pred_batch.cpu(), os.path.join(protein_pred_dir, "msa_representation.pt"))
            torch.save(z_pred_batch.cpu(), os.path.join(protein_pred_dir, "pair_representation.pt"))

            # Save metadata for structure module
            metadata = {
                'protein_id': protein_id,
                'num_residues': m_pred.shape[-2],
                'num_sequences': m_pred.shape[0],
                'msa_channels': m_pred.shape[-1],
                'pair_channels': z_pred.shape[-1],
                'model_prediction': True,
                'evoformer_complete': True
            }

            import json
            with open(os.path.join(protein_pred_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

        # Compute loss components
        loss_dict = compute_adaptive_loss(m_pred, m48, z_pred, z48)

        # Compute additional metrics
        # MSA similarity (cosine similarity)
        m_pred_flat = m_pred.flatten()
        m48_flat = m48.flatten()
        msa_cosine_sim = F.cosine_similarity(m_pred_flat.unsqueeze(0), m48_flat.unsqueeze(0)).item()

        # Pair similarity
        z_pred_flat = z_pred.flatten()
        z48_flat = z48.flatten()
        pair_cosine_sim = F.cosine_similarity(z_pred_flat.unsqueeze(0), z48_flat.unsqueeze(0)).item()

        # Prediction quality metrics
        msa_mae = F.l1_loss(m_pred, m48).item()
        pair_mae = F.l1_loss(z_pred, z48).item()

        # Relative error
        msa_rel_error = (F.mse_loss(m_pred, m48) / (m48.var() + 1e-8)).item()
        pair_rel_error = (F.mse_loss(z_pred, z48) / (z48.var() + 1e-8)).item()

    return {
        'protein_id': protein_id,
        'inference_time': inference_time,
        'loss': loss_dict,
        'msa_cosine_similarity': msa_cosine_sim,
        'pair_cosine_similarity': pair_cosine_sim,
        'msa_mae': msa_mae,
        'pair_mae': pair_mae,
        'msa_relative_error': msa_rel_error,
        'pair_relative_error': pair_rel_error,
        'num_residues': m0.shape[-2],
        'num_sequences': m0.shape[0],
        'predictions_saved': save_predictions
    }


def load_model(model_path: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained model and its configuration"""

    print(f"📦 Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Extract configuration
    config = checkpoint.get('config', {})

    # Determine model architecture
    c_m = 256  # MSA embedding dimension
    c_z = 128  # Pair embedding dimension
    hidden_dim = config.get('hidden_dim', 64)
    use_fast_ode = config.get('use_fast_ode', True)

    # Initialize model
    if use_fast_ode:
        model = EvoformerODEFuncFast(c_m, c_z, hidden_dim).to(device)
    else:
        model = EvoformerODEFunc(c_m, c_z, hidden_dim).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Architecture: {'EvoformerODEFuncFast' if use_fast_ode else 'EvoformerODEFunc'}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")

    # Extract training stats if available
    training_stats = checkpoint.get('training_stats', {})
    if training_stats:
        print(f"   Training epochs: {training_stats.get('total_epochs', 'N/A')}")
        print(f"   Best val loss: {training_stats.get('best_val_loss', 'N/A')}")
        print(f"   Training method: {training_stats.get('method', 'N/A')}")

    return model, config


def filter_proteins_by_size(proteins: List[str], data_dir: str, max_residues: int = None) -> List[str]:
    """Filter proteins by residue count"""
    if max_residues is None:
        return proteins

    valid_proteins = []
    for protein_id in proteins:
        info = get_protein_info(protein_id, data_dir)
        if info and info['num_residues'] <= max_residues:
            valid_proteins.append(protein_id)

    return valid_proteins


def print_results_summary(results: List[Dict]):
    """Print a summary of test results"""

    if not results:
        print("❌ No test results to summarize")
        return

    # Compute statistics
    total_losses = [r['loss']['total'] for r in results]
    msa_cosine_sims = [r['msa_cosine_similarity'] for r in results]
    pair_cosine_sims = [r['pair_cosine_similarity'] for r in results]
    inference_times = [r['inference_time'] for r in results]
    protein_sizes = [r['num_residues'] for r in results]

    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Proteins tested: {len(results)}")
    print(f"Average protein size: {sum(protein_sizes) / len(protein_sizes):.1f} residues")
    print(f"Size range: [{min(protein_sizes)}, {max(protein_sizes)}] residues")

    print(f"\n🎯 LOSS METRICS:")
    print(f"Average total loss: {sum(total_losses) / len(total_losses):.4f}")
    print(f"Loss std: {torch.tensor(total_losses).std().item():.4f}")
    print(f"Loss range: [{min(total_losses):.4f}, {max(total_losses):.4f}]")

    print(f"\n📐 SIMILARITY METRICS:")
    print(f"Average MSA cosine similarity: {sum(msa_cosine_sims) / len(msa_cosine_sims):.4f}")
    print(f"Average Pair cosine similarity: {sum(pair_cosine_sims) / len(pair_cosine_sims):.4f}")

    print(f"\n⏱️  PERFORMANCE:")
    print(f"Average inference time: {sum(inference_times) / len(inference_times):.2f} seconds")
    print(f"Total test time: {sum(inference_times):.1f} seconds")

    # Find best and worst performers
    best_idx = min(range(len(results)), key=lambda i: results[i]['loss']['total'])
    worst_idx = max(range(len(results)), key=lambda i: results[i]['loss']['total'])

    print(f"\n🏆 BEST PERFORMER:")
    best = results[best_idx]
    print(f"   {best['protein_id']}: loss={best['loss']['total']:.4f}, "
          f"msa_sim={best['msa_cosine_similarity']:.4f}, "
          f"pair_sim={best['pair_cosine_similarity']:.4f}")

    print(f"\n📉 WORST PERFORMER:")
    worst = results[worst_idx]
    print(f"   {worst['protein_id']}: loss={worst['loss']['total']:.4f}, "
          f"msa_sim={worst['msa_cosine_similarity']:.4f}, "
          f"pair_sim={worst['pair_cosine_similarity']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test trained Neural ODE model on test data')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing protein data')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory containing data splits')

    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--max_residues', type=int, default=None,
                        help='Skip proteins larger than this')
    parser.add_argument('--max_proteins', type=int, default=None,
                        help='Test only first N proteins')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save detailed results to JSON file')
    parser.add_argument('--protein_list', type=str, nargs='+', default=None,
                        help='Test specific proteins (list of IDs)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predicted m and z tensors for structure module')
    parser.add_argument('--predictions_dir', type=str, default='post_evoformer_predictions',
                        help='Directory to save predictions for structure module')

    args = parser.parse_args()

    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'

    print("🧪 NEURAL ODE MODEL TESTING")
    print("=" * 50)
    print(f"📦 Model: {args.model_path}")
    print(f"📁 Data: {args.data_dir}")
    print(f"📂 Splits: {args.splits_dir}")
    print(f"💻 Device: {args.device}")

    # Set up predictions directory if saving
    predictions_dir = None
    if args.save_predictions:
        predictions_dir = args.predictions_dir
        os.makedirs(predictions_dir, exist_ok=True)
        print(f"💾 Predictions will be saved to: {predictions_dir}")
        print(f"   Format: {predictions_dir}/{{protein_id}}/{{msa|pair}}_representation.pt")

    # Load model
    model, config = load_model(args.model_path, args.device)

    # Update config with current args
    config['data_dir'] = args.data_dir

    # Get test proteins
    if args.protein_list:
        # Use specific proteins provided
        test_proteins = args.protein_list
        print(f"🎯 Testing specific proteins: {test_proteins}")
    else:
        # Use test split
        test_proteins = get_available_proteins(args.data_dir, args.splits_dir, 'testing')
        print(f"📋 Found {len(test_proteins)} test proteins")

    if not test_proteins:
        print("❌ No test proteins found!")
        return 1

    # Filter by size if specified
    if args.max_residues:
        original_count = len(test_proteins)
        test_proteins = filter_proteins_by_size(test_proteins, args.data_dir, args.max_residues)
        print(f"📏 Filtered by size (≤{args.max_residues}): {len(test_proteins)}/{original_count}")

    # Limit number if specified
    if args.max_proteins:
        test_proteins = test_proteins[:args.max_proteins]
        print(f"🔢 Limited to first {len(test_proteins)} proteins")

    print(f"\n🧬 Testing {len(test_proteins)} proteins:")
    for protein_id in test_proteins:
        info = get_protein_info(protein_id, args.data_dir)
        if info:
            print(f"   {protein_id}: {info['num_residues']} residues, {info['num_sequences']} sequences")

    # Run tests
    print(f"\n🚀 Starting tests...")
    results = []
    start_time = time.time()

    for i, protein_id in enumerate(test_proteins):
        print(f"\n[{i + 1}/{len(test_proteins)}] Testing {protein_id}... ", end='', flush=True)

        try:
            result = test_single_protein(
                protein_id, model, config, args.device,
                save_predictions=args.save_predictions,
                predictions_dir=predictions_dir
            )
            results.append(result)

            print(f"✅")
            print(f"   Loss: {result['loss']['total']:.4f}")
            print(f"   MSA similarity: {result['msa_cosine_similarity']:.4f}")
            print(f"   Pair similarity: {result['pair_cosine_similarity']:.4f}")
            print(f"   Time: {result['inference_time']:.2f}s")
            if result['predictions_saved']:
                print(f"   💾 Predictions saved: {predictions_dir}/{protein_id}/")

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}...")
            continue

    total_time = time.time() - start_time

    # Print summary
    print_results_summary(results)

    print(f"\n⏱️  Total testing time: {total_time:.1f} seconds")
    print(f"📈 Success rate: {len(results)}/{len(test_proteins)} ({len(results) / len(test_proteins) * 100:.1f}%)")

    # Save detailed results if requested
    if args.output_file:
        output_data = {
            'model_path': args.model_path,
            'test_date': datetime.now().isoformat(),
            'config': config,
            'test_proteins': test_proteins,
            'results': results,
            'summary': {
                'total_proteins': len(test_proteins),
                'successful_tests': len(results),
                'average_loss': sum(r['loss']['total'] for r in results) / len(results) if results else None,
                'average_msa_similarity': sum(r['msa_cosine_similarity'] for r in results) / len(
                    results) if results else None,
                'average_pair_similarity': sum(r['pair_cosine_similarity'] for r in results) / len(
                    results) if results else None,
                'total_test_time': total_time
            }
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Detailed results saved to: {args.output_file}")

    # Save summary of predictions if they were saved
    if args.save_predictions and results:
        print(f"\n📦 PREDICTIONS SAVED FOR STRUCTURE MODULE:")
        print(f"   Directory: {predictions_dir}")
        print(f"   Proteins: {len(results)}")
        print(f"   Files per protein:")
        print(f"     - msa_representation.pt   [N_seq, N_res, 256]")
        print(f"     - pair_representation.pt  [N_res, N_res, 128]")
        print(f"     - metadata.json")
        print(f"   Ready for OpenFold structure module!")

    print(f"\n🎯 Testing complete!")
    return 0


if __name__ == "__main__":
    exit(main())