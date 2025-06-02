#!/usr/bin/env python3
"""
Loss Analysis Script for Evoformer Neural ODE Training
Determines what constitutes good vs bad loss values by analyzing baselines
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from datetime import datetime


def load_protein_block(protein_id: str, block_idx: int, data_dir: str, max_cluster_size: int = None) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Load M and Z tensors for a specific block"""
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, f"m_block_{block_idx}.pt")
    z_path = os.path.join(protein_dir, f"z_block_{block_idx}.pt")

    if not os.path.exists(m_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Block files not found for {protein_id} block {block_idx}")

    m = torch.load(m_path, map_location='cpu')
    z = torch.load(z_path, map_location='cpu')

    # Remove batch dimension
    if m.dim() == 4:
        m = m.squeeze(0)
    if z.dim() == 4:
        z = z.squeeze(0)

    # Limit cluster size for memory efficiency
    if max_cluster_size and m.shape[0] > max_cluster_size:
        m = m[:max_cluster_size]

    return m, z


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


def analyze_baseline_losses(protein_id: str, data_dir: str, max_cluster_size: int = 64) -> Dict:
    """Analyze different baseline prediction strategies for a protein"""
    print(f"🔍 Analyzing baselines for {protein_id}...")

    try:
        # Load consecutive blocks
        m0, z0 = load_protein_block(protein_id, 0, data_dir, max_cluster_size)
        m1, z1 = load_protein_block(protein_id, 1, data_dir, max_cluster_size)

        # Find available blocks for this protein
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
        available_blocks = []
        for i in range(50):  # Check up to block 50
            m_path = os.path.join(protein_dir, f"m_block_{i}.pt")
            if os.path.exists(m_path):
                available_blocks.append(i)
            else:
                break

        results = {
            'protein_id': protein_id,
            'num_sequences': m0.shape[0],
            'num_residues': m0.shape[1],
            'msa_channels': m0.shape[2],
            'pair_channels': z0.shape[2],
            'available_blocks': len(available_blocks),
            'baselines': {}
        }

        print(f"  📊 Shape: MSA {m0.shape}, Pair {z0.shape}")
        print(f"  📋 Available blocks: {len(available_blocks)}")

        # 1. Identity baseline (predict block 0 as block 1)
        identity_loss = compute_adaptive_loss(m0, m1, z0, z1)
        results['baselines']['identity'] = identity_loss
        print(f"  🔄 Identity loss: {identity_loss['total']:.3f}")

        # 2. Zero prediction baseline
        zero_m = torch.zeros_like(m1)
        zero_z = torch.zeros_like(z1)
        zero_loss = compute_adaptive_loss(zero_m, m1, zero_z, z1)
        results['baselines']['zero'] = zero_loss
        print(f"  0️⃣  Zero loss: {zero_loss['total']:.3f}")

        # 3. Mean prediction baseline
        mean_m = torch.full_like(m1, m1.mean())
        mean_z = torch.full_like(z1, z1.mean())
        mean_loss = compute_adaptive_loss(mean_m, m1, mean_z, z1)
        results['baselines']['mean'] = mean_loss
        print(f"  📊 Mean loss: {mean_loss['total']:.3f}")

        # 4. Random prediction baseline (Gaussian noise)
        random_m = torch.randn_like(m1) * m1.std() + m1.mean()
        random_z = torch.randn_like(z1) * z1.std() + z1.mean()
        random_loss = compute_adaptive_loss(random_m, m1, random_z, z1)
        results['baselines']['random'] = random_loss
        print(f"  🎲 Random loss: {random_loss['total']:.3f}")

        # 5. Small perturbation baseline (identity + small noise)
        noise_scale = 0.01
        perturb_m = m0 + torch.randn_like(m0) * noise_scale
        perturb_z = z0 + torch.randn_like(z0) * noise_scale
        perturb_loss = compute_adaptive_loss(perturb_m, m1, perturb_z, z1)
        results['baselines']['small_perturbation'] = perturb_loss
        print(f"  🔀 Small perturbation loss: {perturb_loss['total']:.3f}")

        # 6. Linear interpolation baseline (halfway between blocks 0 and 1)
        if len(available_blocks) >= 3:
            m2, z2 = load_protein_block(protein_id, 2, data_dir, max_cluster_size)
            interp_m = 0.5 * (m0 + m2)  # Interpolate between 0 and 2 to predict 1
            interp_z = 0.5 * (z0 + z2)
            interp_loss = compute_adaptive_loss(interp_m, m1, interp_z, z1)
            results['baselines']['interpolation'] = interp_loss
            print(f"  📈 Interpolation loss: {interp_loss['total']:.3f}")

        # 7. Data statistics
        results['data_stats'] = {
            'm0_mean': m0.mean().item(),
            'm0_std': m0.std().item(),
            'm0_min': m0.min().item(),
            'm0_max': m0.max().item(),
            'm1_mean': m1.mean().item(),
            'm1_std': m1.std().item(),
            'm1_min': m1.min().item(),
            'm1_max': m1.max().item(),
            'z0_mean': z0.mean().item(),
            'z0_std': z0.std().item(),
            'z0_min': z0.min().item(),
            'z0_max': z0.max().item(),
            'z1_mean': z1.mean().item(),
            'z1_std': z1.std().item(),
            'z1_min': z1.min().item(),
            'z1_max': z1.max().item(),
        }

        return results

    except Exception as e:
        print(f"  ❌ Error analyzing {protein_id}: {e}")
        return None


def analyze_multiple_proteins(data_dir: str, max_proteins: int = 10, max_cluster_size: int = 64) -> Dict:
    """Analyze baseline losses across multiple proteins"""
    print(f"🧬 Analyzing baseline losses across proteins...")
    print(f"📁 Data directory: {data_dir}")
    print(f"🔢 Max proteins: {max_proteins}")
    print(f"📏 Max cluster size: {max_cluster_size}")

    # Find available proteins
    proteins = []
    for item in Path(data_dir).iterdir():
        if item.is_dir() and item.name.endswith('_evoformer_blocks'):
            protein_id = item.name.replace('_evoformer_blocks', '')
            proteins.append(protein_id)

    proteins = sorted(proteins)[:max_proteins]
    print(f"🧬 Found {len(proteins)} proteins: {proteins}")

    all_results = []
    baseline_stats = {
        'identity': [],
        'zero': [],
        'mean': [],
        'random': [],
        'small_perturbation': [],
        'interpolation': []
    }

    for i, protein_id in enumerate(proteins):
        print(f"\n[{i + 1}/{len(proteins)}] {protein_id}")
        result = analyze_baseline_losses(protein_id, data_dir, max_cluster_size)

        if result:
            all_results.append(result)

            # Collect baseline statistics
            for baseline_name, baseline_data in result['baselines'].items():
                if baseline_name in baseline_stats:
                    baseline_stats[baseline_name].append(baseline_data['total'])

    # Calculate summary statistics
    summary = {
        'num_proteins': len(all_results),
        'proteins_analyzed': [r['protein_id'] for r in all_results],
        'baseline_summary': {}
    }

    for baseline_name, losses in baseline_stats.items():
        if losses:
            summary['baseline_summary'][baseline_name] = {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'min': np.min(losses),
                'max': np.max(losses),
                'median': np.median(losses),
                'count': len(losses)
            }

    return {
        'summary': summary,
        'individual_results': all_results,
        'analysis_date': datetime.now().isoformat()
    }


def determine_loss_thresholds(analysis_results: Dict) -> Dict:
    """Determine what constitutes good vs bad loss based on baseline analysis"""
    print(f"\n🎯 Determining loss thresholds...")

    baselines = analysis_results['summary']['baseline_summary']

    # Extract key baseline values
    identity_mean = baselines.get('identity', {}).get('mean', float('inf'))
    zero_mean = baselines.get('zero', {}).get('mean', float('inf'))
    mean_mean = baselines.get('mean', {}).get('mean', float('inf'))
    random_mean = baselines.get('random', {}).get('mean', float('inf'))
    perturb_mean = baselines.get('small_perturbation', {}).get('mean', float('inf'))

    # Define thresholds based on baseline performance
    thresholds = {
        'excellent': {
            'threshold': identity_mean * 0.5,
            'description': 'Much better than identity mapping',
            'color': '🟢'
        },
        'good': {
            'threshold': identity_mean * 0.8,
            'description': 'Better than identity mapping',
            'color': '🟡'
        },
        'acceptable': {
            'threshold': identity_mean * 1.2,
            'description': 'Close to identity mapping',
            'color': '🟠'
        },
        'poor': {
            'threshold': max(zero_mean, mean_mean) * 0.8,
            'description': 'Approaching trivial baselines',
            'color': '🔴'
        },
        'terrible': {
            'threshold': random_mean,
            'description': 'Worse than random predictions',
            'color': '💀'
        }
    }

    # Add reference baselines
    thresholds['baselines'] = {
        'identity': identity_mean,
        'zero': zero_mean,
        'mean': mean_mean,
        'random': random_mean,
        'small_perturbation': perturb_mean
    }

    return thresholds


def print_loss_analysis(analysis_results: Dict, thresholds: Dict):
    """Print comprehensive loss analysis"""
    print(f"\n" + "=" * 60)
    print(f"📊 LOSS ANALYSIS REPORT")
    print(f"=" * 60)

    summary = analysis_results['summary']
    baselines = summary['baseline_summary']

    print(f"\n🧬 Dataset Summary:")
    print(f"  Proteins analyzed: {summary['num_proteins']}")
    print(f"  Proteins: {', '.join(summary['proteins_analyzed'][:5])}")
    if len(summary['proteins_analyzed']) > 5:
        print(f"           ... and {len(summary['proteins_analyzed']) - 5} more")

    print(f"\n📊 Baseline Performance (Adaptive Loss):")
    print(f"{'Baseline':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print(f"-" * 60)

    baseline_order = ['identity', 'small_perturbation', 'interpolation', 'mean', 'zero', 'random']
    for baseline_name in baseline_order:
        if baseline_name in baselines:
            stats = baselines[baseline_name]
            print(
                f"{baseline_name:<20} {stats['mean']:<8.3f} {stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f}")

    print(f"\n🎯 LOSS QUALITY THRESHOLDS:")
    print(f"{'Quality':<12} {'Threshold':<10} {'Description'}")
    print(f"-" * 50)

    for quality, info in thresholds.items():
        if quality != 'baselines':
            print(f"{info['color']} {quality:<10} < {info['threshold']:<9.3f} {info['description']}")

    print(f"\n📈 TRAINING LOSS INTERPRETATION:")
    print(f"🟢 EXCELLENT  (< {thresholds['excellent']['threshold']:.3f}): Your model is learning very well!")
    print(f"🟡 GOOD       (< {thresholds['good']['threshold']:.3f}): Model is learning, continue training")
    print(f"🟠 ACCEPTABLE (< {thresholds['acceptable']['threshold']:.3f}): Model is barely learning")
    print(f"🔴 POOR       (< {thresholds['poor']['threshold']:.3f}): Model is struggling, check hyperparameters")
    print(f"💀 TERRIBLE   (> {thresholds['terrible']['threshold']:.3f}): Model is broken, major issues!")

    print(f"\n💡 RECOMMENDATIONS:")
    print(
        f"• If your loss > {thresholds['baselines']['identity']:.1f}: Model worse than identity - reduce learning rate")
    print(f"• If your loss > {thresholds['baselines']['zero']:.1f}: Model worse than zeros - check data/model")
    print(f"• If your loss > {thresholds['baselines']['random']:.1f}: Model completely broken - restart")
    print(f"• Target loss for good training: < {thresholds['good']['threshold']:.3f}")


def evaluate_training_loss(loss_value: float, thresholds: Dict) -> str:
    """Evaluate a given training loss value"""
    if loss_value < thresholds['excellent']['threshold']:
        return f"🟢 EXCELLENT ({loss_value:.3f}) - {thresholds['excellent']['description']}"
    elif loss_value < thresholds['good']['threshold']:
        return f"🟡 GOOD ({loss_value:.3f}) - {thresholds['good']['description']}"
    elif loss_value < thresholds['acceptable']['threshold']:
        return f"🟠 ACCEPTABLE ({loss_value:.3f}) - {thresholds['acceptable']['description']}"
    elif loss_value < thresholds['poor']['threshold']:
        return f"🔴 POOR ({loss_value:.3f}) - {thresholds['poor']['description']}"
    else:
        return f"💀 TERRIBLE ({loss_value:.3f}) - {thresholds['terrible']['description']}"


def save_analysis(analysis_results: Dict, thresholds: Dict, output_file: str):
    """Save analysis results to JSON file"""
    output_data = {
        'analysis_results': analysis_results,
        'thresholds': thresholds,
        'metadata': {
            'script_version': '1.0',
            'analysis_date': datetime.now().isoformat(),
            'description': 'Evoformer Neural ODE Loss Analysis'
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"📄 Analysis saved to: {output_file}")


def main(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser(description='Analyze what constitutes good vs bad loss for Evoformer Neural ODE')

        parser.add_argument('--data_dir', type=str, required=True,
                            help='Directory containing protein evoformer blocks')
        parser.add_argument('--max_proteins', type=int, default=10,
                            help='Maximum number of proteins to analyze')
        parser.add_argument('--max_cluster_size', type=int, default=64,
                            help='Maximum cluster size to use (for memory)')
        parser.add_argument('--output_file', type=str, default=None,
                            help='Output JSON file for results')
        parser.add_argument('--evaluate_loss', type=float, default=None,
                            help='Evaluate a specific loss value')

        args = parser.parse_args()
    else:
        # Parse from dictionary (used when calling main() directly)
        class Args:
            pass
        args = Args()
        args.data_dir = args_dict.get("data_dir")
        args.max_proteins = args_dict.get("max_proteins", 10)
        args.max_cluster_size = args_dict.get("max_cluster_size", 64)
        args.output_file = args_dict.get("output_file", None)
        args.evaluate_loss = args_dict.get("evaluate_loss", None)

    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"loss_analysis_{timestamp}.json"

    print("🔬 EVOFORMER NEURAL ODE LOSS ANALYZER")
    print("=" * 50)

    analysis_results = analyze_multiple_proteins(
        args.data_dir,
        args.max_proteins,
        args.max_cluster_size
    )

    thresholds = determine_loss_thresholds(analysis_results)

    print_loss_analysis(analysis_results, thresholds)

    if args.evaluate_loss is not None:
        print(f"\n🎯 EVALUATING YOUR LOSS:")
        evaluation = evaluate_training_loss(args.evaluate_loss, thresholds)
        print(f"   {evaluation}")

    save_analysis(analysis_results, thresholds, args.output_file)

    print(f"\n✅ Analysis complete!")
    print(f"💡 Use --evaluate_loss <value> to evaluate specific training losses")

if __name__ == "__main__":
    main({
        "data_dir": Path("/media/visitor/Extreme SSD/data/complete_blocks"),
        "max_proteins": 10
    })