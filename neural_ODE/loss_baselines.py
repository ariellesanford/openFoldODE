
"""
Enhanced Loss Analysis Script for Evoformer Neural ODE Training
Two modes:
1. baselines_only: Compute loss baselines on validation data (block 0→48 and incremental)
2. hyperparameter_search: Test hyperparameter configurations with training
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import gc
import subprocess
import sys
import random
from itertools import product
import copy


def load_protein_block(protein_id: str, block_idx: int, data_dir: str, reduced_cluster_size: int = None) -> Tuple[
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
    if reduced_cluster_size and m.shape[0] > reduced_cluster_size:
        m = m[:reduced_cluster_size]

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


def load_split_proteins(splits_dir: str, mode: str) -> List[str]:
    """Load protein IDs from the appropriate split file"""
    split_files = {
        'training': 'training_chains.txt',
        'validation': 'validation_chains.txt',
        'testing': 'testing_chains.txt'
    }

    split_file = os.path.join(splits_dir, split_files[mode])
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

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

    # Check which ones actually exist in data_dir
    available_proteins = []
    for protein_id in split_proteins:
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
        if os.path.isdir(protein_dir):
            available_proteins.append(protein_id)

    return available_proteins


def get_protein_info(protein_id: str, data_dir: str) -> Dict:
    """Get basic info about a protein"""
    try:
        m0, _ = load_protein_block(protein_id, 0, data_dir)
        num_residues = m0.shape[-2]
        num_sequences = m0.shape[0]

        # Count available blocks
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
        available_blocks = 0
        for i in range(50):  # Check up to block 49
            m_path = os.path.join(protein_dir, f"m_block_{i}.pt")
            if os.path.exists(m_path):
                available_blocks += 1
            else:
                break

        return {
            'protein_id': protein_id,
            'num_residues': num_residues,
            'num_sequences': num_sequences,
            'available_blocks': available_blocks
        }
    except Exception as e:
        return None


def compute_incremental_baseline_loss(protein_id: str, data_dir: str, baseline_type: str,
                                      reduced_cluster_size: int = None) -> float:
    """Compute incremental baseline loss across all blocks (like current training)"""
    total_loss = 0
    num_comparisons = 0

    # Load initial block
    m_prev, z_prev = load_protein_block(protein_id, 0, data_dir, reduced_cluster_size)

    # Iterate through all subsequent blocks
    for block_idx in range(1, 49):  # blocks 1 through 48
        try:
            m_target, z_target = load_protein_block(protein_id, block_idx, data_dir, reduced_cluster_size)

            # Generate baseline prediction
            if baseline_type == 'identity':
                m_pred, z_pred = m_prev, z_prev
            elif baseline_type == 'zero':
                m_pred = torch.zeros_like(m_target)
                z_pred = torch.zeros_like(z_target)
            elif baseline_type == 'mean':
                m_pred = torch.full_like(m_target, m_target.mean())
                z_pred = torch.full_like(z_target, z_target.mean())
            elif baseline_type == 'random':
                m_pred = torch.randn_like(m_target) * m_target.std() + m_target.mean()
                z_pred = torch.randn_like(z_target) * z_target.std() + z_target.mean()
            elif baseline_type == 'small_perturbation':
                noise_scale = 0.01
                m_pred = m_prev + torch.randn_like(m_prev) * noise_scale
                z_pred = z_prev + torch.randn_like(z_prev) * noise_scale

            # Compute loss for this step
            loss_dict = compute_adaptive_loss(m_pred, m_target, z_pred, z_target)
            total_loss += loss_dict['total']
            num_comparisons += 1

            # Update previous for next iteration (for identity baseline)
            m_prev, z_prev = m_target, z_target

        except FileNotFoundError:
            break  # Stop if we run out of blocks

    return total_loss / max(num_comparisons, 1)


def analyze_baseline_losses(protein_id: str, data_dir: str, reduced_cluster_size: int = 64) -> Dict:
    """Analyze different baseline prediction strategies for a protein"""
    try:
        # Load initial and final blocks
        m0, z0 = load_protein_block(protein_id, 0, data_dir, reduced_cluster_size)
        m48, z48 = load_protein_block(protein_id, 48, data_dir, reduced_cluster_size)

        protein_info = get_protein_info(protein_id, data_dir)
        if not protein_info:
            return None

        results = {
            'protein_id': protein_id,
            'num_sequences': m0.shape[0],
            'num_residues': m0.shape[-2],
            'msa_channels': m0.shape[2],
            'pair_channels': z0.shape[2],
            'available_blocks': protein_info['available_blocks'],
            'reduced_cluster_size': reduced_cluster_size,
            'baselines_0_to_48': {},
            'baselines_incremental': {}
        }

        baseline_types = ['identity', 'zero', 'mean', 'random', 'small_perturbation']

        # Compute 0→48 baselines
        for baseline_type in baseline_types:
            if baseline_type == 'identity':
                m_pred, z_pred = m0, z0
            elif baseline_type == 'zero':
                m_pred = torch.zeros_like(m48)
                z_pred = torch.zeros_like(z48)
            elif baseline_type == 'mean':
                m_pred = torch.full_like(m48, m48.mean())
                z_pred = torch.full_like(z48, z48.mean())
            elif baseline_type == 'random':
                m_pred = torch.randn_like(m48) * m48.std() + m48.mean()
                z_pred = torch.randn_like(z48) * z48.std() + z48.mean()
            elif baseline_type == 'small_perturbation':
                noise_scale = 0.01
                m_pred = m0 + torch.randn_like(m0) * noise_scale
                z_pred = z0 + torch.randn_like(z0) * noise_scale

            loss_dict = compute_adaptive_loss(m_pred, m48, z_pred, z48)
            results['baselines_0_to_48'][baseline_type] = loss_dict

        # Compute incremental baselines
        for baseline_type in baseline_types:
            incremental_loss = compute_incremental_baseline_loss(protein_id, data_dir, baseline_type,
                                                                 reduced_cluster_size)
            results['baselines_incremental'][baseline_type] = {'total': incremental_loss}

        # Data statistics
        results['data_stats'] = {
            'm0_mean': m0.mean().item(),
            'm0_std': m0.std().item(),
            'm48_mean': m48.mean().item(),
            'm48_std': m48.std().item(),
            'z0_mean': z0.mean().item(),
            'z0_std': z0.std().item(),
            'z48_mean': z48.mean().item(),
            'z48_std': z48.std().item(),
        }

        # Clean up memory
        del m0, z0, m48, z48, m_pred, z_pred
        gc.collect()

        return results

    except Exception as e:
        print(f"Error analyzing {protein_id}: {e}")
        return None


def run_baselines_analysis(data_dir: str, splits_dir: str, num_val_proteins: int, reduced_cluster_size: int) -> Dict:
    """Mode 1: Run baseline analysis on validation data"""
    print(f"\n🔬 MODE 1: BASELINE ANALYSIS")
    print("=" * 50)

    # Get validation proteins
    val_proteins = get_available_proteins(data_dir, splits_dir, 'validation')
    print(f"📋 Found {len(val_proteins)} validation proteins")

    # Randomly sample proteins (reproducible)
    random.seed(42)
    selected_proteins = random.sample(val_proteins, min(num_val_proteins, len(val_proteins)))

    print(f"🔀 Randomly selected {len(selected_proteins)} proteins for analysis")
    print(f"🧬 Selected proteins:")

    # Print protein info
    for protein_id in selected_proteins:
        info = get_protein_info(protein_id, data_dir)
        if info:
            print(f"   {protein_id}: {info['num_residues']} residues, {info['num_sequences']} sequences")

    print(f"\n📊 Running baseline analysis...")

    all_results = []
    baseline_stats_0_to_48 = {
        'identity': [],
        'zero': [],
        'mean': [],
        'random': [],
        'small_perturbation': []
    }
    baseline_stats_incremental = {
        'identity': [],
        'zero': [],
        'mean': [],
        'random': [],
        'small_perturbation': []
    }

    for i, protein_id in enumerate(selected_proteins):
        print(f"  [{i + 1}/{len(selected_proteins)}] Analyzing {protein_id}... ", end='', flush=True)

        result = analyze_baseline_losses(protein_id, data_dir, reduced_cluster_size)

        if result:
            all_results.append(result)

            # Collect statistics for 0→48
            for baseline_name, baseline_data in result['baselines_0_to_48'].items():
                if baseline_name in baseline_stats_0_to_48:
                    baseline_stats_0_to_48[baseline_name].append(baseline_data['total'])

            # Collect statistics for incremental
            for baseline_name, baseline_data in result['baselines_incremental'].items():
                if baseline_name in baseline_stats_incremental:
                    baseline_stats_incremental[baseline_name].append(baseline_data['total'])

            print("✅")
        else:
            print("❌")

    # Calculate summary statistics
    summary = {
        'num_proteins': len(all_results),
        'proteins_analyzed': [r['protein_id'] for r in all_results],
        'config': {
            'reduced_cluster_size': reduced_cluster_size,
            'num_val_proteins_requested': num_val_proteins,
            'total_val_proteins_available': len(val_proteins)
        },
        'baseline_summary_0_to_48': {},
        'baseline_summary_incremental': {}
    }

    # Summarize 0→48 baselines
    for baseline_name, losses in baseline_stats_0_to_48.items():
        if losses:
            summary['baseline_summary_0_to_48'][baseline_name] = {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'min': np.min(losses),
                'max': np.max(losses),
                'median': np.median(losses),
                'count': len(losses)
            }

    # Summarize incremental baselines
    for baseline_name, losses in baseline_stats_incremental.items():
        if losses:
            summary['baseline_summary_incremental'][baseline_name] = {
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


def filter_proteins_by_size(proteins: List[str], data_dir: str, max_residues: int) -> List[str]:
    """Filter proteins by residue count"""
    valid_proteins = []

    for protein_id in proteins:
        try:
            info = get_protein_info(protein_id, data_dir)
            if info and info['num_residues'] <= max_residues:
                valid_proteins.append(protein_id)
        except:
            continue

    return valid_proteins


def run_hyperparameter_search(data_dir: str, splits_dir: str, num_train_proteins: int, num_val_proteins: int,
                              epochs: int, loss_mode: str = 'incremental') -> Dict:
    """Mode 2: Test hyperparameter configurations with training"""

    print(f"\n🚀 MODE 2: HYPERPARAMETER SEARCH")
    print("=" * 50)

    # Get all available train and validation proteins
    all_train_proteins = get_available_proteins(data_dir, splits_dir, 'training')
    all_val_proteins = get_available_proteins(data_dir, splits_dir, 'validation')

    print(f"📋 Found {len(all_train_proteins)} total training proteins")
    print(f"📋 Found {len(all_val_proteins)} total validation proteins")
    print(
        f"🎯 Loss mode for training: {loss_mode} ({'0→48 loss' if loss_mode == 'end_to_end' else 'incremental loss'} guides learning)")
    print(f"📊 Dual reporting: Both incremental and 0→48 losses will be computed and reported")


    # Default base (core) values
    base_config = {
        'max_residues': 200,
        'batch_size': 10,
        'reduced_cluster_size': 64,
        'use_fast_ode': True,  # varies in defaults
        'hidden_dim': 64,
        'learning_rate': 1e-3,
        'loss_mode': 'incremental',  # varies in defaults
        'integrator': 'rk4'
    }

    # Values that generate the 4 default configs
    use_fast_ode_options = [True, False]
    loss_mode_options = ['incremental', 'end_to_end']

    # Exploratory variations (one change per config)
    exploratory_variants = {
        'max_residues': [150],
        'batch_size': [5],
        'reduced_cluster_size': [32, 128],
        'hidden_dim': [32, 128],
        'integrator': ['dopri5']
    }

    # 1. Generate the 4 default configurations
    default_configs = []
    for use_fast_ode, loss_mode in product(use_fast_ode_options, loss_mode_options):
        cfg = base_config.copy()
        cfg['use_fast_ode'] = use_fast_ode
        cfg['loss_mode'] = loss_mode
        default_configs.append(cfg)

    # 2. Generate 7 exploratory configs for each default
    exploratory_configs = []

    for default in default_configs:
        for field, variant_values in exploratory_variants.items():
            for val in variant_values:
                new_cfg = default.copy()
                new_cfg[field] = val
                exploratory_configs.append(new_cfg)

    # Combine
    hyperparameter_configs = default_configs + exploratory_configs
    # Ensure all configs are unique

    print(f"\n🧪 Generated {len(hyperparameter_configs)} hyperparameter configurations:")
    print(f"• {len(default_configs)}  default configs")
    print(f"• {len(exploratory_configs)} exploratory configs")

    # Create temp directory for splits
    temp_dir = Path("temp_splits")
    temp_dir.mkdir(exist_ok=True)

    results = {
        'search_date': datetime.now().isoformat(),
        'config': {
            'num_train_proteins_requested': num_train_proteins,
            'num_val_proteins_requested': num_val_proteins,
            'epochs': epochs,
            'total_configs': len(hyperparameter_configs)
        },
        'successful_configs': [],
        'failed_configs': []
    }

    for i, config in enumerate(hyperparameter_configs):
        print(f"\n[{i + 1}/{len(hyperparameter_configs)}] Testing config:")
        print(f"   max_residues={config['max_residues']}, batch_size={config['batch_size']}")
        print(f"   cluster_size={config['reduced_cluster_size']}, fast_ode={config['use_fast_ode']}")
        print(f"   hidden_dim={config['hidden_dim']}, lr={config['learning_rate']}, loss_mode={config['loss_mode']}")

        # Filter proteins by max_residues for this config
        valid_train_proteins = filter_proteins_by_size(all_train_proteins, data_dir, config['max_residues'])
        valid_val_proteins = filter_proteins_by_size(all_val_proteins, data_dir, config['max_residues'])

        print(
            f"   📏 Proteins ≤ {config['max_residues']} residues: {len(valid_train_proteins)} train, {len(valid_val_proteins)} val")

        # Check if we have enough proteins
        if len(valid_train_proteins) < num_train_proteins or len(valid_val_proteins) < num_val_proteins:
            print(f"   ❌ Insufficient proteins (need {num_train_proteins} train, {num_val_proteins} val)")
            config_result = {
                'config': config,
                'success': False,
                'error': f'Insufficient proteins: {len(valid_train_proteins)} train, {len(valid_val_proteins)} val available',
                'is_oom': False,
                'experiment_name': None
            }
            results['failed_configs'].append(config_result)
            continue

        # Randomly sample proteins for this config (reproducible)
        random.seed(42)
        selected_train = random.sample(valid_train_proteins, num_train_proteins)
        selected_val = random.sample(valid_val_proteins, num_val_proteins)

        print(f"   🔀 Selected {len(selected_train)} training proteins:")
        for protein_id in selected_train:
            info = get_protein_info(protein_id, data_dir)
            if info:
                print(f"      {protein_id}: {info['num_residues']} residues")

        print(f"   🔀 Selected {len(selected_val)} validation proteins:")
        for protein_id in selected_val:
            info = get_protein_info(protein_id, data_dir)
            if info:
                print(f"      {protein_id}: {info['num_residues']} residues")

        # Create temporary splits files for this config
        temp_train_file = temp_dir / f"training_chains_{i}.txt"
        temp_val_file = temp_dir / f"validation_chains_{i}.txt"

        with open(temp_train_file, 'w') as f:
            for protein in selected_train:
                f.write(f"{protein}\n")

        with open(temp_val_file, 'w') as f:
            for protein in selected_val:
                f.write(f"{protein}\n")

        # Create temporary experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"hyperparam_test_{i}_{timestamp}"

        # Build command for train_evoformer_ode.py
        cmd = [
            sys.executable,
            "train_evoformer_ode.py",
            "--data_dir", data_dir,
            "--splits_dir", str(temp_dir),
            "--mode", "training",
            "--epochs", str(epochs),
            "--output_dir", "temp_outputs",
            "--experiment_name", experiment_name,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--learning_rate", str(config['learning_rate']),
            "--reduced_cluster_size", str(config['reduced_cluster_size']),
            "--hidden_dim", str(config['hidden_dim']),
            "--batch_size", str(config['batch_size']),
            "--max_residues", str(config['max_residues']),
            "--loss_mode", config['loss_mode'],  # Add loss mode parameter
            "--integrator", str(config['integrator']),
            "--lr_patience", "2",
            "--early_stopping_patience", "7"
        ]

        # Override splits file names for this config
        cmd.extend(["--splits_dir", str(temp_dir)])

        if config['use_fast_ode']:
            cmd.append("--use_fast_ode")

        if torch.cuda.is_available():
            cmd.append("--use_amp")

        try:
            print(f"   🏋️ Running training...")

            # Temporarily rename split files to standard names
            standard_train_file = temp_dir / "training_chains.txt"
            standard_val_file = temp_dir / "validation_chains.txt"

            # Move config-specific files to standard names
            temp_train_file.rename(standard_train_file)
            temp_val_file.rename(standard_val_file)

            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=os.getcwd()
            )

            if result.returncode == 0:
                print(f"   ✅ Training completed successfully")

                # Try to extract both final validation losses from output
                final_val_loss_primary = None
                final_val_loss_0_to_48 = None
                final_val_loss_incremental = None

                output_lines = result.stdout.split('\n')
                for line in reversed(output_lines):
                    # Look for validation summary with both losses
                    if 'Primary Loss' in line and 'Validation Summary' in output_lines[max(0, output_lines.index(
                            line) - 5):output_lines.index(line) + 1]:
                        try:
                            # Extract primary loss (used for LR scheduling)
                            if 'Primary Loss' in line:
                                final_val_loss_primary = float(line.split(':')[1].strip().split()[0])
                        except:
                            continue
                    elif 'Loss 0→48:' in line:
                        try:
                            final_val_loss_0_to_48 = float(line.split(':')[1].strip().split()[0])
                        except:
                            continue
                    elif 'Loss Incremental:' in line:
                        try:
                            final_val_loss_incremental = float(line.split(':')[1].strip().split()[0])
                        except:
                            continue

                    # Break if we found all three losses
                    if all(x is not None for x in
                           [final_val_loss_primary, final_val_loss_0_to_48, final_val_loss_incremental]):
                        break

                config_result = {
                    'config': config,
                    'success': True,
                    'final_validation_loss_primary': final_val_loss_primary,
                    'final_validation_loss_0_to_48': final_val_loss_0_to_48,
                    'final_validation_loss_incremental': final_val_loss_incremental,
                    'experiment_name': experiment_name,
                    'selected_train_proteins': selected_train,
                    'selected_val_proteins': selected_val
                }
                results['successful_configs'].append(config_result)

                print(f"   📊 Final validation losses:")
                if final_val_loss_primary is not None:
                    print(f"      Primary ({config['loss_mode']}): {final_val_loss_primary:.4f}")
                if final_val_loss_0_to_48 is not None:
                    print(f"      0→48: {final_val_loss_0_to_48:.4f}")
                if final_val_loss_incremental is not None:
                    print(f"      Incremental: {final_val_loss_incremental:.4f}")

                if final_val_loss_primary is None:
                    print(f"   ⚠️  Could not extract primary validation loss")

            else:
                print(f"   ❌ Training failed")
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"

                # Check if it's OOM
                is_oom = "CUDA out of memory" in error_msg or "out of memory" in error_msg

                config_result = {
                    'config': config,
                    'success': False,
                    'error': error_msg,
                    'is_oom': is_oom,
                    'experiment_name': experiment_name,
                    'selected_train_proteins': selected_train,
                    'selected_val_proteins': selected_val
                }
                results['failed_configs'].append(config_result)

                if is_oom:
                    print(f"   💀 OUT OF MEMORY")
                else:
                    print(f"   💥 Error: {error_msg[:100]}...")

        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout after 1 hour")
            config_result = {
                'config': config,
                'success': False,
                'error': 'Timeout after 1 hour',
                'is_oom': False,
                'experiment_name': experiment_name,
                'selected_train_proteins': selected_train,
                'selected_val_proteins': selected_val
            }
            results['failed_configs'].append(config_result)

        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
            config_result = {
                'config': config,
                'success': False,
                'error': str(e),
                'is_oom': False,
                'experiment_name': experiment_name,
                'selected_train_proteins': selected_train,
                'selected_val_proteins': selected_val
            }
            results['failed_configs'].append(config_result)

        finally:
            # Clean up temp files for this config
            for f in [standard_train_file, standard_val_file, temp_train_file, temp_val_file]:
                f.unlink(missing_ok=True)

    # Clean up temp directory
    temp_dir.rmdir()

    return results


def print_baseline_results(baseline_results: Dict):
    """Print baseline analysis results"""
    print(f"\n📊 BASELINE ANALYSIS RESULTS")
    print("=" * 50)

    summary = baseline_results['summary']

    print(f"Proteins analyzed: {summary['num_proteins']}")
    print(f"Configuration: cluster_size={summary['config']['reduced_cluster_size']}")

    print(f"\n📈 BLOCK 0 → 48 BASELINES:")
    print(f"{'Baseline':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)

    baseline_order = ['identity', 'small_perturbation', 'mean', 'zero', 'random']
    for baseline_name in baseline_order:
        if baseline_name in summary['baseline_summary_0_to_48']:
            stats = summary['baseline_summary_0_to_48'][baseline_name]
            print(
                f"{baseline_name:<20} {stats['mean']:<8.3f} {stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f}")

    print(f"\n📈 INCREMENTAL BASELINES (like current training):")
    print(f"{'Baseline':<20} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)

    for baseline_name in baseline_order:
        if baseline_name in summary['baseline_summary_incremental']:
            stats = summary['baseline_summary_incremental'][baseline_name]
            print(
                f"{baseline_name:<20} {stats['mean']:<8.3f} {stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f}")

    # Loss interpretation
    identity_0_48 = summary['baseline_summary_0_to_48'].get('identity', {}).get('mean', float('inf'))
    identity_incr = summary['baseline_summary_incremental'].get('identity', {}).get('mean', float('inf'))

    print(f"\n💡 TRAINING LOSS INTERPRETATION:")
    print(f"🎯 Block 0→48 Loss Targets:")
    print(f"   • Excellent:  < {identity_0_48 * 0.5:.3f} (much better than identity)")
    print(f"   • Good:       < {identity_0_48 * 0.8:.3f} (better than identity)")
    print(f"   • Acceptable: < {identity_0_48 * 1.2:.3f} (close to identity)")

    print(f"🎯 Incremental Loss Targets (current training approach):")
    print(f"   • Excellent:  < {identity_incr * 0.5:.3f} (much better than identity)")
    print(f"   • Good:       < {identity_incr * 0.8:.3f} (better than identity)")
    print(f"   • Acceptable: < {identity_incr * 1.2:.3f} (close to identity)")


def print_hyperparameter_results(hyperparam_results: Dict):
    """Print hyperparameter search results"""
    print(f"\n🚀 HYPERPARAMETER SEARCH RESULTS")
    print("=" * 50)

    config = hyperparam_results['config']
    successful = hyperparam_results['successful_configs']
    failed = hyperparam_results['failed_configs']

    print(f"Training proteins: {config['num_train_proteins_requested']}")
    print(f"Validation proteins: {config['num_val_proteins_requested']}")
    print(f"Epochs per test: {config['epochs']}")
    print(f"Total configs tested: {config['total_configs']}")

    print(f"\n📊 Results Summary:")
    print(f"✅ Successful configs: {len(successful)}")
    print(f"❌ Failed configs: {len(failed)}")
    print(f"📈 Success rate: {len(successful) / (len(successful) + len(failed)) * 100:.1f}%")

    if successful:
        print(f"\n✅ SUCCESSFUL CONFIGURATIONS:")
        # Sort by primary validation loss if available
        successful_sorted = sorted(successful, key=lambda x: x.get('final_validation_loss_primary', float('inf')))

        for i, result in enumerate(successful_sorted, 1):
            config = result['config']
            primary_loss = result.get('final_validation_loss_primary', 'N/A')
            loss_0_to_48 = result.get('final_validation_loss_0_to_48', 'N/A')
            loss_incremental = result.get('final_validation_loss_incremental', 'N/A')
            ode_type = "Fast ODE" if config['use_fast_ode'] else "Full ODE"

            print(f"  {i}. max_res={config['max_residues']}, batch={config['batch_size']}, "
                  f"cluster={config['reduced_cluster_size']}, {ode_type}")
            print(
                f"     lr={config['learning_rate']}, hidden_dim={config['hidden_dim']}, loss_mode={config['loss_mode']}")

            print(f"     📊 Final validation losses:")
            if primary_loss != 'N/A':
                print(f"        Primary ({config['loss_mode']}): {primary_loss:.4f}")
            if loss_0_to_48 != 'N/A':
                print(f"        0→48: {loss_0_to_48:.4f}")
            if loss_incremental != 'N/A':
                print(f"        Incremental: {loss_incremental:.4f}")

            print(f"     🧪 Experiment: {result['experiment_name']}")

    if failed:
        print(f"\n❌ FAILED CONFIGURATIONS:")
        oom_count = sum(1 for r in failed if r.get('is_oom', False))
        print(f"   💀 Out of memory failures: {oom_count}")
        print(f"   💥 Other failures: {len(failed) - oom_count}")

        for i, result in enumerate(failed, 1):
            config = result['config']
            ode_type = "Fast ODE" if config['use_fast_ode'] else "Full ODE"

            print(f"  {i}. max_res={config['max_residues']}, batch={config['batch_size']}, "
                  f"cluster={config['reduced_cluster_size']}, {ode_type}, loss_mode={config['loss_mode']}")

            if result.get('is_oom', False):
                print(f"     💀 OUT OF MEMORY")
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"     💥 Error: {error_msg[:50]}...")

    # Recommendations
    if successful:
        # Find best config for each loss type
        best_primary = min(successful, key=lambda x: x.get('final_validation_loss_primary', float('inf')))
        best_0_to_48 = min(successful, key=lambda x: x.get('final_validation_loss_0_to_48', float('inf')))
        best_incremental = min(successful, key=lambda x: x.get('final_validation_loss_incremental', float('inf')))

        print(f"\n🏆 BEST CONFIGURATIONS BY LOSS TYPE:")

        if best_primary.get('final_validation_loss_primary') is not None:
            config = best_primary['config']
            print(f"   Primary Loss ({config['loss_mode']}):")
            print(f"     max_residues={config['max_residues']}, batch_size={config['batch_size']}")
            print(f"     cluster_size={config['reduced_cluster_size']}, use_fast_ode={config['use_fast_ode']}")
            print(f"     hidden_dim={config['hidden_dim']}, learning_rate={config['learning_rate']}")
            print(f"     📊 Primary loss: {best_primary['final_validation_loss_primary']:.4f}")

        if best_0_to_48.get('final_validation_loss_0_to_48') is not None:
            config = best_0_to_48['config']
            print(f"   0→48 Loss:")
            print(f"     max_residues={config['max_residues']}, batch_size={config['batch_size']}")
            print(f"     cluster_size={config['reduced_cluster_size']}, use_fast_ode={config['use_fast_ode']}")
            print(f"     hidden_dim={config['hidden_dim']}, learning_rate={config['learning_rate']}")
            print(f"     📊 0→48 loss: {best_0_to_48['final_validation_loss_0_to_48']:.4f}")

        if best_incremental.get('final_validation_loss_incremental') is not None:
            config = best_incremental['config']
            print(f"   Incremental Loss:")
            print(f"     max_residues={config['max_residues']}, batch_size={config['batch_size']}")
            print(f"     cluster_size={config['reduced_cluster_size']}, use_fast_ode={config['use_fast_ode']}")
            print(f"     hidden_dim={config['hidden_dim']}, learning_rate={config['learning_rate']}")
            print(f"     📊 Incremental loss: {best_incremental['final_validation_loss_incremental']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Loss Analysis for Evoformer Neural ODE')

    # Core arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing protein evoformer blocks')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory containing data splits')
    parser.add_argument('--mode', type=str, choices=['baselines_only', 'hyperparameter_search'],
                        default='baselines_only',
                        help='Analysis mode')

    # Mode 1 arguments (baselines_only)
    parser.add_argument('--num_val_proteins', type=int, default=10,
                        help='Number of validation proteins to analyze (mode 1)')
    parser.add_argument('--reduced_cluster_size', type=int, default=64,
                        help='Maximum cluster size (mode 1)')

    # Mode 2 arguments (hyperparameter_search)
    parser.add_argument('--num_train_proteins', type=int, default=20,
                        help='Number of training proteins to use (mode 2)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs for hyperparameter testing (mode 2)')

    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.mode == 'hyperparameter_search':
            args.output_file = f"hyperparameter_search_{timestamp}.json"
        else:
            args.output_file = f"baseline_analysis_{timestamp}.json"

    print("🔬 EVOFORMER NEURAL ODE LOSS ANALYZER")
    print("=" * 50)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📂 Splits directory: {args.splits_dir}")
    print(f"🎯 Mode: {args.mode}")

    if args.mode == 'baselines_only':
        print(f"🔢 Validation proteins: {args.num_val_proteins}")
        print(f"📏 Reduced cluster size: {args.reduced_cluster_size}")

        # Run baseline analysis
        baseline_results = run_baselines_analysis(
            args.data_dir,
            args.splits_dir,
            args.num_val_proteins,
            args.reduced_cluster_size
        )

        # Print results
        print_baseline_results(baseline_results)

        # Save results
        output_data = {
            'mode': 'baselines_only',
            'results': baseline_results,
            'metadata': {
                'script_version': '3.0',
                'analysis_date': datetime.now().isoformat(),
                'description': 'Evoformer Neural ODE Baseline Analysis - Block 0→48 and Incremental'
            }
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n📄 Results saved to: {args.output_file}")

    elif args.mode == 'hyperparameter_search':
        print(f"🔢 Training proteins: {args.num_train_proteins}")
        print(f"🔢 Validation proteins: {args.num_val_proteins}")
        print(f"📈 Epochs per test: {args.epochs}")

        # Run hyperparameter search
        hyperparam_results = run_hyperparameter_search(
            args.data_dir,
            args.splits_dir,
            args.num_train_proteins,
            args.num_val_proteins,
            args.epochs
        )

        # Print results
        print_hyperparameter_results(hyperparam_results)

        # Save results
        output_data = {
            'mode': 'hyperparameter_search',
            'results': hyperparam_results,
            'metadata': {
                'script_version': '3.0',
                'analysis_date': datetime.now().isoformat(),
                'description': 'Evoformer Neural ODE Hyperparameter Search with Training'
            }
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n📄 Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()