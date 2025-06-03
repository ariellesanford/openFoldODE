#!/usr/bin/env python3
"""
Enhanced Loss Analysis Script for Evoformer Neural ODE Training
Now includes hyperparameter search for memory-critical parameters
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
from datetime import datetime
import gc
import traceback
from torchdiffeq import odeint
import torch.optim as optim


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

    if mode not in split_files:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {list(split_files.keys())}")

    split_file = os.path.join(splits_dir, split_files[mode])

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    proteins = []
    with open(split_file, 'r') as f:
        for line in f:
            protein_id = line.strip()
            if protein_id:  # Skip empty lines
                proteins.append(protein_id)

    return proteins


def get_available_proteins(data_dir: str, splits_dir: str, mode: str) -> List[str]:
    """Get list of available protein IDs for the specified mode"""
    # Load proteins from split file
    split_proteins = load_split_proteins(splits_dir, mode)

    # Check which ones actually exist in data_dir
    available_proteins = []
    for protein_id in split_proteins:
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
        if os.path.isdir(protein_dir):
            available_proteins.append(protein_id)

    return available_proteins


def filter_proteins_by_size(proteins: List[str], data_dir: str, max_residues: int = None) -> Tuple[
    List[str], List[str]]:
    """Filter proteins by residue count, return (valid_proteins, oversized_proteins)"""
    if max_residues is None:
        return proteins, []

    valid_proteins = []
    oversized_proteins = []

    for protein_id in proteins:
        try:
            protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
            m_path = os.path.join(protein_dir, "m_block_0.pt")

            if os.path.exists(m_path):
                m_test = torch.load(m_path, map_location='cpu')
                num_residues = m_test.shape[-2] if m_test.dim() == 4 else m_test.shape[-2]
                del m_test  # Free memory immediately

                if num_residues <= max_residues:
                    valid_proteins.append(protein_id)
                else:
                    oversized_proteins.append(protein_id)
            else:
                oversized_proteins.append(protein_id)

        except Exception as e:
            oversized_proteins.append(protein_id)

    return valid_proteins, oversized_proteins


def analyze_baseline_losses(protein_id: str, data_dir: str, reduced_cluster_size: int = 64,
                            max_residues: int = None) -> Dict:
    """Analyze different baseline prediction strategies for a protein"""
    try:
        # Load consecutive blocks
        m0, z0 = load_protein_block(protein_id, 0, data_dir, reduced_cluster_size)
        m1, z1 = load_protein_block(protein_id, 1, data_dir, reduced_cluster_size)

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
            'num_residues': m0.shape[-2],
            'msa_channels': m0.shape[2],
            'pair_channels': z0.shape[2],
            'available_blocks': len(available_blocks),
            'reduced_cluster_size': reduced_cluster_size,
            'max_residues': max_residues,
            'baselines': {}
        }

        # 1. Identity baseline (predict block 0 as block 1)
        identity_loss = compute_adaptive_loss(m0, m1, z0, z1)
        results['baselines']['identity'] = identity_loss

        # 2. Zero prediction baseline
        zero_m = torch.zeros_like(m1)
        zero_z = torch.zeros_like(z1)
        zero_loss = compute_adaptive_loss(zero_m, m1, zero_z, z1)
        results['baselines']['zero'] = zero_loss

        # 3. Mean prediction baseline
        mean_m = torch.full_like(m1, m1.mean())
        mean_z = torch.full_like(z1, z1.mean())
        mean_loss = compute_adaptive_loss(mean_m, m1, mean_z, z1)
        results['baselines']['mean'] = mean_loss

        # 4. Random prediction baseline (Gaussian noise)
        random_m = torch.randn_like(m1) * m1.std() + m1.mean()
        random_z = torch.randn_like(z1) * z1.std() + z1.mean()
        random_loss = compute_adaptive_loss(random_m, m1, random_z, z1)
        results['baselines']['random'] = random_loss

        # 5. Small perturbation baseline (identity + small noise)
        noise_scale = 0.01
        perturb_m = m0 + torch.randn_like(m0) * noise_scale
        perturb_z = z0 + torch.randn_like(z0) * noise_scale
        perturb_loss = compute_adaptive_loss(perturb_m, m1, perturb_z, z1)
        results['baselines']['small_perturbation'] = perturb_loss

        # 6. Linear interpolation baseline (if enough blocks available)
        if len(available_blocks) >= 3:
            m2, z2 = load_protein_block(protein_id, 2, data_dir, reduced_cluster_size)
            interp_m = 0.5 * (m0 + m2)  # Interpolate between 0 and 2 to predict 1
            interp_z = 0.5 * (z0 + z2)
            interp_loss = compute_adaptive_loss(interp_m, m1, interp_z, z1)
            results['baselines']['interpolation'] = interp_loss

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

        # Clean up memory
        del m0, z0, m1, z1, zero_m, zero_z, mean_m, mean_z, random_m, random_z, perturb_m, perturb_z
        if len(available_blocks) >= 3:
            del m2, z2, interp_m, interp_z
        gc.collect()

        return results

    except Exception as e:
        return None


def analyze_multiple_proteins(data_dir: str, proteins: List[str], reduced_cluster_size: int = 64,
                              max_residues: int = None) -> Dict:
    """Analyze baseline losses across multiple proteins"""
    all_results = []
    baseline_stats = {
        'identity': [],
        'zero': [],
        'mean': [],
        'random': [],
        'small_perturbation': [],
        'interpolation': []
    }

    for protein_id in proteins:
        result = analyze_baseline_losses(protein_id, data_dir, reduced_cluster_size, max_residues)

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
        'config': {
            'reduced_cluster_size': reduced_cluster_size,
            'max_residues': max_residues,
            'total_proteins_provided': len(proteins)
        },
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


def test_memory_feasibility(data_dir: str, memory_config: Dict, test_proteins: List[str]) -> Dict:
    """Simple data loading test - kept for compatibility but simplified"""
    max_residues = memory_config.get('max_residues')
    reduced_cluster_size = memory_config.get('reduced_cluster_size', 64)

    if not test_proteins:
        return {
            'feasible': False,
            'error': 'No test proteins available',
            'proteins_tested': 0,
            'max_memory_mb': 0
        }

    try:
        # Just test if we can load one protein's data
        for protein_id in test_proteins[:1]:
            m0, z0 = load_protein_block(protein_id, 0, data_dir, reduced_cluster_size)
            m1, z1 = load_protein_block(protein_id, 1, data_dir, reduced_cluster_size)

            # Quick size check
            if max_residues and m0.shape[-2] > max_residues:
                continue

            # Clean up immediately
            del m0, z0, m1, z1

            return {
                'feasible': True,
                'error': None,
                'proteins_tested': 1,
                'max_memory_mb': 0  # Not tracking for simple test
            }

        return {
            'feasible': False,
            'error': 'No suitable proteins found',
            'proteins_tested': 0,
            'max_memory_mb': 0
        }

    except Exception as e:
        return {
            'feasible': False,
            'error': str(e),
            'proteins_tested': 0,
            'max_memory_mb': 0
        }


def test_single_protein_training_memory(data_dir: str, memory_config: Dict, target_residue_range: Tuple[int, int],
                                        available_proteins: List[str]) -> Dict:
    """Test training memory on a single protein within a specific residue range"""

    min_residues, max_residues = target_residue_range
    reduced_cluster_size = memory_config.get('reduced_cluster_size', 64)
    batch_size = memory_config.get('batch_size', 10)
    use_fast_ode = memory_config.get('use_fast_ode', True)
    hidden_dim = memory_config.get('hidden_dim', 64)
    integrator = memory_config.get('integrator', 'rk4')

    print(f"    🎯 Looking for protein with {min_residues}-{max_residues} residues...")

    # Find a protein in the target residue range
    target_protein = None
    actual_residues = 0

    for protein_id in available_proteins:
        try:
            protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
            m_path = os.path.join(protein_dir, "m_block_0.pt")

            if os.path.exists(m_path):
                m_test = torch.load(m_path, map_location='cpu')
                num_residues = m_test.shape[-2] if m_test.dim() == 4 else m_test.shape[-2]
                del m_test

                if min_residues <= num_residues <= max_residues:
                    target_protein = protein_id
                    actual_residues = num_residues
                    break
        except:
            continue

    if not target_protein:
        return {
            'feasible': False,
            'error': f'No protein found with {min_residues}-{max_residues} residues',
            'protein_tested': None,
            'protein_residues': 0,
            'max_memory_mb': 0,
            'config_tested': memory_config
        }

    print(f"    🧬 Testing: {target_protein} ({actual_residues} residues)")
    print(
        f"    ⚙️  Config: batch_size={batch_size}, cluster_size={reduced_cluster_size}, {'Fast' if use_fast_ode else 'Full'} ODE, hidden_dim={hidden_dim}")

    try:
        # Import ODE functions
        try:
            from evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast
        except ImportError:
            return {
                'feasible': False,
                'error': 'Could not import evoformer_ode module',
                'protein_tested': target_protein,
                'protein_residues': actual_residues,
                'max_memory_mb': 0,
                'config_tested': memory_config
            }

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        max_memory_mb = 0

        # Initialize model
        c_m = 256  # MSA embedding dimension
        c_z = 128  # Pair embedding dimension

        if use_fast_ode:
            ode_func = EvoformerODEFuncFast(c_m, c_z, hidden_dim).to(device)
        else:
            ode_func = EvoformerODEFunc(c_m, c_z, hidden_dim).to(device)

        optimizer = optim.Adam(ode_func.parameters(), lr=1e-3)

        if device == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = max(max_memory_mb, current_memory)
            print(f"    📊 Model loaded: {current_memory:.0f} MB")

        # Load protein data
        m0, z0 = load_protein_block(target_protein, 0, data_dir, reduced_cluster_size)
        m1, z1 = load_protein_block(target_protein, 1, data_dir, reduced_cluster_size)

        print(f"    📦 Data shapes: MSA {m0.shape}, Pair {z0.shape}")

        m0, z0, m1, z1 = m0.to(device), z0.to(device), m1.to(device), z1.to(device)

        if device == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = max(max_memory_mb, current_memory)
            print(f"    📊 Data loaded: {current_memory:.0f} MB")

        # Run 1 epoch simulation (simplified)
        print(f"    🏋️  Running training simulation...")

        optimizer.zero_grad()

        # Create time grid for batch processing
        t_grid = torch.linspace(0.0, 1.0, batch_size).to(device)

        print(f"    ⏰ Time grid: {batch_size} steps")

        # Forward pass through ODE
        trajectory = odeint(
            ode_func,
            (m0, z0),
            t_grid,
            method=integrator,
            rtol=1e-4,
            atol=1e-5
        )

        if device == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = max(max_memory_mb, current_memory)
            print(f"    📊 Forward pass: {current_memory:.0f} MB")

        # Compute loss
        m_pred = trajectory[0][-1]  # Final MSA state
        z_pred = trajectory[1][-1]  # Final pair state

        loss = F.mse_loss(m_pred, m1) + F.mse_loss(z_pred, z1)

        if device == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = max(max_memory_mb, current_memory)
            print(f"    📊 Loss computed: {current_memory:.0f} MB")

        # Backward pass
        loss.backward()

        if device == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = max(max_memory_mb, current_memory)
            print(f"    📊 Backward pass: {current_memory:.0f} MB")

        # Optimizer step
        optimizer.step()

        if device == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory_mb = max(max_memory_mb, current_memory)
            print(f"    📊 Optimizer step: {current_memory:.0f} MB")

        print(f"    ✅ Training simulation successful!")
        print(f"    💾 Peak memory usage: {max_memory_mb:.0f} MB")

        # Clean up
        del ode_func, optimizer, m0, z0, m1, z1, trajectory, m_pred, z_pred, loss
        if device == 'cuda':
            torch.cuda.empty_cache()

        return {
            'feasible': True,
            'error': None,
            'protein_tested': target_protein,
            'protein_residues': actual_residues,
            'max_memory_mb': max_memory_mb,
            'config_tested': memory_config
        }

    except Exception as e:
        error_msg = str(e)

        # Clean up on error
        try:
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

        # Check if it's OOM
        is_oom = "CUDA out of memory" in error_msg or "out of memory" in error_msg

        print(f"    ❌ Training failed: {error_msg}")
        if is_oom:
            print(f"    💀 OUT OF MEMORY at {max_memory_mb:.0f} MB")

        return {
            'feasible': False,
            'error': error_msg,
            'protein_tested': target_protein,
            'protein_residues': actual_residues,
            'max_memory_mb': max_memory_mb,
            'config_tested': memory_config,
            'is_oom': is_oom
        }

    finally:
        # Ensure cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def hyperparameter_search_memory_critical(data_dir: str, splits_dir: str, max_proteins: int = 5) -> Dict:
    """Test memory-critical hyperparameter combinations"""
    print(f"\n🚀 HYPERPARAMETER SEARCH - MEMORY CRITICAL PARAMETERS")
    print("=" * 60)

    # Get validation proteins and filter once
    print(f"📋 Loading validation proteins from splits...")
    try:
        val_proteins = get_available_proteins(data_dir, splits_dir, 'validation')
        print(f"🔍 Found {len(val_proteins)} validation proteins")
    except Exception as e:
        print(f"❌ Could not load validation split: {e}")
        print(f"📁 Falling back to scanning data directory...")
        # Fallback: scan data directory
        all_proteins = []
        for item in Path(data_dir).iterdir():
            if item.is_dir() and item.name.endswith('_evoformer_blocks'):
                protein_id = item.name.replace('_evoformer_blocks', '')
                all_proteins.append(protein_id)
        val_proteins = sorted(all_proteins)
        print(f"🔍 Found {len(val_proteins)} total proteins")

    # Shuffle proteins to avoid alphabetical bias
    import random
    random.shuffle(val_proteins)
    print(f"🔀 Shuffled protein order")

    # Define memory-critical parameter combinations to test
    memory_configs = [
        # Add default hidden_dim and integrator for training test
        {'max_residues': 500, 'batch_size': 5, 'reduced_cluster_size': 128, 'use_fast_ode': True, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 400, 'batch_size': 8, 'reduced_cluster_size': 128, 'use_fast_ode': True, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 300, 'batch_size': 10, 'reduced_cluster_size': 128, 'use_fast_ode': True, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 300, 'batch_size': 8, 'reduced_cluster_size': 128, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},  # Full ODE
        {'max_residues': 250, 'batch_size': 10, 'reduced_cluster_size': 128, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 200, 'batch_size': 15, 'reduced_cluster_size': 128, 'use_fast_ode': True, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 200, 'batch_size': 10, 'reduced_cluster_size': 128, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 150, 'batch_size': 20, 'reduced_cluster_size': 128, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},

        # Medium memory configurations
        {'max_residues': 400, 'batch_size': 10, 'reduced_cluster_size': 64, 'use_fast_ode': True, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 300, 'batch_size': 15, 'reduced_cluster_size': 64, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 200, 'batch_size': 20, 'reduced_cluster_size': 64, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},

        # Conservative configurations (should always work)
        {'max_residues': 150, 'batch_size': 10, 'reduced_cluster_size': 64, 'use_fast_ode': True, 'hidden_dim': 64,
         'integrator': 'rk4'},
        {'max_residues': 100, 'batch_size': 20, 'reduced_cluster_size': 64, 'use_fast_ode': False, 'hidden_dim': 64,
         'integrator': 'rk4'},
    ]

    results = {
        'search_date': datetime.now().isoformat(),
        'total_configs_tested': len(memory_configs),
        'feasible_configs': [],
        'infeasible_configs': [],
        'baseline_analyses': {},
        'summary': {}
    }

    print(f"🧪 Testing {len(memory_configs)} memory configurations...")

    for i, config in enumerate(memory_configs):
        print(f"\n[{i + 1}/{len(memory_configs)}] Testing config: {config}")

        # Filter proteins by size for this config
        if config['max_residues']:
            valid_proteins, oversized = filter_proteins_by_size(val_proteins, data_dir, config['max_residues'])
            print(f"  🔍 {len(valid_proteins)} proteins ≤ {config['max_residues']} residues (filtered {len(oversized)})")
        else:
            valid_proteins = val_proteins

        if not valid_proteins:
            print(f"  ❌ No valid proteins for this configuration")
            config_result = {
                'config': config,
                'feasibility': {'feasible': False, 'error': 'No valid proteins', 'proteins_tested': 0,
                                'max_memory_mb': 0},
                'baseline_analysis': None
            }
            results['infeasible_configs'].append(config_result)
            continue

        # Test memory feasibility on first few proteins
        feasibility = test_memory_feasibility(data_dir, config, valid_proteins)

        config_result = {
            'config': config,
            'feasibility': feasibility,
            'baseline_analysis': None
        }

        if feasibility['feasible']:
            print(f"  ✅ Memory test passed ({feasibility['proteins_tested']} proteins)")
            if feasibility['max_memory_mb'] > 0:
                print(f"  💾 Peak GPU memory: {feasibility['max_memory_mb']:.0f} MB")

            # Run baseline analysis for this configuration
            try:
                print(f"  📊 Running baseline analysis...")

                # Use subset of valid proteins for baseline analysis
                analysis_proteins = valid_proteins[:max_proteins]

                baseline_analysis = analyze_multiple_proteins(
                    data_dir=data_dir,
                    proteins=analysis_proteins,
                    reduced_cluster_size=config['reduced_cluster_size'],
                    max_residues=config['max_residues']
                )

                config_result['baseline_analysis'] = baseline_analysis
                results['feasible_configs'].append(config_result)

                # Store baseline analysis with config key for easy access
                config_key = f"res{config['max_residues']}_batch{config['batch_size']}_cluster{config['reduced_cluster_size']}_fast{config['use_fast_ode']}"
                results['baseline_analyses'][config_key] = baseline_analysis

                print(f"  ✅ Baseline analysis complete ({baseline_analysis['summary']['num_proteins']} proteins)")

            except Exception as e:
                print(f"  ❌ Baseline analysis failed: {str(e)}")
                config_result['baseline_analysis'] = {'error': str(e)}
                results['infeasible_configs'].append(config_result)

        else:
            print(f"  ❌ Memory test failed: {feasibility['error']}")
            results['infeasible_configs'].append(config_result)

    # Generate summary
    results['summary'] = {
        'feasible_count': len(results['feasible_configs']),
        'infeasible_count': len(results['infeasible_configs']),
        'success_rate': len(results['feasible_configs']) / len(memory_configs) * 100,
    }

    if results['feasible_configs']:
        # Find the most aggressive feasible config
        feasible_configs = [r['config'] for r in results['feasible_configs']]
        max_residues_feasible = max(c['max_residues'] for c in feasible_configs)
        max_batch_feasible = max(
            c['batch_size'] for c in feasible_configs if c['max_residues'] == max_residues_feasible)

        results['summary']['max_feasible_residues'] = max_residues_feasible
        results['summary']['max_feasible_batch_at_max_residues'] = max_batch_feasible

        # Separate by use_fast_ode
        fast_configs = [c for c in feasible_configs if c['use_fast_ode']]
        full_configs = [c for c in feasible_configs if not c['use_fast_ode']]

        results['summary']['fast_ode_configs'] = len(fast_configs)
        results['summary']['full_ode_configs'] = len(full_configs)

        if fast_configs:
            results['summary']['max_residues_fast_ode'] = max(c['max_residues'] for c in fast_configs)
        if full_configs:
            results['summary']['max_residues_full_ode'] = max(c['max_residues'] for c in full_configs)

    return results


def print_memory_search_summary(search_results: Dict):
    """Print a comprehensive summary of the memory search results"""
    print(f"\n" + "=" * 60)
    print(f"📊 MEMORY-CRITICAL HYPERPARAMETER SEARCH SUMMARY")
    print(f"=" * 60)

    summary = search_results['summary']

    print(f"\n🧪 Search Results:")
    print(f"  Total configurations tested: {search_results['total_configs_tested']}")
    print(f"  ✅ Feasible configurations: {summary['feasible_count']}")
    print(f"  ❌ Infeasible configurations: {summary['infeasible_count']}")
    print(f"  📈 Success rate: {summary['success_rate']:.1f}%")

    if summary['feasible_count'] > 0:
        print(f"\n🚀 Memory Limits Found:")
        print(f"  Max residues (any config): {summary.get('max_feasible_residues', 'N/A')}")

        if 'max_residues_fast_ode' in summary:
            print(f"  Max residues (fast ODE): {summary['max_residues_fast_ode']}")
        if 'max_residues_full_ode' in summary:
            print(f"  Max residues (full ODE): {summary['max_residues_full_ode']}")

        print(f"\n🔧 ODE Implementation Split:")
        print(f"  Fast ODE feasible configs: {summary.get('fast_ode_configs', 0)}")
        print(f"  Full ODE feasible configs: {summary.get('full_ode_configs', 0)}")

        print(f"\n✅ FEASIBLE CONFIGURATIONS (PASSED TRAINING TEST):")
        for i, result in enumerate(search_results['feasible_configs'], 1):
            config = result['config']
            training_result = result['training_result']

            ode_type = "Fast ODE" if config['use_fast_ode'] else "Full ODE"

            print(
                f"  {i}. {config['max_residues']} res, batch={config['batch_size']}, cluster={config['reduced_cluster_size']}, {ode_type}")
            print(
                f"     🧬 Tested on: {training_result['protein_tested']} ({training_result['protein_residues']} residues)")
            print(f"     💾 Peak memory: {training_result['max_memory_mb']:.0f} MB")

            # Show baseline performance if available
            if result['baseline_analysis'] and 'summary' in result['baseline_analysis']:
                baseline_summary = result['baseline_analysis']['summary']['baseline_summary']
                if 'identity' in baseline_summary:
                    identity_loss = baseline_summary['identity']['mean']
                    print(f"     📊 Identity baseline: {identity_loss:.3f}")

        print(f"\n❌ INFEASIBLE CONFIGURATIONS:")
        for i, result in enumerate(search_results['infeasible_configs'], 1):
            config = result['config']
            training_result = result['training_result']
            ode_type = "Fast ODE" if config['use_fast_ode'] else "Full ODE"

            print(
                f"  {i}. {config['max_residues']} res, batch={config['batch_size']}, cluster={config['reduced_cluster_size']}, {ode_type}")

            if training_result['protein_tested']:
                print(
                    f"     🧬 Tested on: {training_result['protein_tested']} ({training_result['protein_residues']} residues)")
                print(f"     💾 Peak memory: {training_result['max_memory_mb']:.0f} MB")

            print(f"     💀 Error: {training_result['error']}")
            if training_result.get('is_oom', False):
                print(f"     🔥 CUDA OUT OF MEMORY")

    else:
        print(f"\n💀 NO FEASIBLE CONFIGURATIONS FOUND!")
        print(f"   All tested configurations failed memory requirements.")
        print(f"   Consider:")
        print(f"   - Reducing max_residues further")
        print(f"   - Using smaller batch_size")
        print(f"   - Using smaller max_cluster_size")
        print(f"   - Using fast_ode=True")
        print(f"   - Using a machine with more GPU memory")


def save_search_results(search_results: Dict, output_file: str):
    """Save search results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(search_results, f, indent=2)
    print(f"📄 Full search results saved to: {output_file}")


def main(args_dict=None):
    if args_dict is None:
        parser = argparse.ArgumentParser(
            description='Enhanced Loss Analysis with Memory-Critical Hyperparameter Search')

        parser.add_argument('--data_dir', type=str, required=True,
                            help='Directory containing protein evoformer blocks')
        parser.add_argument('--splits_dir', type=str, default=None,
                            help='Directory containing data splits (for validation proteins)')
        parser.add_argument('--max_proteins', type=int, default=5,
                            help='Maximum number of proteins to analyze for each config')
        parser.add_argument('--reduced_cluster_size', type=int, default=64,
                            help='Maximum cluster size (for baseline-only mode)')
        parser.add_argument('--max_residues', type=int, default=None,
                            help='Maximum residues (for baseline-only mode)')
        parser.add_argument('--output_file', type=str, default=None,
                            help='Output JSON file for results')
        parser.add_argument('--mode', type=str, choices=['baselines_only', 'memory_search'],
                            default='memory_search',
                            help='Analysis mode: baselines_only or memory_search')

        args = parser.parse_args()
    else:
        # Parse from dictionary (used when calling main() directly)
        class Args:
            pass

        args = Args()
        args.data_dir = args_dict.get("data_dir")
        args.max_proteins = args_dict.get("max_proteins", 5)
        args.reduced_cluster_size = args_dict.get("reduced_cluster_size", 64)
        args.max_residues = args_dict.get("max_residues", None)
        args.output_file = args_dict.get("output_file", None)
        args.mode = args_dict.get("mode", "memory_search")
        args.splits_dir = args_dict.get("splits_dir", None)

    if args.output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.mode == 'memory_search':
            args.output_file = f"memory_search_results_{timestamp}.json"
        else:
            args.output_file = f"loss_analysis_{timestamp}.json"

    print("🔬 ENHANCED EVOFORMER NEURAL ODE LOSS ANALYZER")
    print("=" * 50)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🎯 Mode: {args.mode}")
    print(f"🔢 Max proteins per config: {args.max_proteins}")

    if args.mode == 'baselines_only':
        print(f"📏 Reduced cluster size: {args.reduced_cluster_size}")
        print(f"📐 Max residues: {args.max_residues}")

        if not args.splits_dir:
            print("⚠️  No splits_dir provided - scanning all proteins in data directory")
            # Fallback: scan data directory
            all_proteins = []
            for item in Path(data_dir).iterdir():
                if item.is_dir() and item.name.endswith('_evoformer_blocks'):
                    protein_id = item.name.replace('_evoformer_blocks', '')
                    all_proteins.append(protein_id)

            # Filter by size if specified
            if args.max_residues is not None:
                valid_proteins, oversized = filter_proteins_by_size(all_proteins, args.data_dir, args.max_residues)
                print(f"🔍 {len(valid_proteins)} proteins ≤ {args.max_residues} residues (filtered {len(oversized)})")
            else:
                valid_proteins = all_proteins

            # Shuffle and limit
            import random
            random.shuffle(valid_proteins)
            analysis_proteins = valid_proteins[:args.max_proteins]
        else:
            # Use validation split
            val_proteins = get_available_proteins(args.data_dir, args.splits_dir, 'validation')
            if args.max_residues is not None:
                valid_proteins, oversized = filter_proteins_by_size(val_proteins, args.data_dir, args.max_residues)
                print(
                    f"🔍 {len(valid_proteins)} validation proteins ≤ {args.max_residues} residues (filtered {len(oversized)})")
            else:
                valid_proteins = val_proteins

            import random
            random.shuffle(valid_proteins)
            analysis_proteins = valid_proteins[:args.max_proteins]

        print(f"📊 Analyzing: {analysis_proteins}")
        print("")

        # Run baseline analysis only
        analysis_results = analyze_multiple_proteins(
            args.data_dir,
            analysis_proteins,
            args.reduced_cluster_size,
            args.max_residues
        )

        # Determine thresholds and print analysis (reuse existing functions)
        thresholds = determine_loss_thresholds(analysis_results)
        print_loss_analysis(analysis_results, thresholds)

        # Save results
        output_data = {
            'analysis_results': analysis_results,
            'thresholds': thresholds,
            'mode': 'baselines_only',
            'config': {
                'reduced_cluster_size': args.reduced_cluster_size,
                'max_residues': args.max_residues,
                'max_proteins': args.max_proteins
            },
            'metadata': {
                'script_version': '2.0',
                'analysis_date': datetime.now().isoformat(),
                'description': 'Evoformer Neural ODE Loss Analysis - Baselines Only'
            }
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"📄 Analysis saved to: {args.output_file}")

    elif args.mode == 'memory_search':
        if not args.splits_dir:
            print("⚠️  No splits_dir provided - will fall back to scanning data directory")

        print(f"🧪 Testing memory-critical hyperparameter combinations...")
        print("")

        # Run memory-critical hyperparameter search
        search_results = hyperparameter_search_memory_critical(
            args.data_dir,
            args.splits_dir,
            args.max_proteins
        )

        # Print comprehensive summary
        print_memory_search_summary(search_results)

        # Save detailed results
        save_search_results(search_results, args.output_file)

        print(f"\n✅ Memory search complete!")
        print(f"📊 Found {search_results['summary']['feasible_count']} feasible configurations")
        print(f"💡 Use feasible configs for next phase of hyperparameter search")

        # Quick recommendations
        if search_results['summary']['feasible_count'] > 0:
            print(f"\n🎯 RECOMMENDATIONS FOR NEXT PHASE:")
            print(
                f"  1. Focus on the {min(3, search_results['summary']['feasible_count'])} most aggressive feasible configs")
            print(f"  2. Test learning_rate variations: [1e-4, 5e-4, 1e-3, 2e-3]")
            print(f"  3. Test hidden_dim variations: [32, 64, 128]")
            print(f"  4. Test integrator variations: ['rk4', 'dopri5']")
            print(f"  5. Return these results for the next search phase")


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
    print(
        f"  Config: cluster_size={summary['config']['reduced_cluster_size']}, max_residues={summary['config']['max_residues']}")
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


if __name__ == "__main__":
    main()