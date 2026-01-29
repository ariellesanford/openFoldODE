#!/usr/bin/env python3
"""
Ablation Study Runner
Trains models with each module ablated and compares performance/efficiency
"""

import os
import sys
import argparse
import time
import json
import torch
from pathlib import Path
from datetime import datetime
from torchdiffeq import odeint

from evoformer_ode_ablation import EvoformerODEFuncAblation, ABLATION_CONFIGS


def get_data_dir():
    """Return the data directory for the current system"""
    candidates = [
        Path("/Volumes/Extreme SSD/data"),
        Path("/media/visitor/Extreme SSD/data"),
        Path("/pscratch/sd/a/arielles/data"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No data directory found")


def load_protein_data(protein_id, data_dir, device='cpu', max_cluster_size=64, blocks_subdir="endpoint_blocks"):
    """Load start and end blocks for a protein"""
    blocks_dir = data_dir / blocks_subdir / f"{protein_id}_evoformer_blocks" / "recycle_0"
    
    m_start = torch.load(blocks_dir / "m_block_0.pt", map_location=device)
    z_start = torch.load(blocks_dir / "z_block_0.pt", map_location=device)
    m_end = torch.load(blocks_dir / "m_block_48.pt", map_location=device)
    z_end = torch.load(blocks_dir / "z_block_48.pt", map_location=device)
    
    # Remove batch dimension if present
    if m_start.dim() == 4:
        m_start = m_start.squeeze(0)
    if z_start.dim() == 4:
        z_start = z_start.squeeze(0)
    if m_end.dim() == 4:
        m_end = m_end.squeeze(0)
    if z_end.dim() == 4:
        z_end = z_end.squeeze(0)
    
    # Limit cluster size for memory
    if max_cluster_size and m_start.shape[0] > max_cluster_size:
        m_start = m_start[:max_cluster_size]
        m_end = m_end[:max_cluster_size]
    
    return (m_start, z_start), (m_end, z_end)


def train_ablation_model(model, train_proteins, data_dir, device, epochs=100, lr=1e-3, max_cluster_size=64):
    """Train a single ablation model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_proteins = 0
        
        for protein_id in train_proteins:
            try:
                (m_start, z_start), (m_end, z_end) = load_protein_data(
                    protein_id, data_dir, device=device, max_cluster_size=max_cluster_size
                )
                
                optimizer.zero_grad()
                
                # Integrate ODE from t=0 to t=1
                t_span = torch.tensor([0.0, 1.0], device=device)
                state0 = (m_start, z_start)
                
                trajectory = odeint(model, state0, t_span, method='rk4')
                m_pred = trajectory[0][-1]
                z_pred = trajectory[1][-1]
                
                # Compute loss
                loss_m = torch.nn.functional.mse_loss(m_pred, m_end)
                loss_z = torch.nn.functional.mse_loss(z_pred, z_end)
                loss = loss_m + loss_z
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_proteins += 1
                
            except Exception as e:
                print(f"  Skipping {protein_id}: {e}")
                continue
        
        avg_loss = epoch_loss / max(n_proteins, 1)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return train_losses


def evaluate_model(model, test_proteins, data_dir, device, max_cluster_size=64):
    """Evaluate model on test set, measuring loss and inference time"""
    model.eval()
    
    total_loss = 0.0
    total_time = 0.0
    n_proteins = 0
    results = []
    
    with torch.no_grad():
        for protein_id in test_proteins:
            try:
                (m_start, z_start), (m_end, z_end) = load_protein_data(
                    protein_id, data_dir, device=device, max_cluster_size=max_cluster_size
                )
                
                # Time inference
                t_span = torch.tensor([0.0, 1.0], device=device)
                state0 = (m_start, z_start)
                
                start_time = time.time()
                trajectory = odeint(model, state0, t_span, method='rk4')
                m_pred = trajectory[0][-1]
                z_pred = trajectory[1][-1]
                if device == 'cuda':
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                
                # Compute loss
                loss_m = torch.nn.functional.mse_loss(m_pred, m_end).item()
                loss_z = torch.nn.functional.mse_loss(z_pred, z_end).item()
                loss = loss_m + loss_z
                
                # Cosine similarity
                m_cos = torch.nn.functional.cosine_similarity(
                    m_pred.flatten(), m_end.flatten(), dim=0
                ).item()
                z_cos = torch.nn.functional.cosine_similarity(
                    z_pred.flatten(), z_end.flatten(), dim=0
                ).item()
                
                results.append({
                    'protein_id': protein_id,
                    'loss': loss,
                    'loss_m': loss_m,
                    'loss_z': loss_z,
                    'msa_cosine': m_cos,
                    'pair_cosine': z_cos,
                    'inference_time': inference_time,
                    'n_residues': m_start.shape[1],
                })
                
                total_loss += loss
                total_time += inference_time
                n_proteins += 1
                
            except Exception as e:
                print(f"  Eval skipping {protein_id}: {e}")
                continue
    
    return {
        'avg_loss': total_loss / max(n_proteins, 1),
        'avg_time': total_time / max(n_proteins, 1),
        'total_time': total_time,
        'n_proteins': n_proteins,
        'per_protein': results,
    }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_ablation_study(args):
    """Run full ablation study"""
    data_dir = Path(args.data_dir) if args.data_dir else get_data_dir()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load protein splits
    splits_dir = Path(args.splits_dir)
    train_proteins = (splits_dir / "training_chains.txt").read_text().strip().split('\n')
    test_proteins = (splits_dir / "testing_chains.txt").read_text().strip().split('\n')
    
    if args.max_proteins:
        train_proteins = train_proteins[:args.max_proteins]
        test_proteins = test_proteins[:min(args.max_proteins // 2, len(test_proteins))]
    
    print(f"Data dir: {data_dir}")
    print(f"Train proteins: {len(train_proteins)}")
    print(f"Test proteins: {len(test_proteins)}")
    print(f"Device: {device}")
    print()
    
    # Model dimensions
    c_m = 256
    c_z = 128
    hidden_dim = args.hidden_dim
    
    # Results storage
    all_results = {}
    
    # Run each ablation
    ablations_to_run = args.ablations if args.ablations else list(ABLATION_CONFIGS.keys())
    
    for ablation_name in ablations_to_run:
        print(f"{'='*60}")
        print(f"Running ablation: {ablation_name}")
        print(f"{'='*60}")
        
        config = ABLATION_CONFIGS.get(ablation_name, {})
        model = EvoformerODEFuncAblation(c_m, c_z, hidden_dim, **config)
        
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")
        print(f"  Config: {config if config else 'Full model (no ablation)'}")
        
        # Train
        print(f"  Training...")
        train_start = time.time()
        train_losses = train_ablation_model(
            model, train_proteins, data_dir, device,
            epochs=args.epochs, lr=args.lr
        )
        train_time = time.time() - train_start
        
        # Evaluate
        print(f"  Evaluating...")
        eval_results = evaluate_model(model, test_proteins, data_dir, device)
        
        # Store results
        all_results[ablation_name] = {
            'config': config,
            'n_params': n_params,
            'train_time': train_time,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'train_losses': train_losses,
            'eval': eval_results,
        }
        
        print(f"  Results:")
        print(f"    Parameters: {n_params:,}")
        print(f"    Train time: {train_time:.1f}s")
        print(f"    Final train loss: {train_losses[-1]:.6f}")
        print(f"    Test loss: {eval_results['avg_loss']:.6f}")
        print(f"    Avg inference time: {eval_results['avg_time']*1000:.2f}ms")
        print()
        
        # Save model
        model_path = output_dir / f"ablation_{ablation_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'ablation_name': ablation_name,
            'results': all_results[ablation_name],
        }, model_path)
    
    # Save summary
    summary_path = output_dir / "ablation_summary.json"
    
    # Convert to JSON-serializable format
    summary = {}
    for name, results in all_results.items():
        summary[name] = {
            'config': results['config'],
            'n_params': results['n_params'],
            'train_time': results['train_time'],
            'final_train_loss': results['final_train_loss'],
            'test_loss': results['eval']['avg_loss'],
            'avg_inference_time_ms': results['eval']['avg_time'] * 1000,
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Ablation':<25} {'Params':>12} {'Train Loss':>12} {'Test Loss':>12} {'Infer (ms)':>12}")
    print("-"*80)
    
    for name, results in all_results.items():
        print(f"{name:<25} {results['n_params']:>12,} {results['final_train_loss']:>12.6f} "
              f"{results['eval']['avg_loss']:>12.6f} {results['eval']['avg_time']*1000:>12.2f}")
    
    print("-"*80)
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--splits-dir', type=str, required=True, help='Splits directory')
    parser.add_argument('--output-dir', type=str, default='ablation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per ablation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--max-proteins', type=int, help='Limit proteins for quick testing')
    parser.add_argument('--ablations', nargs='+', help='Specific ablations to run')
    
    args = parser.parse_args()
    run_ablation_study(args)


if __name__ == "__main__":
    main()