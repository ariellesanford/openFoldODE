#!/usr/bin/env python3

"""
Simplified and more effective Neural ODE training for Evoformer
Fixes the key issues: learning signal, loss normalization, and training stability
FIXED: Proper TrainingLogger usage throughout
"""

import os
import gc
import torch
import argparse
import sys
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
from training_logger import TrainingLogger


def get_project_root():
    """Get the path to the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def clear_memory(device: str):
    """Clear GPU memory if using CUDA"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def get_dataset(data_dir: str) -> List[str]:
    """Get list of available protein IDs"""
    datasets = []
    for name in os.listdir(data_dir):
        full_path = os.path.join(data_dir, name)
        if (os.path.isdir(full_path) and name.endswith('_evoformer_blocks') and
                os.path.isdir(os.path.join(full_path, 'recycle_0'))):
            protein_id = name.replace('_evoformer_blocks', '')
            datasets.append(protein_id)
    return sorted(datasets)


def load_protein_block(protein_id: str, block_idx: int, data_dir: str,
                       device: str, max_cluster_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load M and Z tensors for a specific block"""
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, f"m_block_{block_idx}.pt")
    z_path = os.path.join(protein_dir, f"z_block_{block_idx}.pt")

    m = torch.load(m_path, map_location=device)
    z = torch.load(z_path, map_location=device)

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
                          pred_z: torch.Tensor, target_z: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute loss with proper scaling that preserves learning signal
    """
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
        'total': total_loss,
        'msa_raw': msa_loss,
        'pair_raw': pair_loss,
        'msa_scaled': msa_scaled,
        'pair_scaled': pair_scaled
    }


def train_single_protein_batched(protein_id: str, ode_func: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                 scaler: GradScaler, args: argparse.Namespace) -> Dict:
    """Train using temporal batching - break sequence into smaller chunks"""

    # Find all available blocks
    protein_dir = os.path.join(args.data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
    available_blocks = []
    for i in range(args.max_blocks):
        m_path = os.path.join(protein_dir, f"m_block_{i}.pt")
        z_path = os.path.join(protein_dir, f"z_block_{i}.pt")
        if os.path.exists(m_path) and os.path.exists(z_path):
            available_blocks.append(i)
        else:
            break

    num_blocks = len(available_blocks)
    batch_size = args.batch_size

    print(f"Batched: {num_blocks} blocks, batch_size={batch_size}")

    # Load initial state
    m_current, z_current = load_protein_block(
        protein_id, available_blocks[0], args.data_dir, args.device, args.reduced_cluster_size
    )

    total_loss = 0
    total_batches = 0
    all_losses = []

    # Process in batches
    for batch_start in range(0, num_blocks - 1, batch_size):
        batch_end = min(batch_start + batch_size, num_blocks - 1)

        # Create time grid for this batch
        batch_blocks = available_blocks[batch_start:batch_end + 1]
        t_grid = torch.linspace(0.0, 1.0, len(batch_blocks)).to(args.device)

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            # Solve ODE for this batch
            trajectory = odeint(
                ode_func,
                (m_current, z_current),
                t_grid,
                method=args.integrator,
                rtol=1e-4,
                atol=1e-5
            )

            # Compute loss for this batch
            batch_loss = 0
            valid_steps = 0

            for i, block_idx in enumerate(batch_blocks[1:], 1):  # Skip first (initial state)
                m_target, z_target = load_protein_block(
                    protein_id, block_idx, args.data_dir, args.device, args.reduced_cluster_size
                )

                m_pred = trajectory[0][i]
                z_pred = trajectory[1][i]

                loss_dict = compute_adaptive_loss(m_pred, m_target, z_pred, z_target)
                batch_loss += loss_dict['total']
                valid_steps += 1

            batch_loss = batch_loss / valid_steps

        # Backward pass for this batch
        if args.use_amp:
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += batch_loss.item()
        total_batches += 1
        all_losses.append(batch_loss.item())

        # Update current state for next batch (detached from computation graph)
        if batch_end < num_blocks - 1:  # Not the last batch
            m_current = trajectory[0][-1].detach()
            z_current = trajectory[1][-1].detach()

        # AGGRESSIVE MEMORY CLEARING
        del trajectory, batch_loss, loss_dict, m_pred, z_pred, m_target, z_target
        clear_memory(args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()

    avg_loss = total_loss / total_batches

    return {
        'protein': protein_id,
        'approach': 'batched',
        'num_blocks': num_blocks,
        'batch_size': batch_size,
        'num_batches': total_batches,
        'total_loss': avg_loss,
        'batch_losses': all_losses
    }


def train_single_protein_strided(protein_id: str, ode_func: torch.nn.Module, optimizer: torch.optim.Optimizer,
                                 scaler: GradScaler, args: argparse.Namespace) -> Dict:
    """Train using block striding - use every nth block"""

    # Find all available blocks
    protein_dir = os.path.join(args.data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
    available_blocks = []
    for i in range(args.max_blocks):
        m_path = os.path.join(protein_dir, f"m_block_{i}.pt")
        z_path = os.path.join(protein_dir, f"z_block_{i}.pt")
        if os.path.exists(m_path) and os.path.exists(z_path):
            available_blocks.append(i)
        else:
            break

    # Select blocks with stride (ensure we include first and last)
    stride = args.block_stride
    selected_blocks = []

    # Always include block 0
    selected_blocks.append(available_blocks[0])

    # Add strided blocks
    for i in range(stride, len(available_blocks), stride):
        selected_blocks.append(available_blocks[i])

    # Ensure we include the last block if not already included
    if available_blocks[-1] not in selected_blocks:
        selected_blocks.append(available_blocks[-1])

    print(f"Strided: {len(selected_blocks)}/{len(available_blocks)} blocks, stride={stride}")

    # Load initial state
    m_init, z_init = load_protein_block(
        protein_id, selected_blocks[0], args.data_dir, args.device, args.reduced_cluster_size
    )

    # Create time grid
    t_grid = torch.linspace(0.0, 1.0, len(selected_blocks)).to(args.device)

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        # Solve ODE for selected blocks
        trajectory = odeint(
            ode_func,
            (m_init, z_init),
            t_grid,
            method=args.integrator,
            rtol=1e-4,
            atol=1e-5
        )

        # Compute loss against selected blocks
        total_loss = 0
        valid_steps = 0

        for i, block_idx in enumerate(selected_blocks[1:], 1):  # Skip first block
            m_target, z_target = load_protein_block(
                protein_id, block_idx, args.data_dir, args.device, args.reduced_cluster_size
            )

            m_pred = trajectory[0][i]
            z_pred = trajectory[1][i]

            loss_dict = compute_adaptive_loss(m_pred, m_target, z_pred, z_target)
            total_loss += loss_dict['total']
            valid_steps += 1

        total_loss = total_loss / valid_steps

    # Backward pass
    if args.use_amp:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
        optimizer.step()

    # Store loss value before deletion
    total_loss_value = total_loss.item()

    # AGGRESSIVE MEMORY CLEARING
    del trajectory, total_loss, loss_dict, m_pred, z_pred, m_target, z_target, m_init, z_init
    clear_memory(args.device)
    if args.device == 'cuda':
        torch.cuda.synchronize()

    return {
        'protein': protein_id,
        'approach': 'strided',
        'num_blocks': len(selected_blocks),
        'total_available': len(available_blocks),
        'stride': stride,
        'selected_blocks': selected_blocks,
        'total_loss': total_loss_value
    }


def main():
    parser = argparse.ArgumentParser(description='Simplified Neural ODE Training')

    # Core settings
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for logging')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    # Model settings
    parser.add_argument('--use_fast_ode', action='store_true', help='Use fast ODE implementation')
    parser.add_argument('--reduced_cluster_size', type=int, default=64, help='Max cluster size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--integrator', choices=['dopri5', 'rk4', 'euler'], default='rk4', help='ODE integrator')

    # Timestep control - choose ONE approach
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for temporal batching (1-10)')
    parser.add_argument('--block_stride', type=int, default=None, help='Use every nth block (must divide 48)')
    parser.add_argument('--max_blocks', type=int, default=49, help='Maximum blocks to use total')

    # Optimization settings
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--test_single_protein', type=str, default=None, help='Test on single protein')
    parser.add_argument('--max_residues', type=int, default=None, help='Skip proteins with more than N residues')

    args = parser.parse_args()

    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'

    if args.device == 'cpu':
        args.use_amp = False  # AMP only works on CUDA

    print(f"🚀 Starting Neural ODE Training")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"💻 Device: {args.device}")
    print(f"🔧 Settings: LR={args.learning_rate}, Fast ODE={args.use_fast_ode}, AMP={args.use_amp}")

    # Initialize CUDA if using GPU
    if args.device == 'cuda':
        torch.cuda.init()
        print(f"🚀 CUDA initialized: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Load dataset
    dataset = get_dataset(args.data_dir)
    if not dataset:
        print(f"❌ No proteins found in {args.data_dir}")
        return

    if args.test_single_protein:
        if args.test_single_protein in dataset:
            dataset = [args.test_single_protein]
        else:
            print(f"❌ Protein {args.test_single_protein} not found")
            return

    print(f"🧬 Found {len(dataset)} proteins: {dataset}")

    # Initialize model
    c_m = 256  # MSA embedding dimension
    c_z = 128  # Pair embedding dimension

    if args.use_fast_ode:
        ode_func = EvoformerODEFuncFast(c_m, c_z, args.hidden_dim).to(args.device)
    else:
        ode_func = EvoformerODEFunc(c_m, c_z, args.hidden_dim).to(args.device)

    optimizer = optim.Adam(ode_func.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=args.use_amp)

    print(f"🤖 Model initialized: {sum(p.numel() for p in ode_func.parameters())} parameters")

    # FIXED: Initialize training logger properly
    logger = None
    if args.experiment_name and args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logger = TrainingLogger(args.output_dir, args.experiment_name)

            # Log configuration
            model_info = {
                'total_params': sum(p.numel() for p in ode_func.parameters()),
                'model_type': 'EvoformerODEFuncFast' if args.use_fast_ode else 'EvoformerODEFunc',
                'loss_function': 'Adaptive MSE (variance-scaled)',
            }

            optimizer_info = {
                'learning_rate': args.learning_rate,
            }

            logger.log_configuration(args, model_info, optimizer_info)
            print(f"📝 Training logger initialized: {args.experiment_name}")
            print(f"📊 Reports will be saved to: {args.output_dir}/{args.experiment_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize logger: {e}")
            logger = None

    # Training loop
    previous_losses = []
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")

        # FIXED: Log epoch start
        if logger:
            logger.log_epoch_start(epoch + 1, args.epochs, dataset)

        epoch_losses = []
        successful_proteins = 0
        epoch_start_time = time.time()

        for protein_idx, protein_id in enumerate(dataset):
            print(f"  [{protein_idx + 1}/{len(dataset)}] Processing {protein_id}... ", end='', flush=True)

            protein_start_time = time.time()

            # Optional: Skip large proteins to avoid OOM
            if args.max_residues is not None:
                try:
                    protein_dir = os.path.join(args.data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
                    m_test = torch.load(os.path.join(protein_dir, "m_block_0.pt"), map_location='cpu')
                    num_residues = m_test.shape[-2] if m_test.dim() == 4 else m_test.shape[-2]
                    if num_residues > args.max_residues:
                        print(f"⏭️  SKIPPED ({num_residues} residues > {args.max_residues} limit)")
                        del m_test
                        continue
                    del m_test
                except:
                    pass  # If we can't check size, proceed anyway

            # Choose training approach
            if args.batch_size is not None:
                step_info = train_single_protein_batched(protein_id, ode_func, optimizer, scaler, args)
            else:
                step_info = train_single_protein_strided(protein_id, ode_func, optimizer, scaler, args)

            protein_time = time.time() - protein_start_time
            epoch_losses.append(step_info['total_loss'])
            successful_proteins += 1
            print(f"✅ Loss: {step_info['total_loss']:.3f}")

            # FIXED: Log protein step
            if logger:
                logger.log_protein_step(
                    protein_id=protein_id,
                    step_idx=protein_idx,
                    loss=step_info['total_loss'],
                    step_info=step_info,
                    time_taken=protein_time
                )

            # Show detailed metrics for first protein of first epoch
            if epoch == 0 and protein_idx == 0:
                if step_info['approach'] == 'batched':
                    print(f"      📦 Batches: {step_info['num_batches']}, batch_size: {step_info['batch_size']}")
                else:
                    print(f"      🔢 Stride: {step_info['stride']}, blocks: {step_info['selected_blocks']}")
                print(f"      📊 Using {step_info['num_blocks']} blocks")

            # FINAL CLEANUP after each protein
            clear_memory(args.device)
            if args.device == 'cuda':
                torch.cuda.synchronize()

        # FIXED: Log epoch end
        if logger:
            logger.log_epoch_end()

        # Epoch summary
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start_time

            print(f"📊 Epoch {epoch + 1} Summary:")
            print(f"    Average Loss: {avg_loss:.3f}")
            print(f"    Successful proteins: {successful_proteins}/{len(dataset)}")
            print(f"    Loss range: [{min(epoch_losses):.3f}, {max(epoch_losses):.3f}]")
            print(f"    Epoch time: {epoch_time:.1f} seconds")

            # Check for learning progress
            if epoch > 0 and len(previous_losses) > 0:
                prev_avg = sum(previous_losses) / len(previous_losses)
                improvement = (prev_avg - avg_loss) / prev_avg * 100
                if improvement > 0:
                    print(f"    📈 Improvement: {improvement:.1f}% better than previous epoch")
                else:
                    print(f"    📉 Change: {abs(improvement):.1f}% worse than previous epoch")

            previous_losses = epoch_losses[:]

    # FIXED: Save final model and close logger
    if logger:
        # Save final model
        model_path = os.path.join(args.output_dir, f"{args.experiment_name}_final_model.pt")
        torch.save({
            'model_state_dict': ode_func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(args),
            'final_loss': avg_loss if epoch_losses else None
        }, model_path)

        logger.log_training_complete(model_path)
        print(f"📊 Training complete! Full report saved to: {args.output_dir}/{args.experiment_name}")
        print(f"🤖 Final model saved to: {model_path}")

    print(f"\n🎯 Training completed!")


if __name__ == "__main__":
    main()