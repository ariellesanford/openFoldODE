"""
Enhanced Neural ODE training for Evoformer with:
- Smart learning rate scheduling based on validation loss
- Early stopping with validation loss monitoring
- Improved training stability
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


class LearningRateScheduler:
    """Smart learning rate scheduler based on validation loss trends"""

    def __init__(self, optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.last_lr_reduction = 0
        self.lr_reductions = 0

    def step(self, val_loss, epoch):
        """Update learning rate based on validation loss"""
        current_lr = self.optimizer.param_groups[0]['lr']

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            if self.verbose:
                print(f"📈 New best validation loss: {val_loss:.4f}")
        else:
            self.patience_counter += 1

        # Reduce learning rate if no improvement for patience epochs
        if self.patience_counter >= self.patience and current_lr > self.min_lr:
            new_lr = max(current_lr * self.factor, self.min_lr)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            self.lr_reductions += 1
            self.last_lr_reduction = epoch
            self.patience_counter = 0  # Reset patience after LR reduction

            if self.verbose:
                print(f"🔽 Learning rate reduced: {current_lr:.2e} → {new_lr:.2e} (reduction #{self.lr_reductions})")

            return True  # LR was reduced

        return False  # No LR change

    def get_last_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """Early stopping based on validation loss with smart criteria"""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, epoch, model=None):
        """Check if training should stop"""

        # Check for improvement
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0

            # Save best model state
            if model is not None and self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()

            if self.verbose:
                print(f"🎯 New best validation loss: {val_loss:.4f} (epoch {epoch})")

        else:
            self.patience_counter += 1

        # Check if we should stop
        if self.patience_counter >= self.patience:
            self.should_stop = True

            if self.verbose:
                print(f"🛑 Early stopping triggered!")
                print(f"   No improvement for {self.patience} epochs")
                print(f"   Best validation loss: {self.best_loss:.4f} (epoch {self.best_epoch})")

            # Restore best weights
            if self.restore_best_weights and self.best_model_state is not None and model is not None:
                model.load_state_dict(self.best_model_state)
                if self.verbose:
                    print(f"🔄 Restored best model weights from epoch {self.best_epoch}")

        return self.should_stop

    def get_best_loss(self):
        return self.best_loss

    def get_best_epoch(self):
        return self.best_epoch


def get_project_root():
    """Get the path to the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def clear_memory(device: str):
    """Clear GPU memory if using CUDA"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


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


def get_train_val_datasets(data_dir: str, splits_dir: str) -> Tuple[List[str], List[str]]:
    """Get training and validation datasets"""
    train_proteins = get_available_proteins(data_dir, splits_dir, 'training')
    val_proteins = get_available_proteins(data_dir, splits_dir, 'validation')
    return train_proteins, val_proteins


def filter_proteins_by_size(proteins: List[str], data_dir: str, max_residues: int = None) -> Tuple[
    List[str], List[str]]:
    """Filter proteins by residue count, return (valid_proteins, oversized_proteins)"""
    if max_residues is None:
        return proteins, []

    valid_proteins = []
    oversized_proteins = []

    print(f"🔍 Filtering {len(proteins)} proteins by size limit ({max_residues} residues)...")

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
                    oversized_proteins.append((protein_id, num_residues))
            else:
                # If we can't load the file, skip it
                oversized_proteins.append((protein_id, "unknown"))

        except Exception as e:
            # If there's any error checking size, skip the protein
            oversized_proteins.append((protein_id, f"error: {e}"))

    print(f"  ✅ Valid proteins: {len(valid_proteins)}")
    print(f"  ⏭️  Oversized proteins: {len(oversized_proteins)}")

    if oversized_proteins and len(oversized_proteins) <= 10:
        # Show details if not too many
        print(f"  📊 Oversized proteins:")
        for protein_id, size in oversized_proteins:
            if isinstance(size, int):
                print(f"    - {protein_id}: {size} residues")
            else:
                print(f"    - {protein_id}: {size}")
    elif oversized_proteins:
        print(f"  📊 Oversized proteins (showing first 5):")
        for protein_id, size in oversized_proteins[:5]:
            if isinstance(size, int):
                print(f"    - {protein_id}: {size} residues")
            else:
                print(f"    - {protein_id}: {size}")
        print(f"    ... and {len(oversized_proteins) - 5} more")

    return valid_proteins, [p[0] for p in oversized_proteins]


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


def validate_single_protein_batched(protein_id: str, ode_func: torch.nn.Module, args: argparse.Namespace) -> Dict:
    """Validate using temporal batching - no gradient updates"""
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

    # Load initial state
    m_current, z_current = load_protein_block(
        protein_id, available_blocks[0], args.data_dir, args.device, args.reduced_cluster_size
    )

    total_loss = 0
    total_batches = 0

    # Process in batches
    for batch_start in range(0, num_blocks - 1, batch_size):
        batch_end = min(batch_start + batch_size, num_blocks - 1)

        # Create time grid for this batch
        batch_blocks = available_blocks[batch_start:batch_end + 1]
        t_grid = torch.linspace(0.0, 1.0, len(batch_blocks)).to(args.device)

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
            batch_loss += loss_dict['total'].item()  # Convert to scalar immediately
            valid_steps += 1

        batch_loss = batch_loss / valid_steps
        total_loss += batch_loss
        total_batches += 1

        # Update current state for next batch (detached)
        if batch_end < num_blocks - 1:  # Not the last batch
            m_current = trajectory[0][-1].detach()
            z_current = trajectory[1][-1].detach()

        # Clean up
        del trajectory, m_target, z_target, m_pred, z_pred
        clear_memory(args.device)

    avg_loss = total_loss / total_batches
    return {
        'protein': protein_id,
        'approach': 'batched',
        'num_blocks': num_blocks,
        'batch_size': batch_size,
        'num_batches': total_batches,
        'total_loss': avg_loss
    }


def validate_model(ode_func: torch.nn.Module, val_proteins: List[str], args: argparse.Namespace) -> Dict:
    """Run validation on validation set using batched approach"""
    ode_func.eval()
    val_losses = []
    successful_validations = 0

    with torch.no_grad():
        for val_idx, protein_id in enumerate(val_proteins):
            print(f"    [{val_idx + 1}/{len(val_proteins)}] Validating {protein_id}... ", end='', flush=True)

            try:
                step_info = validate_single_protein_batched(protein_id, ode_func, args)
                val_losses.append(step_info['total_loss'])
                successful_validations += 1
                print(f"✅ Loss: {step_info['total_loss']:.3f}")

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                continue

            # Memory cleanup after each validation protein
            clear_memory(args.device)
            if args.device == 'cuda':
                torch.cuda.synchronize()

    ode_func.train()  # Switch back to training mode

    if val_losses:
        return {
            'avg_loss': sum(val_losses) / len(val_losses),
            'min_loss': min(val_losses),
            'max_loss': max(val_losses),
            'num_proteins': len(val_losses),
            'successful_validations': successful_validations,
            'skipped_validations': 0
        }
    else:
        return {
            'avg_loss': float('inf'),
            'num_proteins': 0,
            'successful_validations': 0,
            'skipped_validations': 0
        }


def main():
    parser = argparse.ArgumentParser(description='Enhanced Neural ODE Training with LR Scheduling and Early Stopping')

    # Core settings
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--splits_dir', type=str, default=None, help='Directory containing split files')
    parser.add_argument('--mode', type=str, choices=['training', 'testing', 'single_test'],
                        default='training', help='Training mode (validation runs during training)')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for logging')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')

    # Learning rate scheduling
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Epochs to wait before reducing LR if no validation improvement')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor to reduce learning rate by')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                        help='Epochs to wait before early stopping if no validation improvement')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum improvement to count as progress for early stopping')
    parser.add_argument('--restore_best_weights', action='store_true', default=True,
                        help='Restore best model weights when early stopping triggers')

    # Model settings
    parser.add_argument('--use_fast_ode', action='store_true', help='Use fast ODE implementation')
    parser.add_argument('--reduced_cluster_size', type=int, default=64, help='Max cluster size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--integrator', choices=['dopri5', 'rk4', 'euler'], default='rk4', help='ODE integrator')

    # Timestep control
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for temporal batching (1-10)')
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

    print(f"🚀 Starting Enhanced Neural ODE Training")
    print(f"📁 Data directory: {args.data_dir}")
    if args.splits_dir:
        print(f"📂 Splits directory: {args.splits_dir}")
        print(f"🎯 Mode: {args.mode}")
    print(f"💻 Device: {args.device}")
    print(f"🔧 Settings: LR={args.learning_rate}, Fast ODE={args.use_fast_ode}, AMP={args.use_amp}")
    print(f"📉 LR Scheduling: patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.min_lr}")
    print(f"🛑 Early Stopping: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")

    # Initialize CUDA if using GPU
    if args.device == 'cuda':
        torch.cuda.init()
        print(f"🚀 CUDA initialized: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Load dataset based on mode
    if args.mode == 'single_test' and args.test_single_protein:
        # Single protein test mode
        train_dataset = [args.test_single_protein]
        val_dataset = []
        print(f"🧪 Single protein test mode: {args.test_single_protein}")
    elif args.splits_dir and args.mode in ['training', 'testing']:
        if args.mode == 'training':
            # Training mode - use train/val splits
            train_dataset_raw, val_dataset_raw = get_train_val_datasets(args.data_dir, args.splits_dir)
            if not train_dataset_raw:
                print(f"❌ No training proteins found in {args.data_dir}")
                return

            print(f"🧬 Raw training proteins: {len(train_dataset_raw)}")
            print(f"🔍 Raw validation proteins: {len(val_dataset_raw)}")

            # Filter by size if max_residues is specified
            if args.max_residues is not None:
                print(f"\n📏 Applying size filter (max {args.max_residues} residues)...")

                print(f"\n📚 Training set:")
                train_dataset, train_oversized = filter_proteins_by_size(
                    train_dataset_raw, args.data_dir, args.max_residues
                )

                print(f"\n🔍 Validation set:")
                val_dataset, val_oversized = filter_proteins_by_size(
                    val_dataset_raw, args.data_dir, args.max_residues
                )

                print(f"\n📊 Final dataset sizes:")
                print(f"  Training: {len(train_dataset)} / {len(train_dataset_raw)} ({len(train_oversized)} filtered)")
                print(f"  Validation: {len(val_dataset)} / {len(val_dataset_raw)} ({len(val_oversized)} filtered)")

                if not train_dataset:
                    print(f"❌ No training proteins remain after size filtering!")
                    return

            else:
                train_dataset = train_dataset_raw
                val_dataset = val_dataset_raw
                print(f"📊 No size filtering applied")

            # Require validation set for training mode
            if not val_dataset:
                print(f"❌ No validation proteins found - required for LR scheduling and early stopping!")
                return

            # Report missing proteins from split files
            try:
                train_split = load_split_proteins(args.splits_dir, 'training')
                val_split = load_split_proteins(args.splits_dir, 'validation')

                missing_train = set(train_split) - set(train_dataset_raw)
                missing_val = set(val_split) - set(val_dataset_raw)

                if missing_train:
                    print(f"⚠️  {len(missing_train)} training proteins not found in data")
                if missing_val:
                    print(f"⚠️  {len(missing_val)} validation proteins not found in data")
            except Exception as e:
                print(f"⚠️  Could not load split info: {e}")

        else:  # testing mode
            # For testing mode, disable LR scheduling and early stopping
            train_dataset = get_available_proteins(args.data_dir, args.splits_dir, 'testing')
            val_dataset = []

            if args.max_residues is not None:
                print(f"\n📏 Applying size filter to testing set (max {args.max_residues} residues)...")
                train_dataset, test_oversized = filter_proteins_by_size(
                    train_dataset, args.data_dir, args.max_residues
                )
                print(f"📊 Testing: {len(train_dataset)} ({len(test_oversized)} filtered)")

            print(f"🧪 Testing mode: {len(train_dataset)} proteins")
            print("⚠️  LR scheduling and early stopping disabled in testing mode")
    else:
        # Fallback to scanning all data (disable advanced features)
        all_proteins = []
        for name in os.listdir(args.data_dir):
            full_path = os.path.join(args.data_dir, name)
            if (os.path.isdir(full_path) and name.endswith('_evoformer_blocks') and
                    os.path.isdir(os.path.join(full_path, 'recycle_0'))):
                protein_id = name.replace('_evoformer_blocks', '')
                all_proteins.append(protein_id)

        train_dataset = sorted(all_proteins)
        val_dataset = []

        if args.max_residues is not None:
            print(f"\n📏 Applying size filter (max {args.max_residues} residues)...")
            train_dataset, train_oversized = filter_proteins_by_size(
                train_dataset, args.data_dir, args.max_residues
            )
            print(f"📊 Training: {len(train_dataset)} ({len(train_oversized)} filtered)")

        print("⚠️  LR scheduling and early stopping disabled (no validation set)")

    dataset = train_dataset  # For compatibility with existing code
    print(f"📊 Final proteins to process: {len(dataset)}")

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

    # Initialize LR scheduler and early stopping (only if we have validation data)
    lr_scheduler = None
    early_stopping = None

    if val_dataset:
        lr_scheduler = LearningRateScheduler(
            optimizer,
            patience=args.lr_patience,
            factor=args.lr_factor,
            min_lr=args.min_lr,
            verbose=True
        )

        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=args.restore_best_weights,
            verbose=True
        )

        print(f"📉 LR Scheduler initialized: ReduceLROnPlateau")
        print(f"🛑 Early Stopping initialized: patience={args.early_stopping_patience}")
    else:
        print("⚠️  Running without LR scheduling and early stopping (no validation set)")

    # Initialize training logger
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
                'mode': args.mode,
                'train_proteins': len(train_dataset),
                'val_proteins': len(val_dataset),
                'lr_scheduling': lr_scheduler is not None,
                'early_stopping': early_stopping is not None
            }

            optimizer_info = {
                'learning_rate': args.learning_rate,
                'lr_patience': args.lr_patience if lr_scheduler else 'N/A',
                'lr_factor': args.lr_factor if lr_scheduler else 'N/A',
                'min_lr': args.min_lr if lr_scheduler else 'N/A',
                'early_stopping_patience': args.early_stopping_patience if early_stopping else 'N/A'
            }

            logger.log_configuration(args, model_info, optimizer_info)
            print(f"📝 Training logger initialized: {args.experiment_name}")
            print(f"📊 Reports will be saved to: {args.output_dir}/{args.experiment_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize logger: {e}")
            logger = None

    # Training loop
    previous_losses = []
    training_start_time = time.time()

    # FIXED: Log training start time
    if logger:
        logger.log_training_start()

    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")

        # Show current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🎛️  Current learning rate: {current_lr:.2e}")

        # Log epoch start
        if logger:
            logger.log_epoch_start(epoch + 1, args.epochs, dataset)

        epoch_losses = []
        successful_proteins = 0
        epoch_start_time = time.time()

        for protein_idx, protein_id in enumerate(dataset):
            print(f"  [{protein_idx + 1}/{len(dataset)}] Processing {protein_id}... ", end='', flush=True)

            protein_start_time = time.time()

            # Train using batched approach
            step_info = train_single_protein_batched(protein_id, ode_func, optimizer, scaler, args)

            protein_time = time.time() - protein_start_time
            epoch_losses.append(step_info['total_loss'])
            successful_proteins += 1
            print(f"✅ Loss: {step_info['total_loss']:.3f}")

            # Log protein step
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
                print(f"      📦 Batches: {step_info['num_batches']}, batch_size: {step_info['batch_size']}")
                print(f"      📊 Using {step_info['num_blocks']} blocks")

            # Memory cleanup after each protein
            clear_memory(args.device)
            if args.device == 'cuda':
                torch.cuda.synchronize()

        # Run validation after each epoch if we have validation data
        val_results = None
        if val_dataset:
            print(f"\n🔍 Running validation on {len(val_dataset)} proteins...")
            val_start_time = time.time()
            val_results = validate_model(ode_func, val_dataset, args)
            val_time = time.time() - val_start_time

            print(f"📊 Validation Summary:")
            print(f"    Loss: {val_results['avg_loss']:.3f}")
            print(f"    Range: [{val_results['min_loss']:.3f}, {val_results['max_loss']:.3f}]")
            print(f"    Successful: {val_results['successful_validations']}/{len(val_dataset)}")
            print(f"    Time: {val_time:.1f}s")

        # Log epoch end
        if logger:
            logger.log_epoch_end(val_results)

        # Compute epoch statistics
        if epoch_losses:
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start_time

            print(f"📊 Epoch {epoch + 1} Summary:")
            print(f"    Training Loss: {avg_train_loss:.3f}")
            if val_results:
                print(f"    Validation Loss: {val_results['avg_loss']:.3f}")
                val_success_rate = val_results['successful_validations'] / len(val_dataset) * 100
                print(
                    f"    Val Success Rate: {val_success_rate:.1f}% ({val_results['successful_validations']}/{len(val_dataset)})")
            print(f"    Successful proteins: {successful_proteins}/{len(dataset)}")
            print(f"    Loss range: [{min(epoch_losses):.3f}, {max(epoch_losses):.3f}]")
            print(f"    Epoch time: {epoch_time:.1f} seconds")

            # Show training progress compared to previous epoch
            if epoch > 0 and len(previous_losses) > 0:
                prev_avg = sum(previous_losses) / len(previous_losses)
                improvement = (prev_avg - avg_train_loss) / prev_avg * 100
                if improvement > 0:
                    print(f"    📈 Training improvement: {improvement:.1f}% better than previous epoch")
                else:
                    print(f"    📉 Training change: {abs(improvement):.1f}% worse than previous epoch")

            # Update learning rate scheduler (if validation data available)
            if lr_scheduler is not None and val_results is not None:
                lr_reduced = lr_scheduler.step(val_results['avg_loss'], epoch + 1)
                if lr_reduced:
                    new_lr = lr_scheduler.get_last_lr()
                    print(f"    🔽 Learning rate updated: {new_lr:.2e}")

            # Check early stopping (if validation data available)
            if early_stopping is not None and val_results is not None:
                should_stop = early_stopping(val_results['avg_loss'], epoch + 1, ode_func)
                if should_stop:
                    total_time = time.time() - training_start_time
                    print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs!")
                    print(f"⏱️  Total training time: {total_time / 60:.1f} minutes")
                    print(
                        f"🎯 Best validation loss: {early_stopping.get_best_loss():.4f} (epoch {early_stopping.get_best_epoch()})")
                    break

            previous_losses = epoch_losses[:]

            # Show total training time so far
            total_time = time.time() - training_start_time
            print(f"    ⏱️  Total time so far: {total_time / 60:.1f} minutes")

    # Training completion
    total_training_time = time.time() - training_start_time
    print(f"\n🎯 Training completed!")
    print(f"⏱️  Total training time: {total_training_time / 60:.1f} minutes")

    if early_stopping is not None:
        print(f"🏆 Best validation loss: {early_stopping.get_best_loss():.4f} (epoch {early_stopping.get_best_epoch()})")

    if lr_scheduler is not None:
        print(f"📉 Learning rate reductions: {lr_scheduler.lr_reductions}")
        print(f"🎛️  Final learning rate: {lr_scheduler.get_last_lr():.2e}")

    # Save final model and close logger
    if logger:
        # Save final model
        model_path = os.path.join(args.output_dir, f"{args.experiment_name}_final_model.pt")

        # Add scheduler and early stopping info to saved model
        save_dict = {
            'model_state_dict': ode_func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(args),
            'training_stats': {
                'total_epochs': epoch + 1,
                'total_time_minutes': total_training_time / 60,
                'final_train_loss': avg_train_loss if epoch_losses else None,
                'final_val_loss': val_results['avg_loss'] if val_results else None,
                'best_val_loss': early_stopping.get_best_loss() if early_stopping else None,
                'best_val_epoch': early_stopping.get_best_epoch() if early_stopping else None,
                'lr_reductions': lr_scheduler.lr_reductions if lr_scheduler else 0,
                'final_lr': lr_scheduler.get_last_lr() if lr_scheduler else args.learning_rate,
                'early_stopped': early_stopping.should_stop if early_stopping else False
            }
        }

        torch.save(save_dict, model_path)

        logger.log_training_complete(model_path)
        print(f"📊 Training complete! Full report saved to: {args.output_dir}/{args.experiment_name}.txt")
        print(f"🤖 Final model saved to: {model_path}")


if __name__ == "__main__":
    main()