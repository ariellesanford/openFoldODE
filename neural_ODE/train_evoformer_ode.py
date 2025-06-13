"""
Simplified Neural ODE training for Evoformer using only blocks 0 and 48
Uses adjoint method from torchdiffeq with memory optimizations
MODIFIED: Support multiple data directories
"""

import os
import gc
import torch
import argparse
import sys
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
from training_logger import TrainingLogger


def aggressive_memory_cleanup():
    """Aggressive GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for _ in range(3):
            torch.cuda.empty_cache()


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

        if self.patience_counter >= self.patience and current_lr > self.min_lr:
            new_lr = max(current_lr * self.factor, self.min_lr)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            self.lr_reductions += 1
            self.last_lr_reduction = epoch
            self.patience_counter = 0

            if self.verbose:
                print(f"🔽 Learning rate reduced: {current_lr:.2e} → {new_lr:.2e} (reduction #{self.lr_reductions})")

            return True

        return False

    def get_last_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """Early stopping based on validation loss"""

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

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0

            if model is not None and self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()

        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True

            if self.verbose:
                print(f"🛑 Early stopping triggered!")
                print(f"   No improvement for {self.patience} epochs")
                print(f"   Best validation loss: {self.best_loss:.4f} (epoch {self.best_epoch})")

            if self.restore_best_weights and self.best_model_state is not None and model is not None:
                model.load_state_dict(self.best_model_state)
                if self.verbose:
                    print(f"🔄 Restored best model weights from epoch {self.best_epoch}")

        return self.should_stop

    def get_best_loss(self):
        return self.best_loss

    def get_best_epoch(self):
        return self.best_epoch


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


def get_available_proteins_multi_dir(data_dirs: List[str], splits_dir: str, mode: str) -> List[Tuple[str, str]]:
    """Get list of available protein IDs and their data directories for the specified mode"""
    split_proteins = load_split_proteins(splits_dir, mode)

    available_proteins = []
    for protein_id in split_proteins:
        # Search all data directories for this protein
        for data_dir in data_dirs:
            protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
            if os.path.isdir(protein_dir):
                available_proteins.append((protein_id, data_dir))
                break  # Found it, don't search other directories

    return available_proteins


def get_train_val_datasets_multi_dir(data_dirs: List[str], splits_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Get training and validation datasets with their data directories"""
    train_proteins = get_available_proteins_multi_dir(data_dirs, splits_dir, 'training')
    val_proteins = get_available_proteins_multi_dir(data_dirs, splits_dir, 'validation')
    return train_proteins, val_proteins


def filter_proteins_by_size_multi_dir(proteins: List[Tuple[str, str]], max_residues: int = None) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Filter proteins by residue count, keeping (protein_id, data_dir) tuples"""
    if max_residues is None:
        return proteins, []

    valid_proteins = []
    oversized_proteins = []
    for protein_id, data_dir in proteins:
        try:
            protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
            m_path = os.path.join(protein_dir, "m_block_0.pt")

            if os.path.exists(m_path):
                m_test = torch.load(m_path, map_location='cpu')
                num_residues = m_test.shape[-2] if m_test.dim() == 4 else m_test.shape[-2]
                del m_test

                if num_residues <= max_residues:
                    valid_proteins.append((protein_id, data_dir))
                else:
                    oversized_proteins.append(protein_id)
            else:
                oversized_proteins.append(protein_id)

        except Exception as e:
            oversized_proteins.append(protein_id)
    return valid_proteins, oversized_proteins


def load_protein_blocks_sequential(protein_id: str, data_dir: str, device: str, max_cluster_size: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load M and Z tensors sequentially to minimize memory spikes"""
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

    # Load block 0 - one tensor at a time
    m0_path = os.path.join(protein_dir, "m_block_0.pt")
    m0 = torch.load(m0_path, map_location='cpu')
    if m0.dim() == 4:
        m0 = m0.squeeze(0)
    if max_cluster_size and m0.shape[0] > max_cluster_size:
        m0 = m0[:max_cluster_size]
    m0 = m0.to(device)

    aggressive_memory_cleanup()

    z0_path = os.path.join(protein_dir, "z_block_0.pt")
    z0 = torch.load(z0_path, map_location='cpu')
    if z0.dim() == 4:
        z0 = z0.squeeze(0)
    z0 = z0.to(device)

    aggressive_memory_cleanup()

    # Load block 48
    m48_path = os.path.join(protein_dir, "m_block_48.pt")
    m48 = torch.load(m48_path, map_location='cpu')
    if m48.dim() == 4:
        m48 = m48.squeeze(0)
    if max_cluster_size and m48.shape[0] > max_cluster_size:
        m48 = m48[:max_cluster_size]
    m48 = m48.to(device)

    aggressive_memory_cleanup()

    z48_path = os.path.join(protein_dir, "z_block_48.pt")
    z48 = torch.load(z48_path, map_location='cpu')
    if z48.dim() == 4:
        z48 = z48.squeeze(0)
    z48 = z48.to(device)

    return m0, z0, m48, z48


def load_protein_blocks_standard(protein_id: str, data_dir: str, device: str, max_cluster_size: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load M and Z tensors for blocks 0 and 48 (standard method)"""
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

    # Load block 0
    m0_path = os.path.join(protein_dir, "m_block_0.pt")
    z0_path = os.path.join(protein_dir, "z_block_0.pt")
    m0 = torch.load(m0_path, map_location=device)
    z0 = torch.load(z0_path, map_location=device)

    # Load block 48
    m48_path = os.path.join(protein_dir, "m_block_48.pt")
    z48_path = os.path.join(protein_dir, "z_block_48.pt")
    m48 = torch.load(m48_path, map_location=device)
    z48 = torch.load(z48_path, map_location=device)

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
                          pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
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

    return total_loss

def print_gpu_memory(step_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
        print(f"  {step_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class StepCounter:
    def __init__(self, ode_func):
        self.ode_func = ode_func
        self.step_count = 0

    def __call__(self, t, state):
        self.step_count += 1
        return self.ode_func(t, state)

def train_single_protein(protein_tuple: Tuple[str, str], ode_func: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         scaler: GradScaler, args: argparse.Namespace) -> Dict:
    """Train using 0→48 transformation with adjoint method"""

    protein_id, data_dir = protein_tuple


    # Choose loading method based on config
    if getattr(args, 'use_sequential_loading', False):
        load_func = load_protein_blocks_sequential
    else:
        load_func = load_protein_blocks_standard

    # Load initial and final states
    m0, z0, m48, z48 = load_func(protein_id, data_dir, args.device, args.reduced_cluster_size)
    m0, z0, m48, z48 = load_func(protein_id, data_dir, args.device, args.reduced_cluster_size)

    print(f"m0 shape: {m0.shape}, size: {m0.numel() * 4 / 1024 ** 2:.1f}MB")
    print(f"z0 shape: {z0.shape}, size: {z0.numel() * 4 / 1024 ** 2:.1f}MB")
    print(f"Combined state size: {(m0.numel() + z0.numel()) * 4 / 1024 ** 2:.1f}MB")

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        # Solve ODE from 0 to 1 using adjoint method
        print_gpu_memory("Before odeint")
        trajectory = odeint(
            ode_func,  # Wrapped function
            (m0, z0),
            torch.tensor([0.0, 1.0]).to(args.device),
            method=args.integrator,
            rtol=1e-4,
            atol=1e-5
        )

        # Get final prediction
        m_pred = trajectory[0][-1]
        z_pred = trajectory[1][-1]

        # Delete trajectory immediately to save memory
        del trajectory
        if getattr(args, 'aggressive_cleanup', False):
            aggressive_memory_cleanup()

        # Compute loss
        loss = compute_adaptive_loss(m_pred, m48, z_pred, z48)

    # Backward pass (uses adjoint method automatically)
    print_gpu_memory("Before backward")
    if args.use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
        optimizer.step()

    loss_value = loss.item()

    # Clean up all tensors
    del m0, z0, m48, z48, m_pred, z_pred, loss
    if getattr(args, 'aggressive_cleanup', False):
        aggressive_memory_cleanup()


    return {
        'protein': protein_id,
        'loss': loss_value
    }

def validate_single_protein(protein_tuple: Tuple[str, str], ode_func: torch.nn.Module, args: argparse.Namespace) -> Dict:
    """Validate using 0→48 transformation"""
    protein_id, data_dir = protein_tuple  # Unpack the tuple

    # Choose loading method based on config
    if getattr(args, 'use_sequential_loading', False):
        load_func = load_protein_blocks_sequential
    else:
        load_func = load_protein_blocks_standard

    # Load initial and final states (use the specific data_dir for this protein)
    m0, z0, m48, z48 = load_func(protein_id, data_dir, args.device, args.reduced_cluster_size)

    with torch.no_grad():
        # Solve ODE from 0 to 1
        trajectory = odeint(
            ode_func,
            (m0, z0),
            torch.tensor([0.0, 1.0]).to(args.device),
            method=args.integrator,
            rtol=1e-4,
            atol=1e-5
        )

        # Get final prediction
        m_pred = trajectory[0][-1]
        z_pred = trajectory[1][-1]

        # Delete trajectory immediately
        del trajectory
        if getattr(args, 'aggressive_cleanup', False):
            aggressive_memory_cleanup()

        # Compute loss
        loss = compute_adaptive_loss(m_pred, m48, z_pred, z48)

    loss_value = loss.item()

    # Clean up
    del m0, z0, m48, z48, m_pred, z_pred, loss
    if getattr(args, 'aggressive_cleanup', False):
        aggressive_memory_cleanup()

    return {
        'protein': protein_id,
        'loss': loss_value
    }


def validate_model(ode_func: torch.nn.Module, val_proteins: List[Tuple[str, str]], args: argparse.Namespace) -> Dict:
    """Run validation on validation set"""
    ode_func.eval()
    val_losses = []
    successful_validations = 0

    with torch.no_grad():
        for val_idx, protein_tuple in enumerate(val_proteins):
            protein_id = protein_tuple[0]  # Extract protein_id for display
            print(f"    [{val_idx + 1}/{len(val_proteins)}] Validating {protein_id}... ", end='', flush=True)

            try:
                result = validate_single_protein(protein_tuple, ode_func, args)
                val_losses.append(result['loss'])
                successful_validations += 1
                print(f"✅ Loss: {result['loss']:.5f}")

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                continue

            if getattr(args, 'aggressive_cleanup', False):
                aggressive_memory_cleanup()

    ode_func.train()

    if val_losses:
        return {
            'avg_loss': sum(val_losses) / len(val_losses),
            'min_loss': min(val_losses),
            'max_loss': max(val_losses),
            'num_proteins': len(val_losses),
            'successful_validations': successful_validations,
        }
    else:
        return {
            'avg_loss': float('inf'),
            'num_proteins': 0,
            'successful_validations': 0,
        }


def main():
    parser = argparse.ArgumentParser(description='Neural ODE Training with Memory Optimizations')

    # Core settings
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True,
                        help='Data directories (can specify multiple)')
    parser.add_argument('--splits_dir', type=str, default=None, help='Directory containing split files')
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

    # Optimization settings
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--max_residues', type=int, default=None,
                        help='Skip proteins with more than N residues (omit for no limit)')
    parser.add_argument('--max_time_hours', type=float, default=None,
                        help='Maximum training time in hours (omit for no limit)')

    # Memory optimization settings
    parser.add_argument('--use_sequential_loading', action='store_true',
                        help='Load tensors sequentially to reduce memory spikes')
    parser.add_argument('--aggressive_cleanup', action='store_true',
                        help='Use aggressive memory cleanup between operations')

    args = parser.parse_args()

    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'

    if args.device == 'cpu':
        args.use_amp = False

    print(f"🚀 Starting Neural ODE Training with Memory Optimizations")
    print(f"📁 Data directories: {args.data_dirs}")
    if args.splits_dir:
        print(f"📂 Splits directory: {args.splits_dir}")
    print(f"💻 Device: {args.device}")
    print(f"🔧 Settings: LR={args.learning_rate}, Fast ODE={args.use_fast_ode}, AMP={args.use_amp}")
    print(f"🧮 Method: Adjoint method for backpropagation")
    print(f"💾 Memory: Sequential loading={args.use_sequential_loading}, Aggressive cleanup={args.aggressive_cleanup}")
    if args.max_time_hours is not None:
        print(f"⏰ Time limit: {args.max_time_hours} hours")

    # Initialize CUDA if using GPU
    if args.device == 'cuda':
        torch.cuda.init()
        print(f"🚀 CUDA initialized: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Load dataset
    if args.splits_dir:
        train_dataset_raw, val_dataset_raw = get_train_val_datasets_multi_dir(args.data_dirs, args.splits_dir)
        if not train_dataset_raw:
            print(f"❌ No training proteins found in {args.data_dirs}")
            return

        # Convert back to just protein IDs for display
        train_protein_ids = [protein_id for protein_id, _ in train_dataset_raw]
        val_protein_ids = [protein_id for protein_id, _ in val_dataset_raw]

        print(f"🧬 Raw training proteins: {len(train_protein_ids)}")
        print(f"🔍 Raw validation proteins: {len(val_protein_ids)}")

        # Filter by size if args.max_residues is specified
        if args.max_residues is not None:
            print(f"\n📏 Applying size filter (max {args.max_residues} residues)...")

            train_dataset, train_oversized = filter_proteins_by_size_multi_dir(
                train_dataset_raw, args.max_residues
            )
            val_dataset, val_oversized = filter_proteins_by_size_multi_dir(
                val_dataset_raw, args.max_residues
            )

            print(f"\n📊 Final dataset sizes:")
            print(f"  Training: {len(train_dataset)} / {len(train_dataset_raw)} ({len(train_oversized)} filtered)")
            print(f"  Validation: {len(val_dataset)} / {len(val_dataset_raw)} ({len(val_oversized)} filtered)")

            if not train_dataset:
                print(f"❌ No training proteins remain after size filtering!")
                return
        else:
            print(f"\n📏 No size filter applied (processing all proteins)")
            train_dataset = train_dataset_raw
            val_dataset = val_dataset_raw

        if not val_dataset:
            print(f"❌ No validation proteins found - required for LR scheduling and early stopping!")
            return

    else:
        # Fallback to scanning all data directories
        all_proteins = []
        for data_dir in args.data_dirs:
            for name in os.listdir(data_dir):
                full_path = os.path.join(data_dir, name)
                if (os.path.isdir(full_path) and name.endswith('_evoformer_blocks') and
                        os.path.isdir(os.path.join(full_path, 'recycle_0'))):
                    protein_id = name.replace('_evoformer_blocks', '')
                    all_proteins.append((protein_id, data_dir))
                    break  # Only take first occurrence

        train_dataset = sorted(all_proteins)
        val_dataset = []

        if args.max_residues is not None:
            train_dataset, train_oversized = filter_proteins_by_size_multi_dir(
                train_dataset, args.max_residues
            )
            print(f"📏 Size filter applied: {len(train_oversized)} proteins filtered out")
        else:
            print(f"📏 No size filter applied")

        print("⚠️  LR scheduling and early stopping disabled (no validation set)")

    print(f"📊 Final proteins to process: {len(train_dataset)}")

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

    # Initialize LR scheduler and early stopping
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

    else:
        print("⚠️  Running without LR scheduling and early stopping (no validation set)")

    # Initialize training logger
    logger = None
    if args.experiment_name and args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logger = TrainingLogger(args.output_dir, args.experiment_name)

            model_info = {
                'total_params': sum(p.numel() for p in ode_func.parameters()),
                'model_type': 'EvoformerODEFuncFast' if args.use_fast_ode else 'EvoformerODEFunc',
                'loss_function': 'Adaptive MSE (0→48 only)',
                'train_proteins': len(train_dataset),
                'val_proteins': len(val_dataset),
                'lr_scheduling': lr_scheduler is not None,
                'early_stopping': early_stopping is not None,
                'adjoint_method': True,
                'memory_optimizations': {
                    'sequential_loading': args.use_sequential_loading,
                    'aggressive_cleanup': args.aggressive_cleanup
                }
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
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize logger: {e}")
            logger = None

    # Training loop with time limit handling
    previous_losses = []
    training_start_time = time.time()
    final_epoch = 0
    interrupted_by_timeout = False
    max_time_seconds = args.max_time_hours * 3600 if args.max_time_hours is not None else None

    if logger:
        logger.log_training_start()

    for epoch in range(args.epochs):
        final_epoch = epoch + 1

        # Check time limit before starting epoch
        if max_time_seconds is not None:
            elapsed_time = time.time() - training_start_time
            if elapsed_time >= max_time_seconds:
                interrupted_by_timeout = True
                print(f"\n⏰ Time limit reached ({args.max_time_hours} hours). Stopping training...")
                break

        print(f"\n📈 Epoch {epoch + 1}/{args.epochs}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"🎛️  Current learning rate: {current_lr:.2e}")

        if logger:
            # Convert tuples back to protein IDs for logging
            train_protein_ids = [protein_id for protein_id, _ in train_dataset]
            logger.log_epoch_start(epoch + 1, args.epochs, train_protein_ids)

        epoch_losses = []
        successful_proteins = 0
        epoch_start_time = time.time()

        for protein_idx, protein_tuple in enumerate(train_dataset):
            protein_id = protein_tuple[0]  # Extract protein_id for display

            # cleanup memory at the start of each protein processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()

            # Check time limit during epoch
            if max_time_seconds is not None:
                elapsed_time = time.time() - training_start_time
                if elapsed_time >= max_time_seconds:
                    interrupted_by_timeout = True
                    print(f"\n⏰ Time limit reached during epoch {epoch + 1}. Stopping training...")
                    break

            print(f"  [{protein_idx + 1}/{len(train_dataset)}] Processing {protein_id}... ", end='', flush=True)

            protein_start_time = time.time()

            try:
                result = train_single_protein(protein_tuple, ode_func, optimizer, scaler, args)
                epoch_losses.append(result['loss'])
                successful_proteins += 1

                print(f"✅ Loss: {result['loss']:.5f}")

                if logger:
                    protein_time = time.time() - protein_start_time
                    logger.log_protein_step(
                        protein_id=protein_id,
                        step_idx=protein_idx,
                        loss=result['loss'],
                        step_info={'approach': 'adjoint_0_to_48', 'num_blocks': 48},
                        time_taken=protein_time
                    )

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                continue

            if args.aggressive_cleanup:
                aggressive_memory_cleanup()

        # Break out of epoch loop if timeout occurred
        if interrupted_by_timeout:
            break

        # Run validation
        val_results = None
        if val_dataset:
            print(f"\n🔍 Running validation on {len(val_dataset)} proteins...")
            val_start_time = time.time()
            val_results = validate_model(ode_func, val_dataset, args)
            val_time = time.time() - val_start_time

            print(f"📊 Validation Summary:")
            print(f"    Average Loss: {val_results['avg_loss']:.5f}")
            print(f"    Range: [{val_results['min_loss']:.5f}, {val_results['max_loss']:.5f}]")
            print(f"    Successful: {val_results['successful_validations']}/{len(val_dataset)}")
            print(f"    Time: {val_time:.1f}s")

        if logger:
            logger.log_epoch_end(val_results)

        # Compute epoch statistics
        if epoch_losses:
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start_time

            print(f"📊 Epoch {epoch + 1} Summary:")
            print(f"    Training Loss: {avg_train_loss:.5f}")
            print(f"    Training Success: {successful_proteins}/{len(train_dataset)} proteins")
            if val_results:
                print(f"    Validation Loss: {val_results['avg_loss']:.5f}")
                print(f"    Validation Success: {val_results['successful_validations']}/{len(val_dataset)} proteins")
            print(f"    Epoch time: {epoch_time:.1f} seconds")

            # Update learning rate scheduler
            if lr_scheduler is not None and val_results is not None:
                lr_reduced = lr_scheduler.step(val_results['avg_loss'], epoch + 1)
                if lr_reduced:
                    new_lr = lr_scheduler.get_last_lr()
                    print(f"    🔽 Learning rate updated: {new_lr:.2e}")

            # Check early stopping
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

            total_time = time.time() - training_start_time
            print(f"    ⏱️  Total time so far: {total_time / 60:.1f} minutes")
            if max_time_seconds is not None:
                remaining_time = max_time_seconds - total_time
                print(f"    ⏰ Time remaining: {remaining_time / 60:.1f} minutes")

    # Training completion
    total_training_time = time.time() - training_start_time

    if interrupted_by_timeout:
        print(f"\n⏰ Training stopped due to time limit after {final_epoch} epochs!")
        print(f"⏱️  Total training time: {total_training_time / 60:.1f} minutes")
        print("💾 Saving best model...")
    else:
        print(f"\n🎯 Training completed!")
        print(f"⏱️  Total training time: {total_training_time / 60:.1f} minutes")

        if early_stopping is not None:
            print(
                f"🏆 Best validation loss: {early_stopping.get_best_loss():.4f} (epoch {early_stopping.get_best_epoch()})")

        if lr_scheduler is not None:
            print(f"📉 Learning rate reductions: {lr_scheduler.lr_reductions}")
            print(f"🎛️  Final learning rate: {lr_scheduler.get_last_lr():.2e}")

    # Save final model and close logger
    if logger:
        # Mark as interrupted if applicable
        if interrupted_by_timeout:
            logger.interrupted_at_epoch = final_epoch

        # Save final model (use best weights if timeout and early stopping is available)
        model_path = os.path.join(args.output_dir, f"{args.experiment_name}_final_model.pt")

        # If interrupted by timeout and we have early stopping with best weights, use those
        if interrupted_by_timeout and early_stopping is not None and early_stopping.best_model_state is not None:
            print(f"🔄 Using best model weights from epoch {early_stopping.get_best_epoch()}")
            ode_func.load_state_dict(early_stopping.best_model_state)

        save_dict = {
            'model_state_dict': ode_func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(args),
            'training_stats': {
                'total_epochs': final_epoch,
                'total_time_minutes': total_training_time / 60,
                'final_train_loss': avg_train_loss if 'avg_train_loss' in locals() and epoch_losses else None,
                'final_val_loss': val_results['avg_loss'] if 'val_results' in locals() and val_results else None,
                'best_val_loss': early_stopping.get_best_loss() if early_stopping else None,
                'best_val_epoch': early_stopping.get_best_epoch() if early_stopping else None,
                'lr_reductions': lr_scheduler.lr_reductions if lr_scheduler else 0,
                'final_lr': lr_scheduler.get_last_lr() if lr_scheduler else args.learning_rate,
                'early_stopped': early_stopping.should_stop if early_stopping else False,
                'interrupted_by_timeout': interrupted_by_timeout,
                'interrupted_at_epoch': final_epoch if interrupted_by_timeout else None,
                'max_time_hours': args.max_time_hours,
                'method': 'adjoint_0_to_48',
                'memory_optimizations': {
                    'sequential_loading': args.use_sequential_loading,
                    'aggressive_cleanup': args.aggressive_cleanup
                }
            }
        }

        torch.save(save_dict, model_path)

        logger.log_training_complete(model_path)
        status = 'interrupted by timeout' if interrupted_by_timeout else 'complete'
        print(f"📊 Training {status}! Full report saved to: {args.output_dir}/{args.experiment_name}.txt")
        print(f"🤖 Final model saved to: {model_path}")

if __name__ == "__main__":
    main()