"""
Restructured Neural ODE training for Evoformer - Shorter and cleaner version
Uses adjoint method from torchdiffeq with memory optimizations
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
from typing import Dict, List, Tuple, Optional, Union
from training_logger import TrainingLogger
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Unified configuration object"""
    data_dirs: List[str]
    splits_dir: str
    output_dir: str
    experiment_name: str
    device: str = 'cuda'
    epochs: int = 20
    learning_rate: float = 1e-3
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001
    use_fast_ode: bool = False
    loss: str = 'default'
    reduced_cluster_size: int = 64
    hidden_dim: int = 64
    integrator: str = 'rk4'
    use_amp: bool = False
    max_residues: Optional[int] = None
    max_time_hours: Optional[float] = None
    aggressive_cleanup: bool = False

    # Preliminary training
    enable_preliminary_training: bool = False
    prelim_data_dir: Optional[str] = None
    prelim_block_stride: int = 4
    prelim_max_epochs: int = 20
    prelim_chunk_size: int = 4

    @classmethod
    def from_args(cls, args):
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__annotations__})

    def validate(self):
        """Validate configuration"""
        if self.enable_preliminary_training:
            required = ['prelim_data_dir', 'prelim_block_stride', 'prelim_max_epochs']
            missing = [attr for attr in required if getattr(self, attr) is None]
            if missing:
                raise ValueError(f"Preliminary training requires: {missing}")
            if not os.path.exists(self.prelim_data_dir):
                raise ValueError(f"Preliminary data directory not found: {self.prelim_data_dir}")


class MemoryManager:
    """Context manager for memory cleanup"""
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.aggressive:
            self.cleanup()

    @staticmethod
    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for _ in range(3):
                torch.cuda.empty_cache()


class LearningRateScheduler:
    """Smart learning rate scheduler"""
    def __init__(self, optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.lr_reductions = 0

    def step(self, val_loss, epoch):
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
            self.patience_counter = 0
            if self.verbose:
                print(f"🔽 Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            return True
        return False

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def reset(self):
        """Reset scheduler state while preserving optimizer"""
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.lr_reductions = 0
        # Optionally reset LR to initial value
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr  # Store this in __init__


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=7, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, epoch, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            if model is not None:
                self.best_model_state = model.state_dict().copy()
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(f"🛑 Early stopping triggered! Best: {self.best_loss:.4f} (epoch {self.best_epoch})")
            if self.best_model_state is not None and model is not None:
                model.load_state_dict(self.best_model_state)
                if self.verbose:
                    print(f"🔄 Restored best model weights from epoch {self.best_epoch}")
        return self.should_stop


class DataManager:
    """Unified data loading and management"""

    @staticmethod
    def load_split_proteins(splits_dir: str, mode: str) -> List[str]:
        split_files = {'training': 'training_chains.txt', 'validation': 'validation_chains.txt', 'testing': 'testing_chains.txt'}
        split_file = os.path.join(splits_dir, split_files[mode])
        with open(split_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def get_available_proteins(data_dirs: List[str], splits_dir: str, mode: str, single_dir: bool = False) -> Union[List[Tuple[str, str]], List[str]]:
        """Unified function for protein discovery"""
        split_proteins = DataManager.load_split_proteins(splits_dir, mode)

        if single_dir:
            # For preliminary training (single directory)
            data_dir = data_dirs[0] if isinstance(data_dirs, list) else data_dirs
            available = []
            for protein_id in split_proteins:
                protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
                if os.path.isdir(protein_dir):
                    available.append(protein_id)
            return available
        else:
            # For main training (multi-directory)
            available = []
            for protein_id in split_proteins:
                for data_dir in data_dirs:
                    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
                    if os.path.isdir(protein_dir):
                        available.append((protein_id, data_dir))
                        break
            return available

    @staticmethod
    def filter_proteins_by_size(proteins: Union[List[str], List[Tuple[str, str]]],
                                max_residues: int, data_dir: str = None) -> Union[List[str], List[Tuple[str, str]]]:
        """Unified size filtering"""
        if max_residues is None:
            return proteins

        print(f"🔍 Filtering {len(proteins)} proteins by size (max: {max_residues} residues)")
        valid_proteins = []

        for i, protein in enumerate(proteins):
            try:
                if isinstance(protein, tuple):
                    protein_id, protein_data_dir = protein
                else:
                    protein_id, protein_data_dir = protein, data_dir

                protein_dir = os.path.join(protein_data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
                m_path = os.path.join(protein_dir, "m_block_0.pt")

                if os.path.exists(m_path):
                    m_test = torch.load(m_path, map_location='cpu')
                    num_residues = m_test.shape[-2]
                    del m_test

                    if num_residues <= max_residues:
                        valid_proteins.append(protein)
                        print(f"  ✅ {protein_id}: {num_residues} residues")
                    else:
                        print(f"  ❌ {protein_id}: {num_residues} residues (too large)")
            except Exception:
                print(f"  ⚠️  {protein_id}: Error loading")
                continue

        print(f"📊 Kept {len(valid_proteins)}/{len(proteins)} proteins")
        return valid_proteins

    @staticmethod
    def load_protein_blocks(protein_id: str, data_dir: str, device: str, max_cluster_size: int = None,
                           blocks: List[int] = [0, 48], sequential: bool = False,
                           target_block: int = None, strided: bool = False, **kwargs):
        """Unified loading function for all cases"""
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

        if target_block is not None:
            # Load single target block
            m_path = os.path.join(protein_dir, f"m_block_{target_block}.pt")
            z_path = os.path.join(protein_dir, f"z_block_{target_block}.pt")
            m = torch.load(m_path, map_location='cpu')
            z = torch.load(z_path, map_location='cpu')

            if m.dim() == 4:
                m = m.squeeze(0)
            if z.dim() == 4:
                z = z.squeeze(0)
            if max_cluster_size and m.shape[0] > max_cluster_size:
                m = m[:max_cluster_size]

            return m.to(device), z.to(device)

        elif strided:
            # Load for strided training - return initial state and block indices
            available_blocks = []
            max_blocks = kwargs.get('max_blocks', 49)
            for i in range(max_blocks):
                m_path = os.path.join(protein_dir, f"m_block_{i}.pt")
                z_path = os.path.join(protein_dir, f"z_block_{i}.pt")
                if os.path.exists(m_path) and os.path.exists(z_path):
                    available_blocks.append(i)
                else:
                    break

            # Select blocks with stride
            stride = kwargs.get('block_stride', 4)
            selected_blocks = [available_blocks[0]]
            for i in range(stride, len(available_blocks), stride):
                selected_blocks.append(available_blocks[i])
            if available_blocks[-1] not in selected_blocks:
                selected_blocks.append(available_blocks[-1])

            # Load initial block
            m_init, z_init = DataManager.load_protein_blocks(
                protein_id, data_dir, device, max_cluster_size, target_block=selected_blocks[0]
            )
            return m_init, z_init, selected_blocks

        else:
            # Load standard blocks (0 and 48, or custom blocks)
            results = []
            for block in blocks:
                if sequential and len(results) > 0:
                    MemoryManager.cleanup()

                m_path = os.path.join(protein_dir, f"m_block_{block}.pt")
                z_path = os.path.join(protein_dir, f"z_block_{block}.pt")

                m = torch.load(m_path, map_location='cpu' if sequential else device)
                z = torch.load(z_path, map_location='cpu' if sequential else device)

                if m.dim() == 4:
                    m = m.squeeze(0)
                if z.dim() == 4:
                    z = z.squeeze(0)
                if max_cluster_size and m.shape[0] > max_cluster_size:
                    m = m[:max_cluster_size]

                if sequential:
                    m = m.to(device)
                    z = z.to(device)

                results.extend([m, z])

            return tuple(results)


class TrainingPhase:
    """Unified training phase handler"""

    def __init__(self, config: TrainingConfig, is_preliminary: bool = False):
        self.config = config
        self.is_preliminary = is_preliminary
        self.memory_manager = MemoryManager(config.aggressive_cleanup)

    def compute_adaptive_loss(self, pred_m, target_m, pred_z, target_z):
        """Unified loss function with configurable strategy and variance normalization"""

        num_residues = target_m.shape[1] if target_m.dim() == 3 else target_m.shape[0]

        # Choose MSA loss strategy based on config
        if self.config.loss == 'default':
            # Original: full MSA loss
            msa_loss_raw = F.mse_loss(pred_m, target_m)

            # Apply variance normalization
            msa_variance = target_m.var() + 1e-8
            msa_loss = msa_loss_raw / msa_variance

        elif self.config.loss == 'weighted_row':
            # Properly normalized per-row weighting
            num_msa_rows = pred_m.shape[0]

            # Define relative importance (unnormalized)
            first_row_importance = 5.0  # First row is 5x more important
            other_row_importance = 1.0  # Each other row has base importance

            # Calculate total importance
            total_importance = first_row_importance + (num_msa_rows - 1) * other_row_importance

            # Normalize to sum to 1.0
            first_row_weight = first_row_importance / total_importance
            other_row_weight = other_row_importance / total_importance

            # Compute weighted MSA loss with variance normalization
            first_row_loss_raw = F.mse_loss(pred_m[0, :, :], target_m[0, :, :])
            first_row_variance = target_m[0, :, :].var() + 1e-8
            first_row_loss = first_row_loss_raw / first_row_variance

            msa_loss = first_row_weight * first_row_loss

            if num_msa_rows > 1:
                for i in range(1, num_msa_rows):
                    row_loss_raw = F.mse_loss(pred_m[i, :, :], target_m[i, :, :])
                    row_variance = target_m[i, :, :].var() + 1e-8
                    row_loss = row_loss_raw / row_variance
                    msa_loss += other_row_weight * row_loss

        elif self.config.loss == 'single_row':
            # First MSA row only (what structure module uses)
            msa_loss_raw = F.mse_loss(pred_m[0, :, :], target_m[0, :, :])

            # Apply variance normalization
            msa_variance = target_m[0, :, :].var() + 1e-8
            msa_loss = msa_loss_raw / msa_variance

        else:
            raise ValueError(f"Unknown loss strategy: {self.config.loss}")

        # Pair loss (same for all strategies) with variance normalization
        pair_loss_raw = F.mse_loss(pred_z, target_z)
        pair_variance = target_z.var() + 1e-8
        pair_loss = pair_loss_raw / pair_variance

        # Combine losses (now both are variance-normalized)
        base_loss = msa_loss + pair_loss

        # Scale by residues and apply gradient scaling factor
        return base_loss * float(num_residues) / 10


    def process_single_protein(self, protein: Union[str, Tuple[str, str]], model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer = None, scaler: GradScaler = None,
                              training: bool = True) -> Dict:
        """Unified protein processing for training/validation, regular/strided"""

        if isinstance(protein, tuple):
            protein_id, data_dir = protein
        else:
            protein_id, data_dir = protein, self.config.prelim_data_dir

        with self.memory_manager:
            if self.is_preliminary:
                return self._process_protein_strided(protein_id, data_dir, model, optimizer, scaler, training)
            else:
                return self._process_protein_regular(protein_id, data_dir, model, optimizer, scaler, training)

    def _process_protein_regular(self, protein_id: str, data_dir: str, model: torch.nn.Module,
                               optimizer: torch.optim.Optimizer, scaler: GradScaler, training: bool) -> Dict:
        """Regular 0→48 processing"""
        m0, z0, m48, z48 = DataManager.load_protein_blocks(
            protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
            blocks=[0, 48]  # Always use sequential loading (hardcoded in load_protein_blocks)
        )

        if training:
            optimizer.zero_grad()

        with autocast(enabled=self.config.use_amp):
            trajectory = odeint(
                model, (m0, z0), torch.tensor([0.0, 1.0]).to(self.config.device),
                method=self.config.integrator, rtol=1e-4, atol=1e-5
            )

            m_pred, z_pred = trajectory[0][-1], trajectory[1][-1]
            del trajectory

            loss = self.compute_adaptive_loss(m_pred, m48, z_pred, z48)

        if training:
            if self.config.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        loss_value = loss.item()
        del m0, z0, m48, z48, m_pred, z_pred, loss

        return {'protein': protein_id, 'loss': loss_value}

    # def _process_protein_strided(self, protein_id: str, data_dir: str, model: torch.nn.Module,
    #                              optimizer: torch.optim.Optimizer, scaler: GradScaler, training: bool) -> Dict:
    #     """True sequential continuation strided processing with overlapping chunks"""
    #
    #     # Load initial state and selected blocks
    #     m_init, z_init, selected_blocks = DataManager.load_protein_blocks(
    #         protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
    #         strided=True, block_stride=self.config.prelim_block_stride, max_blocks=49
    #     )
    #
    #     if training:
    #         optimizer.zero_grad()
    #
    #     chunk_size = self.config.prelim_chunk_size
    #     total_loss = 0
    #     total_chunks = 0
    #
    #     # Start with initial state
    #     current_m, current_z = m_init, z_init
    #     current_block_idx = selected_blocks[0]  # Track current position (starts at block 0)
    #
    #     # Process overlapping chunks for true sequential continuation
    #     while current_block_idx < selected_blocks[-1]:
    #         # Find next chunk starting from current_block_idx
    #         try:
    #             start_pos = selected_blocks.index(current_block_idx)
    #         except ValueError:
    #             print(f"❌ Current block {current_block_idx} not in selected_blocks")
    #             break
    #
    #         # Create chunk from current position
    #         chunk_end_pos = min(start_pos + chunk_size, len(selected_blocks))
    #         chunk_blocks = selected_blocks[start_pos:chunk_end_pos]
    #
    #         # Skip if chunk is too small
    #         if len(chunk_blocks) <= 1:
    #             print(f"⚠️  Skipping single-block chunk: {chunk_blocks}")
    #             break
    #
    #         # Create time points for this chunk
    #         chunk_times = []
    #         for block_idx in chunk_blocks:
    #             # Map block index to time: block 0 → t=0, block 48 → t=1
    #             t = float(block_idx) / 48.0
    #             chunk_times.append(t)
    #
    #         chunk_time_points = torch.tensor(chunk_times, device=self.config.device)
    #         #print(f"  📦 Chunk {len(chunk_blocks)} blocks: {chunk_blocks} (t={chunk_times[0]:.3f}→{chunk_times[-1]:.3f})")
    #
    #         with autocast(enabled=self.config.use_amp):
    #             # MEMORY FIX: Move current state back to GPU
    #             current_m = current_m.to(self.config.device)
    #             current_z = current_z.to(self.config.device)
    #
    #             # Integrate from current state through this chunk
    #             trajectory = odeint(
    #                 model, (current_m, current_z), chunk_time_points,
    #                 method=self.config.integrator, rtol=1e-4, atol=1e-5
    #             )
    #
    #             # Compute loss for each block in this chunk
    #             chunk_loss = 0
    #             chunk_steps = 0
    #
    #             # Skip the first block only if we're starting from it
    #             # (i.e., don't compare predicted block X to target block X)
    #             for i in range(len(chunk_blocks)):
    #                 block_idx = chunk_blocks[i]
    #
    #                 # Skip if this is the starting block (we're already at this state)
    #                 if block_idx == current_block_idx and i == 0:
    #                     continue
    #
    #                 # Load target for this block
    #                 m_target, z_target = DataManager.load_protein_blocks(
    #                     protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
    #                     target_block=block_idx
    #                 )
    #
    #                 # Compare trajectory prediction to target
    #                 step_loss = self.compute_adaptive_loss(
    #                     trajectory[0][i], m_target,
    #                     trajectory[1][i], z_target
    #                 )
    #
    #                 chunk_loss += step_loss.item() if not training else step_loss
    #                 chunk_steps += 1
    #                 del m_target, z_target
    #
    #             # Average loss for this chunk
    #             if chunk_steps > 0:
    #                 chunk_loss = chunk_loss / chunk_steps
    #
    #                 # MEMORY FIX: Backward pass per chunk instead of accumulating
    #                 if training:
    #                     if self.config.use_amp:
    #                         scaler.scale(chunk_loss).backward()
    #                         scaler.step(optimizer)
    #                         scaler.update()
    #                         optimizer.zero_grad()
    #                     else:
    #                         chunk_loss.backward()
    #                         optimizer.step()
    #                         optimizer.zero_grad()
    #
    #                     total_loss += chunk_loss.item()  # Store scalar only
    #                 else:
    #                     total_loss += chunk_loss.item() if hasattr(chunk_loss, 'item') else chunk_loss
    #
    #                 total_chunks += 1
    #             else:
    #                 print(f"⚠️  Chunk had no valid comparisons")
    #
    #             # SEQUENTIAL CONTINUATION: Update current state to last prediction
    #             # Move to the last block of this chunk
    #             current_block_idx = chunk_blocks[-1]
    #             # MEMORY FIX: Store on CPU to free GPU memory
    #             current_m = trajectory[0][-1].detach().cpu()
    #             current_z = trajectory[1][-1].detach().cpu()
    #
    #             del trajectory
    #             torch.cuda.empty_cache()  # Force cleanup
    #
    #     # Final average loss
    #     if total_chunks > 0:
    #         avg_loss = total_loss / total_chunks
    #     else:
    #         print(f"❌ No valid chunks for protein {protein_id}")
    #         return {'protein': protein_id, 'loss': 0.0, 'error': 'no_valid_chunks'}
    #
    #     # MEMORY FIX: No final backward pass needed since we did per-chunk backward
    #     loss_value = avg_loss
    #
    #     # Cleanup
    #     del m_init, z_init, current_m, current_z
    #
    #     return {
    #         'protein': protein_id,
    #         'loss': loss_value,
    #         'selected_blocks': selected_blocks,
    #         'num_chunks': total_chunks,
    #         'cluster_size': self.config.reduced_cluster_size,
    #         'approach': 'true_sequential_continuation_strided'
    #     }

    def _process_protein_strided(self, protein_id: str, data_dir: str, model: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer, scaler: GradScaler, training: bool) -> Dict:
        """SIMPLIFIED: Back to working approach with just better gradient clipping"""

        # Load initial state and selected blocks
        m_init, z_init, selected_blocks = DataManager.load_protein_blocks(
            protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
            strided=True, block_stride=self.config.prelim_block_stride, max_blocks=49
        )

        if training and optimizer:
            optimizer.zero_grad()

        chunk_size = self.config.prelim_chunk_size
        total_loss = 0
        total_chunks = 0

        # Start with initial state
        current_m, current_z = m_init, z_init
        current_block_idx = selected_blocks[0]

        # Process overlapping chunks for true sequential continuation
        while current_block_idx < selected_blocks[-1]:
            try:
                start_pos = selected_blocks.index(current_block_idx)
            except ValueError:
                break

            chunk_end_pos = min(start_pos + chunk_size, len(selected_blocks))
            chunk_blocks = selected_blocks[start_pos:chunk_end_pos]

            if len(chunk_blocks) <= 1:
                break

            # Create time points for this chunk
            chunk_times = [float(block_idx) / 48.0 for block_idx in chunk_blocks]
            chunk_time_points = torch.tensor(chunk_times, device=self.config.device)

            with autocast(enabled=self.config.use_amp):
                # Ensure state is on correct device
                current_m = current_m.to(self.config.device)
                current_z = current_z.to(self.config.device)

                # Integrate through this chunk
                trajectory = odeint(
                    model, (current_m, current_z), chunk_time_points,
                    method=self.config.integrator, rtol=1e-4, atol=1e-5
                )

                # Compute loss for each prediction in this chunk
                chunk_loss = 0
                chunk_steps = 0

                for i in range(len(chunk_blocks)):
                    block_idx = chunk_blocks[i]

                    # Skip starting block comparison
                    if block_idx == current_block_idx and i == 0:
                        continue

                    # Load target
                    m_target, z_target = DataManager.load_protein_blocks(
                        protein_id, data_dir, self.config.device, self.config.reduced_cluster_size,
                        target_block=block_idx
                    )

                    # Compute step loss
                    step_loss = self.compute_adaptive_loss(
                        trajectory[0][i], m_target,
                        trajectory[1][i], z_target
                    )

                    chunk_loss += step_loss
                    chunk_steps += 1
                    del m_target, z_target

                # Average loss for this chunk
                if chunk_steps > 0:
                    chunk_loss = chunk_loss / chunk_steps

                    # SIMPLIFIED: Immediate backward pass per chunk (like original)
                    if training:
                        if self.config.use_amp:
                            scaler.scale(chunk_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                        else:
                            chunk_loss.backward()
                            # ONLY CHANGE: Better gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                            optimizer.step()
                            optimizer.zero_grad()

                        total_loss += chunk_loss.item()
                    else:
                        total_loss += chunk_loss.item()

                    total_chunks += 1

                # Update current state for next chunk
                current_block_idx = chunk_blocks[-1]
                current_m = trajectory[0][-1].detach().cpu()
                current_z = trajectory[1][-1].detach().cpu()

                del trajectory
                torch.cuda.empty_cache()

        # Final average loss
        if total_chunks > 0:
            avg_loss = total_loss / total_chunks
        else:
            return {'protein': protein_id, 'loss': 0.0, 'error': 'no_valid_chunks'}

        # Cleanup
        del m_init, z_init
        if 'current_m' in locals():
            del current_m, current_z

        return {
            'protein': protein_id,
            'loss': avg_loss,
            'selected_blocks': selected_blocks,
            'num_chunks': total_chunks,
            'approach': 'simple_sequential_strided'
        }

    def run_epoch(self, proteins: List, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  scaler: GradScaler, logger, epoch: int, total_epochs: int) -> Tuple[List[float], int]:
        """Run a single training epoch"""
        logger.log_epoch_start(epoch, total_epochs, [p[0] if isinstance(p, tuple) else p for p in proteins], self.is_preliminary)

        epoch_losses = []
        successful_proteins = 0

        for protein_idx, protein in enumerate(proteins):
            protein_id = protein[0] if isinstance(protein, tuple) else protein
            print(f"  [{protein_idx + 1}/{len(proteins)}] {protein_id}... ", end='', flush=True)

            protein_start_time = time.time()
            try:
                result = self.process_single_protein(protein, model, optimizer, scaler, training=True)
                epoch_losses.append(result['loss'])
                successful_proteins += 1
                print(f"✅ Loss: {result['loss']:.5f}")

                protein_time = time.time() - protein_start_time
                step_info = {'approach': result.get('approach', 'adjoint_0_to_48_unified'),
                            'num_blocks': result.get('selected_blocks', [0, 48]),
                            'cluster_size': self.config.reduced_cluster_size}

                logger.log_protein_step(protein_id, protein_idx, result['loss'], step_info, time_taken=protein_time, is_preliminary=self.is_preliminary)

            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                continue

        return epoch_losses, successful_proteins

    def validate(self, proteins: List, model: torch.nn.Module) -> Dict:
        """Run validation"""
        model.eval()
        val_losses = []
        successful_validations = 0

        with torch.no_grad():
            for val_idx, protein in enumerate(proteins):
                protein_id = protein[0] if isinstance(protein, tuple) else protein
                print(f"    [{val_idx + 1}/{len(proteins)}] {protein_id}... ", end='', flush=True)

                try:
                    result = self.process_single_protein(protein, model, training=False)
                    val_losses.append(result['loss'])
                    successful_validations += 1
                    print(f"✅ Loss: {result['loss']:.5f}")
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
                    continue

        model.train()

        if val_losses:
            return {
                'avg_loss': sum(val_losses) / len(val_losses),
                'min_loss': min(val_losses),
                'max_loss': max(val_losses),
                'num_proteins': len(val_losses),
                'successful_validations': successful_validations,
            }
        return {'avg_loss': float('inf'), 'num_proteins': 0, 'successful_validations': 0}


def setup_training_environment(config: TrainingConfig):
    """Setup model, optimizer, datasets, etc."""
    # Device setup
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        config.use_amp = False

    # Initialize CUDA
    if config.device == 'cuda':
        torch.cuda.init()

    # Model setup
    c_m, c_z = 256, 128
    if config.use_fast_ode:
        model = EvoformerODEFuncFast(c_m, c_z, config.hidden_dim).to(config.device)
    else:
        model = EvoformerODEFunc(c_m, c_z, config.hidden_dim).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler(enabled=config.use_amp)

    # Load datasets
    train_proteins = DataManager.get_available_proteins(config.data_dirs, config.splits_dir, 'training')
    val_proteins = DataManager.get_available_proteins(config.data_dirs, config.splits_dir, 'validation')

    # Filter by size
    if config.max_residues:
        train_proteins = DataManager.filter_proteins_by_size(train_proteins, config.max_residues)
        val_proteins = DataManager.filter_proteins_by_size(val_proteins, config.max_residues)

    return model, optimizer, scaler, train_proteins, val_proteins


def setup_logging(config: TrainingConfig):
    """Setup logger"""
    if not config.experiment_name or not config.output_dir:
        return None

    try:
        os.makedirs(config.output_dir, exist_ok=True)
        logger = TrainingLogger(config.output_dir, config.experiment_name)

        model_info = {
            'total_params': 'TBD',  # Will be filled after model creation
            'model_type': 'EvoformerODEFuncFast' if config.use_fast_ode else 'EvoformerODEFunc',
            'preliminary_training': config.enable_preliminary_training,
        }

        optimizer_info = {'learning_rate': config.learning_rate}
        logger.log_configuration(config, model_info, optimizer_info)
        return logger
    except Exception:
        return None


def setup_schedulers(optimizer, config: TrainingConfig, has_validation: bool, reset_state: bool = False):
    """Setup LR scheduler and early stopping with optional state reset"""
    if not has_validation:
        return None, None

    lr_scheduler = LearningRateScheduler(
        optimizer, patience=config.lr_patience, factor=config.lr_factor, min_lr=config.min_lr
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta
    )

    # If resetting state, ensure fresh start
    if reset_state:
        lr_scheduler.best_loss = float('inf')
        lr_scheduler.patience_counter = 0
        lr_scheduler.lr_reductions = 0

        early_stopping.best_loss = float('inf')
        early_stopping.best_epoch = 0
        early_stopping.patience_counter = 0
        early_stopping.best_model_state = None
        early_stopping.should_stop = False

    return lr_scheduler, early_stopping

# def run_preliminary_training(config: TrainingConfig, model, optimizer, scaler, logger) -> bool:
#     """Run preliminary training phase - now with LR scheduling support"""
#     if not config.enable_preliminary_training:
#         return True
#
#     print(f"\n🚀 PRELIMINARY TRAINING PHASE")
#     print(f"=" * 60)
#
#     # Get preliminary proteins
#     prelim_train_proteins = DataManager.get_available_proteins([config.prelim_data_dir], config.splits_dir, 'training', single_dir=True)
#     prelim_val_proteins = DataManager.get_available_proteins([config.prelim_data_dir], config.splits_dir, 'validation', single_dir=True)
#
#     # Filter by size
#     if config.max_residues:
#         prelim_train_proteins = DataManager.filter_proteins_by_size(prelim_train_proteins, config.max_residues, config.prelim_data_dir)
#         prelim_val_proteins = DataManager.filter_proteins_by_size(prelim_val_proteins, config.max_residues, config.prelim_data_dir)
#
#     if not prelim_train_proteins:
#         print("❌ No preliminary training proteins found")
#         return False
#
#     # Setup preliminary phase
#     prelim_phase = TrainingPhase(config, is_preliminary=True)
#     prelim_lr_scheduler, prelim_early_stopping = setup_schedulers(optimizer, config, bool(prelim_val_proteins))
#
#     # Training loop
#     for epoch in range(config.prelim_max_epochs):
#         print(f"\n📈 Preliminary Epoch {epoch + 1}/{config.prelim_max_epochs}")
#         print(f"🎛️  LR: {optimizer.param_groups[0]['lr']:.2e}")  # Show current LR
#
#         epoch_losses, successful_proteins = prelim_phase.run_epoch(prelim_train_proteins, model, optimizer, scaler, logger, epoch + 1, config.prelim_max_epochs)
#
#         # Validation
#         val_results = None
#         if prelim_val_proteins:
#             print(f"\n🔍 Preliminary validation on {len(prelim_val_proteins)} proteins...")
#             val_results = prelim_phase.validate(prelim_val_proteins, model)
#
#         if logger:
#             logger.log_epoch_end(val_results, is_preliminary=True, optimizer=optimizer)
#
#         # Training summary
#         if epoch_losses:
#             avg_train_loss = sum(epoch_losses) / len(epoch_losses)
#             print(f"📊 Training: {avg_train_loss:.5f} ({successful_proteins}/{len(prelim_train_proteins)})")
#
#             # LR scheduling - NEW: Add LR scheduler step for preliminary training
#             if prelim_lr_scheduler and val_results:
#                 prelim_lr_scheduler.step(val_results['avg_loss'], epoch + 1)
#
#         # Early stopping check
#         if val_results and prelim_early_stopping(val_results['avg_loss'], epoch + 1, model):
#             print(f"\n🛑 Early stopping triggered in preliminary training!")
#             break
#
#     print(f"\n✅ Preliminary training completed!")
#     return True


def run_preliminary_training(config: TrainingConfig, model, optimizer, scaler, logger) -> bool:
    """SIMPLIFIED: Back to working approach with just NaN monitoring"""
    if not config.enable_preliminary_training:
        return True

    print(f"\n🚀 PRELIMINARY TRAINING PHASE")
    print(f"=" * 60)

    # Get preliminary proteins
    prelim_train_proteins = DataManager.get_available_proteins([config.prelim_data_dir], config.splits_dir, 'training',
                                                               single_dir=True)
    prelim_val_proteins = DataManager.get_available_proteins([config.prelim_data_dir], config.splits_dir, 'validation',
                                                             single_dir=True)

    if config.max_residues:
        prelim_train_proteins = DataManager.filter_proteins_by_size(prelim_train_proteins, config.max_residues,
                                                                    config.prelim_data_dir)
        prelim_val_proteins = DataManager.filter_proteins_by_size(prelim_val_proteins, config.max_residues,
                                                                  config.prelim_data_dir)

    if not prelim_train_proteins:
        print("❌ No preliminary training proteins found")
        return False

    prelim_phase = TrainingPhase(config, is_preliminary=True)
    prelim_lr_scheduler, prelim_early_stopping = setup_schedulers(optimizer, config, bool(prelim_val_proteins))

    # Simple NaN monitoring (no recovery, just detection)
    consecutive_nan_epochs = 0
    max_nan_epochs = 3

    for epoch in range(config.prelim_max_epochs):
        print(f"\n📈 Preliminary Epoch {epoch + 1}/{config.prelim_max_epochs}")
        print(f"🎛️  LR: {optimizer.param_groups[0]['lr']:.2e}")

        epoch_losses, successful_proteins = prelim_phase.run_epoch(
            prelim_train_proteins, model, optimizer, scaler, logger, epoch + 1, config.prelim_max_epochs
        )

        # Check for NaN in results
        if epoch_losses:
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)

            if torch.isnan(torch.tensor(avg_train_loss)) or torch.isinf(torch.tensor(avg_train_loss)):
                consecutive_nan_epochs += 1
                print(f"⚠️  NaN/Inf training loss detected! ({consecutive_nan_epochs}/{max_nan_epochs})")

                if consecutive_nan_epochs >= max_nan_epochs:
                    print(f"❌ Too many consecutive NaN epochs, stopping preliminary training")
                    break

                # Simple LR reduction
                for group in optimizer.param_groups:
                    group['lr'] *= 0.5
                print(f"🔽 Emergency LR reduction: {group['lr']:.2e}")
                continue
            else:
                consecutive_nan_epochs = 0
                print(f"📊 Training: {avg_train_loss:.5f} ({successful_proteins}/{len(prelim_train_proteins)})")

        # Validation
        val_results = None
        if prelim_val_proteins and consecutive_nan_epochs == 0:
            print(f"\n🔍 Preliminary validation on {len(prelim_val_proteins)} proteins...")
            val_results = prelim_phase.validate(prelim_val_proteins, model)

        if logger:
            logger.log_epoch_end(val_results, is_preliminary=True, optimizer=optimizer)

        # LR scheduling and early stopping
        if consecutive_nan_epochs == 0:
            if prelim_lr_scheduler and val_results:
                prelim_lr_scheduler.step(val_results['avg_loss'], epoch + 1)

            if val_results and prelim_early_stopping(val_results['avg_loss'], epoch + 1, model):
                print(f"\n🛑 Early stopping triggered in preliminary training!")
                break

    print(f"\n✅ Preliminary training completed!")


def save_final_model(model, config: TrainingConfig, logger, training_stats: Dict):
    """Save final model and complete logging"""
    if not config.experiment_name or not config.output_dir:
        return

    model_path = os.path.join(config.output_dir, f"{config.experiment_name}_final_model.pt")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'training_stats': training_stats
    }

    torch.save(save_dict, model_path)
    logger.log_training_complete(model_path)
    print(f"🤖 Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', required=True)
    parser.add_argument('--splits_dir', required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--reduced_cluster_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--integrator', default='rk4')
    parser.add_argument('--use_fast_ode', action='store_true')
    parser.add_argument('--loss', choices=['default', 'weighted_row', 'single_row'], default='default',
                       help='MSA loss strategy: default (full MSA), weighted_row (emphasize first row), single_row (first row only)')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--max_residues', type=int, default=None)
    parser.add_argument('--max_time_hours', type=float, default=None)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--early_stopping_patience', type=int, default=7)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
    parser.add_argument('--aggressive_cleanup', action='store_true')
    parser.add_argument('--enable_preliminary_training', action='store_true')
    parser.add_argument('--prelim_data_dir', type=str)
    parser.add_argument('--prelim_block_stride', type=int, default=4)
    parser.add_argument('--prelim_max_epochs', type=int, default=20)
    parser.add_argument('--prelim_chunk_size', type=int, default=4)

    args = parser.parse_args()

    config = TrainingConfig.from_args(args)
    config.validate()

    print(f"🚀 Neural ODE Training - Restructured Version")
    print(f"📁 Data directories: {config.data_dirs}")
    print(f"💻 Device: {config.device}")
    print(f"🔧 Model: {'Fast ODE' if config.use_fast_ode else 'Full ODE'}")
    if config.enable_preliminary_training:
        print(f"🔄 Preliminary training enabled")

    model, optimizer, scaler, train_proteins, val_proteins = setup_training_environment(config)

    logger = setup_logging(config)
    if logger:
        logger.config['model_parameters'] = sum(p.numel() for p in model.parameters())
        logger.config['train_proteins'] = len(train_proteins)
        logger.config['val_proteins'] = len(val_proteins)


    print(f"🤖 Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"🧬 Training proteins: {len(train_proteins)}")
    print(f"🔍 Validation proteins: {len(val_proteins)}")

    training_start_time = time.time()
    if logger:
        logger.log_training_start()

    if config.enable_preliminary_training:
        success = run_preliminary_training(config, model, optimizer, scaler, logger)
        # At this point, model already has best preliminary weights loaded
        if success and logger:
            logger.log_main_training_start()

    print(f"\n🚀 MAIN TRAINING PHASE (0→48 blocks)")
    print(f"=" * 60)
    print("🔄 Resetting optimizer and schedulers for main training...")


    # Reset learning rate and optimizer state
    for group in optimizer.param_groups:
        group['lr'] = config.learning_rate
    optimizer.state.clear()

    # Create FRESH schedulers with reset_state=True
    lr_scheduler, early_stopping = setup_schedulers(optimizer, config, bool(val_proteins), reset_state=True)

    # Force early stopping to start fresh (no best model state from preliminary)
    if early_stopping:
        early_stopping.best_model_state = None  # Ensure no preliminary state carries over

    main_phase = TrainingPhase(config, is_preliminary=False)
    interrupted_by_timeout = False
    final_epoch = 0
    max_time_seconds = config.max_time_hours * 3600 if config.max_time_hours else None

    for epoch in range(config.epochs):
        final_epoch = epoch + 1

        if max_time_seconds:
            elapsed_time = time.time() - training_start_time
            if elapsed_time >= max_time_seconds:
                interrupted_by_timeout = True
                print(f"\n⏰ Time limit reached ({config.max_time_hours} hours). Stopping...")
                break

        print(f"\n📈 Main Epoch {epoch + 1}/{config.epochs}")
        print(f"🎛️  LR: {optimizer.param_groups[0]['lr']:.2e}")

        epoch_losses, successful_proteins = main_phase.run_epoch(
            train_proteins, model, optimizer, scaler, logger, epoch + 1, config.epochs
        )

        val_results = None
        if val_proteins:
            print(f"\n🔍 Validation on {len(val_proteins)} proteins...")
            val_results = main_phase.validate(val_proteins, model)

            print(f"📊 Validation: {val_results['avg_loss']:.5f} "
                  f"({val_results['successful_validations']}/{len(val_proteins)})")

        if logger:
            logger.log_epoch_end(val_results, is_preliminary=False, optimizer=optimizer)

        if epoch_losses:
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"📊 Training: {avg_train_loss:.5f} ({successful_proteins}/{len(train_proteins)})")

            if lr_scheduler and val_results:
                lr_scheduler.step(val_results['avg_loss'], epoch + 1)

            if early_stopping and val_results:
                if early_stopping(val_results['avg_loss'], epoch + 1, model):
                    print(f"\n🛑 Early stopping triggered!")
                    break

        elapsed_time = time.time() - training_start_time
        print(f"⏱️  Time: {elapsed_time / 60:.1f} min")
        if max_time_seconds:
            remaining_time = max_time_seconds - elapsed_time
            print(f"⏰ Remaining: {remaining_time / 60:.1f} min")

    total_time = time.time() - training_start_time

    if interrupted_by_timeout:
        print(f"\n⏰ Training stopped due to time limit after {final_epoch} epochs!")
    else:
        print(f"\n🎯 Training completed!")

    print(f"⏱️  Total time: {total_time / 60:.1f} minutes")

    if early_stopping:
        print(f"🏆 Best validation loss: {early_stopping.best_loss:.4f} (epoch {early_stopping.best_epoch})")

    if lr_scheduler:
        print(f"📉 LR reductions: {lr_scheduler.lr_reductions}")
        print(f"🎛️  Final LR: {lr_scheduler.get_last_lr():.2e}")

    training_stats = {
        'total_epochs': final_epoch,
        'total_time_minutes': total_time / 60,
        'interrupted_by_timeout': interrupted_by_timeout,
        'best_val_loss': early_stopping.best_loss if early_stopping else None,
        'best_val_epoch': early_stopping.best_epoch if early_stopping else None,
        'lr_reductions': lr_scheduler.lr_reductions if lr_scheduler else 0,
        'final_lr': lr_scheduler.get_last_lr() if lr_scheduler else config.learning_rate,
        'method': 'adjoint_0_to_48_unified',
        'preliminary_training_enabled': config.enable_preliminary_training,
        'loss_strategy': config.loss,
    }

    if logger:
        if interrupted_by_timeout:
            logger.interrupted_at_epoch = final_epoch
        save_final_model(model, config, logger, training_stats)
        print(f"📊 Full report: {config.output_dir}/{config.experiment_name}.txt")

if __name__ == "__main__":
    main()