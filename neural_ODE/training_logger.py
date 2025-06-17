import os
import datetime
import torch
import json
import sys
from typing import Dict, List, Any


class TrainingLogger:
    """
    Enhanced logger that properly tracks both preliminary and main training phases
    Now includes separate tracking for preliminary training with unified reporting
    """

    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Only one output file
        self.log_file = os.path.join(output_dir, f"{self.experiment_name}.txt")

        # Training data storage
        self.config = {}
        self.system_info = {}

        # Separate tracking for preliminary and main training
        self.preliminary_history = []
        self.main_training_history = []
        self.preliminary_epoch_summaries = []
        self.main_epoch_summaries = []

        # Combined tracking for backward compatibility
        self.training_history = []  # Will contain both phases
        self.epoch_summaries = []  # Will contain both phases

        self.protein_results = {}

        # Timing tracking
        self.training_start_time = None
        self.preliminary_start_time = None
        self.main_training_start_time = None
        self.training_end_time = None

        # Phase tracking
        self.preliminary_completed = False
        self.main_training_started = False

        # Interrupt tracking
        self.interrupted_at_epoch = None

        # Current epoch tracking
        self.current_epoch = None
        self.current_preliminary_epoch = None
        self.current_main_epoch = None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize file
        self._write_header()

    def log_configuration(self, args, model_info: Dict, optimizer_info: Dict):
        """Log all configuration parameters"""
        self.config = {
            'data_dir': getattr(args, 'data_dir', 'N/A'),
            'splits_dir': getattr(args, 'splits_dir', 'N/A'),
            'mode': getattr(args, 'mode', 'N/A'),
            'output_dir': getattr(args, 'output_dir', 'N/A'),
            'use_fast_ode': getattr(args, 'use_fast_ode', False),
            'reduced_cluster_size': getattr(args, 'reduced_cluster_size', 'N/A'),
            'hidden_dim': getattr(args, 'hidden_dim', 'N/A'),
            'model_parameters': model_info.get('total_params', 'N/A'),
            'model_type': model_info.get('model_type', 'N/A'),
            'epochs': getattr(args, 'epochs', 'N/A'),
            'learning_rate': optimizer_info.get('learning_rate', 'N/A'),
            'batch_size': getattr(args, 'batch_size', 'N/A'),
            'integrator': getattr(args, 'integrator', 'N/A'),
            'use_amp': getattr(args, 'use_amp', False),
            'max_residues': getattr(args, 'max_residues', 'N/A'),
            'loss_function': model_info.get('loss_function', 'Adaptive MSE'),
            'train_proteins': model_info.get('train_proteins', 'N/A'),
            'val_proteins': model_info.get('val_proteins', 'N/A'),
            'use_sequential_loading': getattr(args, 'use_sequential_loading', False),
            'aggressive_cleanup': getattr(args, 'aggressive_cleanup', False),
            # Preliminary training config
            'preliminary_training_enabled': getattr(args, 'enable_preliminary_training', False),
            'prelim_data_dir': getattr(args, 'prelim_data_dir', 'N/A'),
            'prelim_block_stride': getattr(args, 'prelim_block_stride', 'N/A'),
            'prelim_max_epochs': getattr(args, 'prelim_max_epochs', 'N/A'),
            'prelim_early_stopping_min_delta': getattr(args, 'prelim_early_stopping_min_delta', 'N/A'),
        }

        self.system_info = {
            'cuda_available': torch.cuda.is_available(),
            'device': getattr(args, 'device', 'N/A'),
            'pytorch_version': torch.__version__,
        }

        if torch.cuda.is_available() and getattr(args, 'device', '') == 'cuda':
            self.system_info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'total_gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"
            })

    def log_training_start(self):
        """Log overall training start time (called once at the very beginning)"""
        self.training_start_time = datetime.datetime.now()

    def log_main_training_start(self):
        """Log main training phase start time (called after preliminary training)"""
        self.main_training_start_time = datetime.datetime.now()
        self.main_training_started = True

    def log_epoch_start(self, epoch: int, total_epochs: int, proteins: List[str], is_preliminary: bool = False):
        """Log the start of an epoch (preliminary or main)"""
        if self.training_start_time is None:
            self.training_start_time = datetime.datetime.now()

        epoch_data = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'proteins': proteins,
            'start_time': datetime.datetime.now(),
            'protein_results': {},
            'total_loss': 0,
            'successful_proteins': 0,
            'total_proteins': len(proteins),
            'memory_stats': {},
            'validation': None,
            'is_preliminary': is_preliminary
        }

        if is_preliminary:
            if self.preliminary_start_time is None:
                self.preliminary_start_time = datetime.datetime.now()
            self.current_preliminary_epoch = epoch_data
        else:
            if self.main_training_start_time is None:
                self.main_training_start_time = datetime.datetime.now()
            self.current_main_epoch = epoch_data

        # Also set current_epoch for backward compatibility
        self.current_epoch = epoch_data

    def log_protein_step(self, protein_id: str, step_idx: int, loss: float,
                         step_info: Dict = None, memory_stats: Dict = None,
                         time_taken: float = None, is_preliminary: bool = False):
        """Log results for a single protein training step"""
        protein_result = {
            'protein_id': protein_id,
            'step_idx': step_idx,
            'total_loss': loss,
            'approach': step_info.get('approach', 'unknown') if step_info else 'unknown',
            'num_blocks': step_info.get('num_blocks', 'unknown') if step_info else 'unknown',
            'time_taken_ms': time_taken * 1000 if time_taken else None,
            'memory_stats': memory_stats or {},
            'is_preliminary': is_preliminary
        }

        if step_info:
            if step_info['approach'] == 'batched':
                protein_result.update({
                    'batch_size': step_info.get('batch_size'),
                    'num_batches': step_info.get('num_batches'),
                    'batch_losses': step_info.get('batch_losses', [])
                })
            elif 'block_stride' in step_info:
                protein_result.update({
                    'block_stride': step_info.get('block_stride'),
                    'selected_blocks': step_info.get('selected_blocks', []),
                    'cluster_size': step_info.get('cluster_size', 'unknown')
                })

        # Add to current epoch (preliminary or main)
        current_epoch = self.current_preliminary_epoch if is_preliminary else self.current_main_epoch
        if current_epoch:
            current_epoch['protein_results'][protein_id] = protein_result
            current_epoch['total_loss'] += loss
            current_epoch['successful_proteins'] += 1

        # Also add to overall current_epoch for backward compatibility
        if self.current_epoch:
            self.current_epoch['protein_results'][protein_id] = protein_result

    def log_epoch_end(self, val_results: Dict = None, is_preliminary: bool = False):
        """Log the end of an epoch and compute summaries"""
        current_epoch = self.current_preliminary_epoch if is_preliminary else self.current_main_epoch

        if not current_epoch:
            return

        current_epoch['end_time'] = datetime.datetime.now()
        current_epoch['duration'] = (
                current_epoch['end_time'] - current_epoch['start_time']).total_seconds()

        # Store validation results
        if val_results:
            current_epoch['validation'] = val_results

        # Compute average loss
        if current_epoch['successful_proteins'] > 0:
            current_epoch['average_loss'] = current_epoch['total_loss'] / current_epoch['successful_proteins']
        else:
            current_epoch['average_loss'] = float('inf')

        # Find best and worst performing proteins
        if current_epoch['protein_results']:
            protein_losses = [(pid, result['total_loss']) for pid, result in
                              current_epoch['protein_results'].items()]
            best_protein = min(protein_losses, key=lambda x: x[1])
            worst_protein = max(protein_losses, key=lambda x: x[1])

            current_epoch['best_protein'] = {'id': best_protein[0], 'loss': best_protein[1]}
            current_epoch['worst_protein'] = {'id': worst_protein[0], 'loss': worst_protein[1]}

        # Add to appropriate history
        if is_preliminary:
            self.preliminary_epoch_summaries.append(current_epoch.copy())
            if current_epoch['epoch'] == current_epoch['total_epochs']:  # Last preliminary epoch
                self.preliminary_completed = True
        else:
            self.main_epoch_summaries.append(current_epoch.copy())

        # Add to combined history for backward compatibility
        self.epoch_summaries.append(current_epoch.copy())

        # Update log file
        self._update_log_file()

    def log_training_complete(self, final_model_path: str = None):
        """Log training completion with proper timing"""
        self.training_end_time = datetime.datetime.now()
        self.final_model_path = final_model_path
        self._write_final_report()

    def close(self):
        """Close logger"""
        pass

    def _write_header(self):
        """Write the initial header to the log file"""
        with open(self.log_file, 'w') as f:
            f.write("EVOFORMER NEURAL ODE TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Report initialized: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _update_log_file(self):
        """Update the log file with current progress"""
        with open(self.log_file, 'w') as f:
            self._write_complete_report(f)

    def _write_final_report(self):
        """Write the complete final report"""
        with open(self.log_file, 'w') as f:
            self._write_complete_report(f, final=True)

    def _write_complete_report(self, f, final=False):
        """Write the complete report to file"""
        # Header with interrupt information
        f.write("EVOFORMER NEURAL ODE TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Check for interruption
        if final and self.interrupted_at_epoch is not None:
            f.write(f"⏰ INTERRUPTED AT {self.interrupted_at_epoch} EPOCHS (TIME LIMIT REACHED)\n")
            f.write("=" * 50 + "\n\n")

        # Experiment info with timing
        f.write(f"Experiment: {self.experiment_name}\n")

        if self.training_start_time:
            f.write(f"Started: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        if final and self.training_end_time:
            f.write(f"Completed: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            if self.training_start_time:
                actual_duration = self.training_end_time - self.training_start_time
                duration_minutes = actual_duration.total_seconds() / 60
                f.write(f"Total Training Time: {duration_minutes:.1f} minutes ({duration_minutes / 60:.1f} hours)\n")
        elif not final:
            f.write(f"Status: Training in progress...\n")
            if self.training_start_time:
                current_duration = datetime.datetime.now() - self.training_start_time
                duration_minutes = current_duration.total_seconds() / 60
                f.write(f"Current Runtime: {duration_minutes:.1f} minutes\n")

        f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # Configuration
        f.write("Configuration Settings:\n")
        f.write("-" * 30 + "\n")
        for key, value in self.config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # System Information
        f.write("System Information:\n")
        f.write("-" * 30 + "\n")
        for key, value in self.system_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # PRELIMINARY TRAINING SECTION
        if self.preliminary_epoch_summaries:
            f.write("PRELIMINARY TRAINING PHASE\n")
            f.write("=" * 50 + "\n\n")

            if self.preliminary_start_time:
                f.write(f"Preliminary Started: {self.preliminary_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            if self.preliminary_completed and self.main_training_start_time:
                prelim_duration = self.main_training_start_time - self.preliminary_start_time
                f.write(f"Preliminary Completed: {self.main_training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Preliminary Duration: {prelim_duration.total_seconds() / 60:.1f} minutes\n")
            f.write("\n")

            # Preliminary training summary table
            f.write("Preliminary Training Progress:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"{'Epoch':<8} {'Train Loss':<14} {'Train Success':<14} {'Val Loss':<14} {'Val Success':<14} {'Duration (s)':<12}\n")
            f.write("-" * 88 + "\n")

            for epoch_data in self.preliminary_epoch_summaries:
                epoch = epoch_data['epoch']
                train_loss = epoch_data['average_loss']
                train_success = f"{epoch_data['successful_proteins']}/{epoch_data['total_proteins']}"

                val_loss = epoch_data.get('validation', {}).get('avg_loss', 'N/A')
                val_success = 'N/A'
                if epoch_data.get('validation'):
                    val_data = epoch_data['validation']
                    val_success = f"{val_data.get('successful_validations', 0)}/{val_data.get('num_proteins', 0)}"

                duration = epoch_data['duration']

                if isinstance(val_loss, (int, float)):
                    val_loss = f"{val_loss:.5f}"

                f.write(
                    f"{epoch:<8} {train_loss:<14.5f} {train_success:<14} {val_loss:<14} {val_success:<14} {duration:<12.1f}\n")

            f.write("-" * 88 + "\n\n")

            # Preliminary training analysis
            if self.preliminary_epoch_summaries:
                best_prelim = min(self.preliminary_epoch_summaries, key=lambda x: x['average_loss'])
                worst_prelim = max(self.preliminary_epoch_summaries, key=lambda x: x['average_loss'])

                f.write("Preliminary Training Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Preliminary Epoch: {best_prelim['epoch']} (Loss: {best_prelim['average_loss']:.5f})\n")
                f.write(
                    f"Worst Preliminary Epoch: {worst_prelim['epoch']} (Loss: {worst_prelim['average_loss']:.5f})\n")

                final_prelim = self.preliminary_epoch_summaries[-1]
                train_success_rate = final_prelim['successful_proteins'] / final_prelim['total_proteins'] * 100
                f.write(f"Final Preliminary Success Rate: {train_success_rate:.1f}%\n")

                if final_prelim.get('validation'):
                    val_data = final_prelim['validation']
                    val_success_rate = val_data['successful_validations'] / val_data['num_proteins'] * 100
                    f.write(f"Final Preliminary Validation Success Rate: {val_success_rate:.1f}%\n")
                f.write("\n")

        # MAIN TRAINING SECTION
        if self.main_epoch_summaries:
            f.write("MAIN TRAINING PHASE\n")
            f.write("=" * 50 + "\n\n")

            if self.main_training_start_time:
                f.write(f"Main Training Started: {self.main_training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            if final and self.training_end_time and self.main_training_start_time:
                main_duration = self.training_end_time - self.main_training_start_time
                f.write(f"Main Training Completed: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Main Training Duration: {main_duration.total_seconds() / 60:.1f} minutes\n")
            f.write("\n")

            # Main training summary table
            f.write("Main Training Progress:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"{'Epoch':<8} {'Train Loss':<14} {'Train Success':<14} {'Val Loss':<14} {'Val Success':<14} {'Duration (s)':<12}\n")
            f.write("-" * 88 + "\n")

            for epoch_data in self.main_epoch_summaries:
                epoch = epoch_data['epoch']
                train_loss = epoch_data['average_loss']
                train_success = f"{epoch_data['successful_proteins']}/{epoch_data['total_proteins']}"

                val_loss = epoch_data.get('validation', {}).get('avg_loss', 'N/A')
                val_success = 'N/A'
                if epoch_data.get('validation'):
                    val_data = epoch_data['validation']
                    val_success = f"{val_data.get('successful_validations', 0)}/{val_data.get('num_proteins', 0)}"

                duration = epoch_data['duration']

                if isinstance(val_loss, (int, float)):
                    val_loss = f"{val_loss:.5f}"

                f.write(
                    f"{epoch:<8} {train_loss:<14.5f} {train_success:<14} {val_loss:<14} {val_success:<14} {duration:<12.1f}\n")

            f.write("-" * 88 + "\n\n")

            # Main training analysis
            best_main = min(self.main_epoch_summaries, key=lambda x: x['average_loss'])
            worst_main = max(self.main_epoch_summaries, key=lambda x: x['average_loss'])

            f.write("Main Training Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Main Training Epoch: {best_main['epoch']} (Loss: {best_main['average_loss']:.5f})\n")
            f.write(f"Worst Main Training Epoch: {worst_main['epoch']} (Loss: {worst_main['average_loss']:.5f})\n")

            final_main = self.main_epoch_summaries[-1]
            train_success_rate = final_main['successful_proteins'] / final_main['total_proteins'] * 100
            f.write(f"Final Main Training Success Rate: {train_success_rate:.1f}%\n")

            if final_main.get('validation'):
                val_data = final_main['validation']
                val_success_rate = val_data['successful_validations'] / val_data['num_proteins'] * 100
                f.write(f"Final Main Training Validation Success Rate: {val_success_rate:.1f}%\n")
            f.write("\n")

        # OVERALL SUMMARY (combines both phases)
        all_epochs = self.preliminary_epoch_summaries + self.main_epoch_summaries
        if all_epochs:
            f.write("OVERALL TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Performance analysis across both phases
            best_overall = min(all_epochs, key=lambda x: x['average_loss'])
            worst_overall = max(all_epochs, key=lambda x: x['average_loss'])

            f.write("OVERALL PERFORMANCE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            phase_name = "Preliminary" if best_overall.get('is_preliminary', False) else "Main"
            f.write(
                f"Best Overall Epoch: {phase_name} Epoch {best_overall['epoch']} (Loss: {best_overall['average_loss']:.5f})\n")

            phase_name = "Preliminary" if worst_overall.get('is_preliminary', False) else "Main"
            f.write(
                f"Worst Overall Epoch: {phase_name} Epoch {worst_overall['epoch']} (Loss: {worst_overall['average_loss']:.5f})\n")

            if worst_overall['average_loss'] > 0:
                improvement = worst_overall['average_loss'] / best_overall['average_loss']
                f.write(f"Overall Training Improvement: {improvement:.1f}x better from worst to best\n")

            # Final success rates
            if self.main_epoch_summaries:
                final_epoch = self.main_epoch_summaries[-1]
            elif self.preliminary_epoch_summaries:
                final_epoch = self.preliminary_epoch_summaries[-1]
            else:
                final_epoch = None

            if final_epoch:
                train_success_rate = final_epoch['successful_proteins'] / final_epoch['total_proteins'] * 100
                f.write(f"Final Training Success Rate: {train_success_rate:.1f}%\n")

                if final_epoch.get('validation'):
                    val_data = final_epoch['validation']
                    val_success_rate = val_data['successful_validations'] / val_data['num_proteins'] * 100
                    f.write(f"Final Validation Success Rate: {val_success_rate:.1f}%\n")

            f.write("\n")

        # Final model information
        if final and hasattr(self, 'final_model_path') and self.final_model_path:
            f.write("FINAL MODEL\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model saved to: {self.final_model_path}\n")
            f.write(f"Total parameters: {self.config.get('model_parameters', 'N/A')}\n")
            f.write(f"Model type: {self.config.get('model_type', 'N/A')}\n")

            if self.interrupted_at_epoch is not None:
                f.write(f"⏰ Training was interrupted at epoch {self.interrupted_at_epoch} due to time limit\n")
                f.write(f"Model represents the best state found before timeout\n")

            f.write("\n")

        f.write("=" * 50 + "\n")