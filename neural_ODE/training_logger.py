import os
import datetime
import torch
import json
import sys
from typing import Dict, List, Any


class TrainingLogger:
    """
    Fixed logger that properly tracks training start and completion times
    Now includes interrupt handling and success rate tracking
    """

    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Only one output file
        self.log_file = os.path.join(output_dir, f"{self.experiment_name}.txt")

        # Training data storage
        self.config = {}
        self.system_info = {}
        self.training_history = []
        self.protein_results = {}
        self.epoch_summaries = []

        # Timing tracking
        self.training_start_time = None
        self.training_end_time = None

        # Interrupt tracking
        self.interrupted_at_epoch = None

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
        """Log actual training start time"""
        self.training_start_time = datetime.datetime.now()

    def log_epoch_start(self, epoch: int, total_epochs: int, proteins: List[str]):
        """Log the start of an epoch"""
        if self.training_start_time is None:
            self.training_start_time = datetime.datetime.now()

        self.current_epoch = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'proteins': proteins,
            'start_time': datetime.datetime.now(),
            'protein_results': {},
            'total_loss': 0,
            'successful_proteins': 0,
            'total_proteins': len(proteins),
            'memory_stats': {},
            'validation': None
        }

    def log_protein_step(self, protein_id: str, step_idx: int, loss: float,
                         step_info: Dict = None, memory_stats: Dict = None, time_taken: float = None):
        """Log results for a single protein training step"""
        protein_result = {
            'protein_id': protein_id,
            'step_idx': step_idx,
            'total_loss': loss,
            'approach': step_info.get('approach', 'unknown') if step_info else 'unknown',
            'num_blocks': step_info.get('num_blocks', 'unknown') if step_info else 'unknown',
            'time_taken_ms': time_taken * 1000 if time_taken else None,
            'memory_stats': memory_stats or {}
        }

        if step_info:
            if step_info['approach'] == 'batched':
                protein_result.update({
                    'batch_size': step_info.get('batch_size'),
                    'num_batches': step_info.get('num_batches'),
                    'batch_losses': step_info.get('batch_losses', [])
                })

        self.current_epoch['protein_results'][protein_id] = protein_result
        self.current_epoch['total_loss'] += loss
        self.current_epoch['successful_proteins'] += 1

    def log_epoch_end(self, val_results: Dict = None):
        """Log the end of an epoch and compute summaries"""
        self.current_epoch['end_time'] = datetime.datetime.now()
        self.current_epoch['duration'] = (
                self.current_epoch['end_time'] - self.current_epoch['start_time']).total_seconds()

        # Store validation results
        if val_results:
            self.current_epoch['validation'] = val_results

        # Compute average loss
        if self.current_epoch['successful_proteins'] > 0:
            self.current_epoch['average_loss'] = self.current_epoch['total_loss'] / self.current_epoch[
                'successful_proteins']
        else:
            self.current_epoch['average_loss'] = float('inf')

        # Find best and worst performing proteins
        if self.current_epoch['protein_results']:
            protein_losses = [(pid, result['total_loss']) for pid, result in
                              self.current_epoch['protein_results'].items()]
            best_protein = min(protein_losses, key=lambda x: x[1])
            worst_protein = max(protein_losses, key=lambda x: x[1])

            self.current_epoch['best_protein'] = {'id': best_protein[0], 'loss': best_protein[1]}
            self.current_epoch['worst_protein'] = {'id': worst_protein[0], 'loss': worst_protein[1]}

        self.epoch_summaries.append(self.current_epoch.copy())
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

        # Training Progress Summary
        if self.epoch_summaries:
            f.write("TRAINING PROGRESS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Summary table with success rates
            f.write(
                f"{'Epoch':<8} {'Train Loss':<14} {'Train Success':<14} {'Val Loss':<14} {'Val Success':<14} {'Duration (s)':<12}\n")
            f.write("-" * 88 + "\n")

            for epoch_data in self.epoch_summaries:
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

            # Performance analysis
            if len(self.epoch_summaries) > 0:
                best_epoch = min(self.epoch_summaries, key=lambda x: x['average_loss'])
                worst_epoch = max(self.epoch_summaries, key=lambda x: x['average_loss'])

                f.write("PERFORMANCE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Training Epoch: {best_epoch['epoch']} (Loss: {best_epoch['average_loss']:.5f})\n")
                f.write(f"Worst Training Epoch: {worst_epoch['epoch']} (Loss: {worst_epoch['average_loss']:.5f})\n")
                if worst_epoch['average_loss'] > 0:
                    improvement = worst_epoch['average_loss'] / best_epoch['average_loss']
                    f.write(f"Training Improvement: {improvement:.1f}x better from worst to best\n")

                # Success rate analysis
                if self.epoch_summaries:
                    final_epoch = self.epoch_summaries[-1]
                    train_success_rate = final_epoch['successful_proteins'] / final_epoch['total_proteins'] * 100
                    f.write(f"Final Training Success Rate: {train_success_rate:.1f}%\n")

                    if final_epoch.get('validation'):
                        val_data = final_epoch['validation']
                        val_success_rate = val_data['successful_validations'] / val_data['num_proteins'] * 100
                        f.write(f"Final Validation Success Rate: {val_success_rate:.1f}%\n")

                # Validation analysis if available
                val_epochs = [e for e in self.epoch_summaries if 'validation' in e and e['validation']]
                if val_epochs:
                    best_val_epoch = min(val_epochs, key=lambda x: x['validation']['avg_loss'])
                    worst_val_epoch = max(val_epochs, key=lambda x: x['validation']['avg_loss'])
                    f.write(
                        f"Best Validation Epoch: {best_val_epoch['epoch']} (Loss: {best_val_epoch['validation']['avg_loss']:.5f})\n")
                    f.write(
                        f"Worst Validation Epoch: {worst_val_epoch['epoch']} (Loss: {worst_val_epoch['validation']['avg_loss']:.5f})\n")

                    # Check for overfitting
                    final_epoch = self.epoch_summaries[-1]
                    if 'validation' in final_epoch and final_epoch['validation']:
                        train_loss = final_epoch['average_loss']
                        val_loss = final_epoch['validation']['avg_loss']
                        if val_loss > train_loss * 1.2:  # 20% worse validation
                            f.write(
                                f"⚠️  Potential overfitting detected: Val loss ({val_loss:.5f}) >> Train loss ({train_loss:.5f})\n")

                f.write("\n")

        # Detailed results for each epoch
        if self.epoch_summaries:
            f.write("DETAILED EPOCH RESULTS\n")
            f.write("=" * 50 + "\n\n")

            for epoch_data in self.epoch_summaries:
                f.write(f"Epoch {epoch_data['epoch']}:\n")
                f.write(f"  Duration: {epoch_data['duration']:.1f} seconds\n")
                f.write(f"  Training Loss: {epoch_data['average_loss']:.5f}\n")
                f.write(
                    f"  Training Success: {epoch_data['successful_proteins']}/{epoch_data['total_proteins']} proteins\n")

                if 'validation' in epoch_data and epoch_data['validation']:
                    val = epoch_data['validation']
                    f.write(f"  Validation Loss: {val['avg_loss']:.5f} ({val['num_proteins']} proteins)\n")
                    f.write(f"  Validation Success: {val['successful_validations']}/{val['num_proteins']} proteins\n")
                    if 'min_loss' in val and 'max_loss' in val:
                        f.write(f"    Val range: [{val['min_loss']:.5f}, {val['max_loss']:.5f}]\n")

                if 'best_protein' in epoch_data:
                    f.write(f"  Best: {epoch_data['best_protein']['id']} ({epoch_data['best_protein']['loss']:.5f})\n")
                    f.write(
                        f"  Worst: {epoch_data['worst_protein']['id']} ({epoch_data['worst_protein']['loss']:.5f})\n")

                f.write(f"  Training Proteins:\n")
                for protein_id, result in epoch_data['protein_results'].items():
                    f.write(f"    {protein_id}: {result['total_loss']:.5f}\n")

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