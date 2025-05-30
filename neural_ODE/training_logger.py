import os
import datetime
import torch
import json
from typing import Dict, List, Any


class TrainingLogger:
    """
    Comprehensive logger for Neural ODE training that saves detailed reports
    similar to memory_config_tester output format
    """

    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = os.path.join(output_dir, f"{self.experiment_name}")

        # Training data storage
        self.config = {}
        self.system_info = {}
        self.training_history = []
        self.protein_results = {}
        self.epoch_summaries = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize log file
        self._write_header()

    def log_configuration(self, args, model_info: Dict, optimizer_info: Dict):
        """Log all configuration parameters"""
        self.config = {
            # Data settings
            'data_dir': getattr(args, 'data_dir', 'N/A'),
            'output_dir': getattr(args, 'output_dir', 'N/A'),

            # Model settings
            'use_fast_ode': getattr(args, 'use_fast_ode', False),
            'reduced_cluster_size': getattr(args, 'reduced_cluster_size', 'N/A'),
            'reduced_hidden_dim': getattr(args, 'reduced_hidden_dim', 'N/A'),
            'model_parameters': model_info.get('total_params', 'N/A'),
            'model_type': model_info.get('model_type', 'N/A'),

            # Training settings
            'epochs': getattr(args, 'epochs', 'N/A'),
            'learning_rate': optimizer_info.get('learning_rate', 'N/A'),
            'num_time_points': getattr(args, 'num_time_points', 'N/A'),
            'batch_size': getattr(args, 'batch_size', 'N/A'),
            'integrator': getattr(args, 'integrator', 'N/A'),
            'gradient_accumulation': getattr(args, 'gradient_accumulation', 'N/A'),
            'chunk_size': getattr(args, 'chunk_size', 'N/A'),

            # Optimization settings
            'use_amp': getattr(args, 'use_amp', False),
            'use_checkpoint': getattr(args, 'use_checkpoint', False),
            'reduced_precision_integration': getattr(args, 'reduced_precision_integration', False),
            'clean_memory': getattr(args, 'clean_memory', False),
            'monitor_memory': getattr(args, 'monitor_memory', False),
            'cpu_only': getattr(args, 'cpu_only', False),
            'memory_split_size': getattr(args, 'memory_split_size', 'N/A'),

            # Loss settings
            'loss_function': model_info.get('loss_function', 'N/A'),
            'msa_weight': model_info.get('msa_weight', 'N/A'),
            'pair_weight': model_info.get('pair_weight', 'N/A'),
            'gradient_clipping': model_info.get('gradient_clipping', 'N/A'),
        }

        # System info
        self.system_info = {
            'cuda_available': torch.cuda.is_available(),
            'device': 'cuda' if torch.cuda.is_available() and not getattr(args, 'cpu_only', False) else 'cpu',
            'pytorch_version': torch.__version__,
        }

        if torch.cuda.is_available() and not getattr(args, 'cpu_only', False):
            self.system_info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'total_gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"
            })

    def log_epoch_start(self, epoch: int, total_epochs: int, proteins: List[str]):
        """Log the start of an epoch"""
        self.current_epoch = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'proteins': proteins,
            'start_time': datetime.datetime.now(),
            'protein_results': {},
            'total_loss': 0,
            'memory_stats': {}
        }

    def log_protein_step(self, protein_id: str, step_idx: int, loss: float,
                         msa_loss: float = None, pair_loss: float = None,
                         memory_stats: Dict = None, time_taken: float = None):
        """Log results for a single protein training step"""
        protein_result = {
            'protein_id': protein_id,
            'step_idx': step_idx,
            'total_loss': loss,
            'msa_loss': msa_loss,
            'pair_loss': pair_loss,
            'time_taken_ms': time_taken * 1000 if time_taken else None,
            'memory_stats': memory_stats or {}
        }

        self.current_epoch['protein_results'][protein_id] = protein_result
        self.current_epoch['total_loss'] += loss

        # Update memory stats
        if memory_stats:
            for key, value in memory_stats.items():
                if key not in self.current_epoch['memory_stats']:
                    self.current_epoch['memory_stats'][key] = value
                else:
                    # Keep the maximum values
                    if 'max' in key.lower() and value > self.current_epoch['memory_stats'][key]:
                        self.current_epoch['memory_stats'][key] = value

    def log_epoch_end(self):
        """Log the end of an epoch and compute summaries"""
        self.current_epoch['end_time'] = datetime.datetime.now()
        self.current_epoch['duration'] = (
                    self.current_epoch['end_time'] - self.current_epoch['start_time']).total_seconds()

        # Compute average loss
        num_proteins = len(self.current_epoch['protein_results'])
        self.current_epoch['average_loss'] = self.current_epoch['total_loss'] / max(num_proteins, 1)

        # Find best and worst performing proteins
        if self.current_epoch['protein_results']:
            protein_losses = [(pid, result['total_loss']) for pid, result in
                              self.current_epoch['protein_results'].items()]
            best_protein = min(protein_losses, key=lambda x: x[1])
            worst_protein = max(protein_losses, key=lambda x: x[1])

            self.current_epoch['best_protein'] = {'id': best_protein[0], 'loss': best_protein[1]}
            self.current_epoch['worst_protein'] = {'id': worst_protein[0], 'loss': worst_protein[1]}

        self.epoch_summaries.append(self.current_epoch.copy())

        # Write to file immediately
        self._update_log_file()

    def log_training_complete(self, final_model_path: str = None):
        """Log training completion and generate final report"""
        self.training_complete_time = datetime.datetime.now()
        self.final_model_path = final_model_path
        self._write_final_report()

    def _write_header(self):
        """Write the initial header to the log file"""
        with open(self.log_file, 'w') as f:
            f.write("EVOFORMER NEURAL ODE TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

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
        # Header
        f.write("EVOFORMER NEURAL ODE TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Experiment info
        f.write(f"Experiment: {self.experiment_name}\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if final and hasattr(self, 'training_complete_time'):
            f.write(f"Completed: {self.training_complete_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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

            # Summary table
            f.write(
                f"{'Epoch':<8} {'Avg Loss':<12} {'Best Loss':<12} {'Worst Loss':<12} {'Duration (s)':<12} {'Memory (MB)'}\n")
            f.write("-" * 80 + "\n")

            for epoch_data in self.epoch_summaries:
                epoch = epoch_data['epoch']
                avg_loss = epoch_data['average_loss']
                best_loss = epoch_data.get('best_protein', {}).get('loss', 'N/A')
                worst_loss = epoch_data.get('worst_protein', {}).get('loss', 'N/A')
                duration = epoch_data['duration']
                max_memory = epoch_data['memory_stats'].get('max_memory_allocated_mb', 'N/A')

                f.write(
                    f"{epoch:<8} {avg_loss:<12.2f} {best_loss:<12.2f} {worst_loss:<12.2f} {duration:<12.1f} {max_memory}\n")

            f.write("-" * 80 + "\n\n")

            # Best/Worst Performance
            if len(self.epoch_summaries) > 0:
                best_epoch = min(self.epoch_summaries, key=lambda x: x['average_loss'])
                worst_epoch = max(self.epoch_summaries, key=lambda x: x['average_loss'])

                f.write("PERFORMANCE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Epoch: {best_epoch['epoch']} (Loss: {best_epoch['average_loss']:.2f})\n")
                f.write(f"Worst Epoch: {worst_epoch['epoch']} (Loss: {worst_epoch['average_loss']:.2f})\n")
                f.write(f"Improvement: {worst_epoch['average_loss'] / best_epoch['average_loss']:.1f}x better\n\n")

        # Detailed Epoch Results
        f.write("DETAILED EPOCH RESULTS\n")
        f.write("=" * 50 + "\n\n")

        for epoch_data in self.epoch_summaries:
            f.write(f"Epoch {epoch_data['epoch']}:\n")
            f.write(f"  Duration: {epoch_data['duration']:.1f} seconds\n")
            f.write(f"  Average Loss: {epoch_data['average_loss']:.4f}\n")
            f.write(f"  Total Loss: {epoch_data['total_loss']:.4f}\n")

            if 'best_protein' in epoch_data:
                f.write(
                    f"  Best Protein: {epoch_data['best_protein']['id']} (Loss: {epoch_data['best_protein']['loss']:.2f})\n")
                f.write(
                    f"  Worst Protein: {epoch_data['worst_protein']['id']} (Loss: {epoch_data['worst_protein']['loss']:.2f})\n")

            # Memory stats
            if epoch_data['memory_stats']:
                f.write(f"  Memory Stats:\n")
                for key, value in epoch_data['memory_stats'].items():
                    f.write(f"    {key}: {value}\n")

            # Individual protein results
            f.write(f"  Protein Results:\n")
            for protein_id, result in epoch_data['protein_results'].items():
                f.write(f"    {protein_id}: Total={result['total_loss']:.2f}")
                if result['msa_loss'] is not None:
                    f.write(f", MSA={result['msa_loss']:.2f}, Pair={result['pair_loss']:.2f}")
                if result['time_taken_ms'] is not None:
                    f.write(f", Time={result['time_taken_ms']:.1f}ms")
                f.write("\n")

            f.write("\n")

        # Final model info
        if final and hasattr(self, 'final_model_path'):
            f.write("FINAL MODEL\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model saved to: {self.final_model_path}\n")
            f.write(
                f"Total training time: {(self.training_complete_time - datetime.datetime.now()).total_seconds():.1f} seconds\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")