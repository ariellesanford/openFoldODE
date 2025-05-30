#!/usr/bin/env python3
"""
PyCharm-friendly Neural ODE Training Runner
Simplified Python version of run_training.sh with concise output
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


class TrainingRunner:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.data_dir = self.script_dir / "data" / "complete_blocks"
        self.output_dir = self.script_dir / "outputs"
        self.training_script = self.script_dir / "train_evoformer_ode.py"

        # Default configuration - matches bash script exactly
        self.config = {
            # Core settings
            'cpu_only': False,
            'test_protein': '',
            'use_fast_ode': True,
            'epochs': 25,
            'learning_rate': 1e-3,  # Start with 1e-5 for stability, can increase to 1e-4 or 1e-3 if stable

            # Memory optimizations (keep your existing settings)
            'memory_split_size': 128,
            'reduced_cluster_size': 32,  # Start small for stability
            'reduced_hidden_dim': 32,  # Start small for stability
            'num_time_points': 5,  # Start small for stability
            'batch_size': 1,
            'integrator': 'euler',  # Most stable
            'gradient_accumulation': 1,
            'chunk_size': 0,

            # Enable/disable features
            'use_amp': True,
            'use_checkpoint': True,
            'monitor_memory': True,
            'clean_memory': False,
            'reduced_precision': True,  # For stability
        }

    def setup_output_dir(self):
        """Create output directory and return timestamped filename"""
        self.output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"training_output_{timestamp}.txt"

    def build_command(self):
        """Build the training command"""
        python_path = sys.executable

        cmd = [
            python_path,
            str(self.training_script),
            "--data_dir", str(self.data_dir),
            "--output_dir", str(self.output_dir),
            "--learning_rate", str(self.config['learning_rate']),
            "--memory_split_size", str(self.config['memory_split_size']),
            "--reduced_cluster_size", str(self.config['reduced_cluster_size']),
            "--reduced_hidden_dim", str(self.config['reduced_hidden_dim']),
            "--num_time_points", str(self.config['num_time_points']),
            "--batch_size", str(self.config['batch_size']),
            "--integrator", self.config['integrator'],
            "--gradient_accumulation", str(self.config['gradient_accumulation']),
            "--chunk_size", str(self.config['chunk_size']),
            "--epochs", str(self.config['epochs'])
        ]

        # Add test protein if specified
        if self.config['test_protein']:
            cmd.extend(["--test-protein", self.config['test_protein']])

        # Add boolean flags - matches all bash script flags exactly
        bool_flags = {
            'cpu_only': '--cpu-only',
            'use_fast_ode': '--use_fast_ode',
            'use_amp': '--use_amp',
            'use_checkpoint': '--use_checkpoint',
            'monitor_memory': '--monitor_memory',
            'clean_memory': '--clean_memory',
            'reduced_precision': '--reduced_precision_integration',
        }

        for key, flag in bool_flags.items():
            if self.config[key]:
                cmd.append(flag)
            else:
                cmd.append(f"--no-{flag.lstrip('--')}")

        return cmd

    def print_config(self, output_file):
        """Print comprehensive configuration matching bash script"""
        config_text = f"""
🚀 NEURAL ODE TRAINING STARTED
{'=' * 50}
📊 ACTIVE CONFIGURATION:
   Device: {'CPU-ONLY' if self.config['cpu_only'] else 'CUDA'}
   Data directory: {self.data_dir}
   Output directory: {self.output_dir}
   Test protein: {self.config['test_protein'] if self.config['test_protein'] else 'all proteins'}
   Fast ODE: {'Enabled' if self.config['use_fast_ode'] else 'Disabled'}

🎛️  TRAINING PARAMETERS:
   Epochs: {self.config['epochs']}
   Learning Rate: {self.config['learning_rate']}
   Time Points: {self.config['num_time_points']} (original: 49)
   Integrator: {self.config['integrator']}
   Batch Size: {self.config['batch_size']}

💾 MEMORY OPTIMIZATIONS:
   Memory Split Size: {self.config['memory_split_size']} MB
   Mixed Precision (AMP): {'Enabled' if self.config['use_amp'] else 'Disabled'}
   Gradient Checkpointing: {'Enabled' if self.config['use_checkpoint'] else 'Disabled'}
   Gradient Accumulation: {self.config['gradient_accumulation']} steps
   Integration Chunking: {'Enabled' if self.config['chunk_size'] > 0 else 'Disabled'}
   Reduced Precision Integration: {'Enabled' if self.config['reduced_precision'] else 'Disabled'}
   Aggressive Memory Cleaning: {'Enabled' if self.config['clean_memory'] else 'Disabled'}
   Memory Usage Monitoring: {'Enabled' if self.config['monitor_memory'] else 'Disabled'}

🧬 MODEL CONFIGURATION:
   Reduced Cluster Size: {self.config['reduced_cluster_size']} (original: 128)
   Reduced Hidden Dimension: {self.config['reduced_hidden_dim']} (default: 128)

⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}
"""
        print(config_text)

        with open(output_file, 'w') as f:
            f.write(config_text)

    def run_training(self, cmd, output_file):
        """Run the training process with concise output"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Track important metrics
            current_epoch = 0
            current_protein = ""
            losses = []

            with open(output_file, 'a') as f:
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue

                    # Write everything to file
                    f.write(line + '\n')
                    f.flush()

                    # Filter important lines for console
                    show_line = False

                    if any(keyword in line.lower() for keyword in [
                        'epoch', 'loss:', 'processing protein', 'error', 'failed',
                        'success', 'completed', 'memory', 'cuda', 'checkpoint'
                    ]):
                        show_line = True

                    # Extract key metrics
                    if 'Epoch' in line and 'Average Loss:' in line:
                        parts = line.split('Average Loss:')
                        if len(parts) > 1:
                            try:
                                loss = float(parts[1].strip())
                                losses.append(loss)
                                current_epoch += 1
                                print(f"📈 Epoch {current_epoch}: Loss = {loss:.2f}")
                            except:
                                pass
                    elif 'Processing protein' in line:
                        protein_name = line.split('Processing protein')[-1].strip()
                        if protein_name != current_protein:
                            current_protein = protein_name
                            print(f"🧬 {protein_name}")
                    elif '- Loss:' in line:
                        try:
                            loss_val = float(line.split('Loss:')[-1].strip())
                            print(f"   Loss: {loss_val:.1f}")
                        except:
                            pass
                    elif show_line:
                        # Show other important lines with simplified formatting
                        if 'CUDA' in line or 'Memory' in line:
                            print(f"💾 {line}")
                        elif 'Error' in line or 'Failed' in line:
                            print(f"❌ {line}")
                        elif 'Success' in line or 'Completed' in line:
                            print(f"✅ {line}")

            process.wait()
            return process.returncode

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            process.terminate()
            return 1
        except Exception as e:
            print(f"❌ Error running training: {e}")
            return 1

    def print_summary(self, return_code, output_file, start_time):
        """Print concise training summary"""
        end_time = datetime.now()
        duration = end_time - start_time

        summary = f"""
{'=' * 50}
🎯 TRAINING COMPLETED
Status: {'✅ Success' if return_code == 0 else '❌ Failed'}
Duration: {duration}
Output: {output_file}
{'=' * 50}
"""
        print(summary)

        with open(output_file, 'a') as f:
            f.write(summary)

    def run(self, **kwargs):
        """Main execution function"""
        # Update config with any provided kwargs
        self.config.update(kwargs)

        # Setup
        output_file = self.setup_output_dir()
        cmd = self.build_command()
        start_time = datetime.now()

        # Run training
        self.print_config(output_file)
        return_code = self.run_training(cmd, output_file)
        self.print_summary(return_code, output_file, start_time)

        return return_code == 0


def main():
    """Command line interface with all bash script options"""
    parser = argparse.ArgumentParser(description='Neural ODE Training Runner')

    # Core parameters
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--test-protein', type=str, default='', help='Specific protein ID to test')

    # Memory optimization options
    parser.add_argument('--memory-split-size', type=int, default=128, help='Memory split size (MB)')
    parser.add_argument('--reduced-cluster-size', type=int, default=32, help='Maximum cluster size')
    parser.add_argument('--reduced-hidden-dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--num-time-points', type=int, default=5, help='Number of integration time points')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for time steps')
    parser.add_argument('--integrator', choices=['dopri5', 'rk4', 'euler'], default='euler',
                        help='ODE integrator method')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--chunk-size', type=int, default=0, help='Chunk size for integration')

    # Boolean flags
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU mode')
    parser.add_argument('--use-fast-ode', action='store_true', default=True, help='Use fast ODE implementation')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    parser.add_argument('--use-checkpoint', action='store_true', default=True, help='Use gradient checkpointing')
    parser.add_argument('--monitor-memory', action='store_true', default=True, help='Monitor memory usage')
    parser.add_argument('--clean-memory', action='store_true', help='Aggressive memory cleaning')
    parser.add_argument('--reduced-precision', action='store_true', default=True,
                        help='Use reduced precision integration')

    args = parser.parse_args()

    # Create runner and execute
    runner = TrainingRunner()
    success = runner.run(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_protein=args.test_protein,
        memory_split_size=args.memory_split_size,
        reduced_cluster_size=args.reduced_cluster_size,
        reduced_hidden_dim=args.reduced_hidden_dim,
        num_time_points=args.num_time_points,
        batch_size=args.batch_size,
        integrator=args.integrator,
        gradient_accumulation=args.gradient_accumulation,
        chunk_size=args.chunk_size,
        cpu_only=args.cpu_only,
        use_fast_ode=args.use_fast_ode,
        use_amp=args.use_amp,
        use_checkpoint=args.use_checkpoint,
        monitor_memory=args.monitor_memory,
        clean_memory=args.clean_memory,
        reduced_precision=args.reduced_precision,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # For PyCharm: can run directly or with custom config
    if len(sys.argv) == 1:
        # Default run for PyCharm - matches bash script stable settings
        print("🚀 Running with default configuration (optimized for stability)...")
        runner = TrainingRunner()

        # Use the same stable settings from bash script
        runner.config.update({
            'cpu_only': False,
            'test_protein': '',  # Leave empty to test all proteins
            'use_fast_ode': True,
            'epochs': 5,
            'learning_rate': 1e-5,  # Start with 1e-5 for stability, can increase to 1e-4 or 1e-3 if stable
            'memory_split_size': 128,
            'reduced_cluster_size': 64,  # Start small for stability
            'reduced_hidden_dim': 64,  # Start small for stability
            'num_time_points': 49,  # Start small for stability
            'batch_size': 1,
            'integrator': 'rk4',  # Most stable
            'gradient_accumulation': 1,
            'chunk_size': 0,
            'use_amp': True,
            'use_checkpoint': True,
            'monitor_memory': True,
            'clean_memory': False,
            'reduced_precision': True,  # For stability
        })

        success = runner.run()
        print(f"\n{'✅ Training completed successfully!' if success else '❌ Training failed!'}")
    else:
        # Command line mode
        main()