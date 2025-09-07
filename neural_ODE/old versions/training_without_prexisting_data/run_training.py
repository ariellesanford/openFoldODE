#!/usr/bin/env python3
"""
Evoformer ODE Training Runner
Python version of run_training.sh with improved configuration management
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import argparse
import json


class TrainingConfig:
    """Configuration class for training parameters"""

    def __init__(self):
        # Core settings
        self.cpu_only = False
        self.use_fast_ode = True
        self.epochs = 25
        self.learning_rate = 1e-3

        # Memory optimizations
        self.reduced_cluster_size = 32
        self.reduced_hidden_dim = 32
        self.num_time_points = 5
        self.integrator = "euler"
        self.batch_size = 1
        self.gradient_accumulation = 1
        self.chunk_size = 0
        self.memory_split_size = 128

        # Optimizations
        self.use_amp = True
        self.use_checkpoint = True
        self.monitor_memory = True
        self.reduced_precision = True
        self.clean_memory = False

        # Early stopping
        self.patience = 5

        # Optional overrides
        self.test_protein = None  # Set to specific protein ID to test single protein

        # Paths (will be set automatically)
        self.script_dir = Path(__file__).parent.absolute()
        self.data_dir = self.script_dir / "mini_data"
        self.output_dir = self.script_dir / "outputs"

    def from_file(self, config_file):
        """Load configuration from JSON file"""
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        return self

    def save_to_file(self, config_file):
        """Save current configuration to JSON file"""
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in self.__dict__.items()}
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def summary(self):
        """Return a concise summary of key settings"""
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'time_points': self.num_time_points,
            'cluster_size': self.reduced_cluster_size,
            'hidden_dim': self.reduced_hidden_dim,
            'use_amp': self.use_amp,
            'use_fast_ode': self.use_fast_ode,
            'cpu_only': self.cpu_only
        }


class TrainingRunner:
    """Main training runner class"""

    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.config.output_dir / f"training_{self.timestamp}.log"

        # Ensure output directory exists
        self.config.output_dir.mkdir(exist_ok=True)

        # Find Python interpreter
        self.python_path = self._find_python()

        # Training script path
        self.train_script = self.config.script_dir / "train_evoformer_ode.py"

    def _find_python(self):
        """Find the appropriate Python interpreter"""
        # Try conda environment first
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_python = Path(conda_prefix) / "bin" / "python"
            if conda_python.exists():
                return str(conda_python)

        # Fall back to system python
        return shutil.which('python') or sys.executable

    def _check_gpu_memory(self):
        """Check available GPU memory"""
        if self.config.cpu_only:
            return None

        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)

            free_memory = int(result.stdout.strip().split('\n')[0])
            return free_memory
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None

    def _build_command(self):
        """Build the training command"""
        cmd = [
            self.python_path,
            str(self.train_script),
            '--data_dir', str(self.config.data_dir),
            '--output_dir', str(self.config.output_dir),
            '--learning_rate', str(self.config.learning_rate),
            '--reduced_cluster_size', str(self.config.reduced_cluster_size),
            '--reduced_hidden_dim', str(self.config.reduced_hidden_dim),
            '--num_time_points', str(self.config.num_time_points),
            '--integrator', self.config.integrator,
            '--epochs', str(self.config.epochs),
            '--batch_size', str(self.config.batch_size),
            '--gradient_accumulation', str(self.config.gradient_accumulation),
            '--chunk_size', str(self.config.chunk_size),
            '--memory_split_size', str(self.config.memory_split_size),
            '--patience', str(self.config.patience)
        ]

        # Add boolean flags
        bool_flags = [
            ('cpu_only', 'cpu-only'),
            ('use_fast_ode', 'use_fast_ode'),
            ('use_amp', 'use_amp'),
            ('use_checkpoint', 'use_checkpoint'),
            ('monitor_memory', 'monitor_memory'),
            ('reduced_precision', 'reduced_precision_integration'),
            ('clean_memory', 'clean_memory')
        ]

        for attr, flag in bool_flags:
            if getattr(self.config, attr):
                cmd.append(f'--{flag}')
            else:
                cmd.append(f'--no-{flag}')

        # Add optional test protein
        if self.config.test_protein:
            cmd.extend(['--test-protein', self.config.test_protein])

        return cmd

    def _print_startup_info(self):
        """Print concise startup information"""
        print("🚀 Evoformer ODE Training")
        print(f"📁 Data: {self.config.data_dir} → Output: {self.config.output_dir}")

        summary = self.config.summary()
        print(
            f"⚙️  Config: LR={summary['learning_rate']}, Epochs={summary['epochs']}, TimePoints={summary['time_points']}")
        print(f"🧠 Memory: Cluster={summary['cluster_size']}, Hidden={summary['hidden_dim']}, AMP={summary['use_amp']}")
        print(f"📄 Log: {self.log_file}")

        # GPU memory check
        if not self.config.cpu_only:
            gpu_memory = self._check_gpu_memory()
            if gpu_memory is not None:
                print(f"💾 Available GPU Memory: {gpu_memory} MB")
                if gpu_memory < 4000:
                    print("⚠️  Warning: Low GPU memory, monitor for OOM errors")

        print("")

    def _save_run_info(self, cmd, start_time):
        """Save run information to log file"""
        with open(self.log_file, 'w') as f:
            f.write("Evoformer ODE Training Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Started: {start_time}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Configuration: {json.dumps(self.config.summary(), indent=2)}\n")
            f.write("=" * 50 + "\n\n")

    def run(self):
        """Execute the training"""
        # Pre-flight checks
        if not self.train_script.exists():
            print(f"❌ Training script not found: {self.train_script}")
            return False

        if not self.config.data_dir.exists():
            print(f"❌ Data directory not found: {self.config.data_dir}")
            return False

        # Print startup info
        self._print_startup_info()

        # Build command
        cmd = self._build_command()

        # Save run info
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_run_info(cmd, start_time)

        print("⏳ Starting training...")

        try:
            # Execute training with real-time output
            with open(self.log_file, 'a') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                # Stream output in real-time
                for line in process.stdout:
                    print(line.rstrip())  # Print to console
                    log_f.write(line)  # Write to log
                    log_f.flush()  # Ensure immediate write

                # Wait for completion
                process.wait()
                exit_code = process.returncode

                # Log completion
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                completion_msg = f"\nCompleted: {end_time}\nExit code: {exit_code}\n"
                log_f.write(completion_msg)

                # Print final status
                if exit_code == 0:
                    print("✅ Training completed successfully!")
                    print(f"📄 Full log: {self.log_file}")
                    return True
                else:
                    print(f"❌ Training failed (exit code: {exit_code})")
                    print(f"📄 Check log: {self.log_file}")
                    return False

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Error running training: {e}")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evoformer ODE Training Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration options
    parser.add_argument('--config', type=str, help='Load configuration from JSON file')
    parser.add_argument('--save-config', type=str, help='Save current configuration to JSON file')

    # Override common settings
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--test-protein', type=str, help='Test single protein')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode')
    parser.add_argument('--fast-ode', action='store_true', help='Use fast ODE implementation')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')

    # Preset configurations
    parser.add_argument('--preset', choices=['fast', 'balanced', 'quality'],
                        help='Use preset configuration')

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig()

    # Load from file if specified
    if args.config:
        config.from_file(args.config)

    # Apply preset configurations
    if args.preset == 'fast':
        config.learning_rate = 1e-3
        config.epochs = 10
        config.num_time_points = 3
        config.reduced_cluster_size = 16
        config.reduced_hidden_dim = 16
        config.use_fast_ode = True
        config.reduced_precision = True
    elif args.preset == 'balanced':
        config.learning_rate = 1e-4
        config.epochs = 25
        config.num_time_points = 5
        config.reduced_cluster_size = 32
        config.reduced_hidden_dim = 32
    elif args.preset == 'quality':
        config.learning_rate = 1e-5
        config.epochs = 50
        config.num_time_points = 10
        config.reduced_cluster_size = 64
        config.reduced_hidden_dim = 64
        config.use_fast_ode = False
        config.reduced_precision = False

    # Apply command line overrides
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.test_protein:
        config.test_protein = args.test_protein
    if args.cpu_only:
        config.cpu_only = True
        config.use_amp = False
    if args.fast_ode:
        config.use_fast_ode = True
    if args.no_amp:
        config.use_amp = False

    # Save configuration if requested
    if args.save_config:
        config.save_to_file(args.save_config)
        print(f"💾 Configuration saved to: {args.save_config}")
        return

    # Run training
    runner = TrainingRunner(config)
    success = runner.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()