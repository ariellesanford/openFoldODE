#!/usr/bin/env python3
"""
Training runner for Neural ODE model
"""

import os
import sys
import subprocess
import time
import torch
from pathlib import Path
from datetime import datetime
import argparse


def run_training(config: dict) -> int:
    """Run training with given configuration"""

    script_dir = Path(__file__).parent
    train_script = script_dir / "train_evoformer_ode_new.py"

    if not train_script.exists():
        print(f"Training script not found: {train_script}")
        return 1

    # Build command
    cmd = [sys.executable, str(train_script)]

    # Add all config items as command line arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        elif isinstance(value, list):
            cmd.extend([f"--{key.replace('_', '-')}", *[str(v) for v in value]])
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    # Print command for debugging
    print(f"\nCommand: {' '.join(cmd)}")
    print("=" * 80)

    # Run training
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running training: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Neural ODE Training Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments with defaults
    parser.add_argument('--data-dir', type=str, default='/media/visitor/Extreme SSD/data',
                        help='Base data directory containing complete_blocks and endpoint_blocks')
    parser.add_argument('--splits-dir', type=str, default='data_splits/jumbo',
                        help='Directory containing train/val/test splits')

    # Epoch control
    parser.add_argument('--max-epochs', type=int, default=200,
                        help='Maximum epochs for main training (default: 200)')
    parser.add_argument('--prelim-max-epochs', type=int, default=100,
                        help='Maximum epochs for preliminary training (default: 100)')

    # Preliminary training options
    parser.add_argument('--no-prelim', action='store_true',
                        help='Disable preliminary training')
    parser.add_argument('--prelim', action='store_true',
                        help='Enable preliminary training (default)')
    parser.add_argument('--prelim-stride', type=int, default=4,
                        help='Stride for preliminary training blocks (default: 4)')
    parser.add_argument('--prelim-chunk-size', type=int, default=4,
                        help='Chunk size for preliminary training (default: 4)')

    # Device selection
    parser.add_argument('device', nargs='?', choices=['cpu', 'cuda'], default=None,
                        help='Force device selection')

    args = parser.parse_args()

    # Print header
    print("Neural ODE Training Runner - Updated for Command Line Arguments")
    print("Features: Blocks 0->48 only, Adjoint backprop, LR Scheduling, Early Stopping, Memory optimized")
    print("")

    # Device configuration
    if args.device == 'cpu':
        device = 'cpu'
        print("Forced CPU mode")
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA requested but not available!")
            return 1
        device = 'cuda'
        print("Forced CUDA mode")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device}")

    # Build data directories
    data_dirs = [
        f"{args.data_dir}/complete_blocks",
        f"{args.data_dir}/endpoint_blocks",
    ]

    # Determine if using preliminary training
    use_prelim = not args.no_prelim

    # Build configuration
    config = {
        "data_dirs": data_dirs,
        "splits_dir": args.splits_dir,
        "device": device,
        "max_epochs": args.max_epochs,

        # Preliminary training
        "use_preliminary_training": use_prelim,
        "prelim_max_epochs": args.prelim_max_epochs,
        "prelim_stride": args.prelim_stride,
        "prelim_chunk_size": args.prelim_chunk_size,

        # Model configuration (from original)
        "num_layers": 2,
        "hidden_dim": 128,
        "time_embedding_dim": 32,

        # Training configuration (from original)
        "learning_rate": 1e-4,
        "batch_size": 1,
        "chunk_size": 4,
        "val_check_interval": 50,
        "log_interval": 10,
        "save_interval": 50,

        # Early stopping (from original)
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-4,

        # LR scheduler (from original)
        "scheduler_patience": 10,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-6,

        # Memory optimization (always enabled in hardcoded script)
        "sequential_loading": True,
        "aggressive_memory_cleanup": True,
        "restore_best_weights": True,
        "max_cluster_size": 256,

        # ODE solver (from original)
        "solver_method": "dopri5",
        "solver_rtol": 1e-5,
        "solver_atol": 1e-5,
        "solver_options": {"dtype": "float32"},
        "adjoint": True,

        # Paths (from original)
        "checkpoint_path": None,
        "log_dir": "logs",
    }

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prelim_tag = "with_prelim" if use_prelim else "no_prelim"
    device_tag = f"_{device}" if args.device else ""

    experiment_name = f"{timestamp}_training_{prelim_tag}{device_tag}"
    config["experiment_name"] = experiment_name
    config["output_dir"] = "trained_models"

    # Show configuration
    print(f"\nCONFIGURATION:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Splits: {args.splits_dir}")
    print(f"   Device: {device}")
    print(f"   Max epochs: {args.max_epochs}")
    if use_prelim:
        print(f"   Preliminary training: Yes")
        print(f"   - Max epochs: {args.prelim_max_epochs}")
        print(f"   - Stride: {args.prelim_stride}")
        print(f"   - Chunk size: {args.prelim_chunk_size}")
    else:
        print(f"   Preliminary training: No")
    print(f"   Experiment: {experiment_name}")

    # Verify data directories exist
    print(f"\nChecking data directories...")
    valid_dirs = []
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f"   [OK] {data_dir}")
            valid_dirs.append(data_dir)
        else:
            print(f"   [MISSING] {data_dir}")

    if not valid_dirs:
        print(f"\nNo valid data directories found!")
        return 1

    # Update config with only valid directories
    config["data_dirs"] = valid_dirs

    # Verify splits directory
    if not Path(args.splits_dir).exists():
        print(f"\nSplits directory not found: {args.splits_dir}")
        return 1

    # Start training
    print(f"\nStarting training...")
    print(f"   Output will be saved to: trained_models/{experiment_name}_final_model.pt")
    print(f"   Log file: trained_models/{experiment_name}.txt")

    start_time = time.time()

    try:
        result_code = run_training(config)

        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        if result_code == 0:
            print(f"\nTraining completed successfully!")
        else:
            print(f"\nTraining failed with code: {result_code}")

        print(f"Total time: {hours}h {minutes}m {seconds}s")

        output_dir = Path("trained_models")
        print(f"\nCheck output directory: {output_dir}")
        print(f"Check the training log for details: {output_dir}/{experiment_name}.txt")

        return result_code

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running training: {e}")
        return 1


if __name__ == "__main__":
    print("Neural ODE Training Runner - Command Line Version")
    print("Features: Blocks 0->48 only, Adjoint backprop, LR Scheduling, Early Stopping, Memory optimized")
    print("")
    print("Usage:")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --max-epochs 10")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --no-prelim")
    print(
        "  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --prelim-max-epochs 5 --max-epochs 10")
    print("")
    sys.exit(main())