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
        print(f"❌ Training script not found: {train_script}")
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
    print(f"\n🖥️  Command: {' '.join(cmd)}")
    print("=" * 80)

    # Run training
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
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

    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (3 epochs each)')

    # Memory options
    parser.add_argument('--small-memory', action='store_true',
                        help='Use memory-optimized settings')

    # Device selection
    parser.add_argument('device', nargs='?', choices=['cpu', 'cuda'], default=None,
                        help='Force device selection')

    args = parser.parse_args()

    # Print header
    print("Neural ODE Training Runner - Updated for Command Line Arguments")
    print("Features: Blocks 0→48 only, Adjoint backprop, LR Scheduling, Early Stopping, Memory optimized")
    print("")

    # Device configuration
    if args.device == 'cpu':
        device = 'cpu'
        print("🖥️  Forced CPU mode")
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA requested but not available!")
            return 1
        device = 'cuda'
        print("🎮 Forced CUDA mode")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Auto-detected device: {device}")

    # Build data directories
    data_dirs = [
        f"{args.data_dir}/complete_blocks",
        f"{args.data_dir}/endpoint_blocks",
    ]

    # Quick test mode overrides
    if args.quick_test:
        args.max_epochs = 3
        args.prelim_max_epochs = 3
        print("🚀 QUICK TEST MODE: 3 epochs only")

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

        # Memory optimization (always enabled in hardcoded script)
        "sequential_loading": True,
        "aggressive_memory_cleanup": True,
        "restore_best_weights": True,
    }

    # Small memory mode
    if args.small_memory:
        config.update({
            "max_cluster_size": 128,
            "chunk_size": 1,
            "aggressive_memory_cleanup": True,
        })
        print("💾 Small memory mode enabled")

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prelim_tag = "with_prelim" if use_prelim else "no_prelim"
    memory_tag = "_smallmem" if args.small_memory else ""
    device_tag = f"_{device}" if args.device else ""

    experiment_name = f"{timestamp}_training_{prelim_tag}{memory_tag}{device_tag}"
    config["experiment_name"] = experiment_name
    config["output_dir"] = "trained_models"

    # Show configuration
    print(f"\n📋 CONFIGURATION:")
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
    print(f"\n🔍 Checking data directories...")
    valid_dirs = []
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f"   ✅ {data_dir}")
            valid_dirs.append(data_dir)
        else:
            print(f"   ❌ {data_dir} (not found)")

    if not valid_dirs:
        print(f"\n❌ No valid data directories found!")
        return 1

    # Update config with only valid directories
    config["data_dirs"] = valid_dirs

    # Verify splits directory
    if not Path(args.splits_dir).exists():
        print(f"\n❌ Splits directory not found: {args.splits_dir}")
        return 1

    # Start training
    print(f"\n🚀 Starting training...")
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
            print(f"\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with code: {result_code}")

        print(f"⏱️  Total time: {hours}h {minutes}m {seconds}s")

        output_dir = Path("trained_models")
        print(f"\n📁 Check output directory: {output_dir}")
        print(f"🔍 Check the training log for details: {output_dir}/{experiment_name}.txt")

        return result_code

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


if __name__ == "__main__":
    print("Neural ODE Training Runner - Command Line Version")
    print("Features: Blocks 0→48 only, Adjoint backprop, LR Scheduling, Early Stopping, Memory optimized")
    print("")
    print("Usage:")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --max-epochs 10")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --quick-test")
    print("  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --no-prelim")
    print(
        "  python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --prelim-max-epochs 5 --max-epochs 10")
    print("")
    sys.exit(main())