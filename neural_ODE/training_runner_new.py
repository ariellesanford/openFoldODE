#!/usr/bin/env python3
"""
Training runner with preliminary training support - Updated for restructured train_evoformer_ode.py
Uses only blocks 0→48 with adjoint method
MODIFIED: Updated for simplified configuration structure
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
from datetime import datetime


def main():
    # Get script directory and set up paths
    script_dir = Path(__file__).parent

    # CHANGED: Support multiple data directories
    data_dirs = [
        Path("/media/visitor/Extreme SSD/data/complete_blocks"),
        Path("/media/visitor/Extreme SSD/data/endpoint_blocks"),
        # Add more directories as needed
    ]

    # NEW: Preliminary training directory (for intermediate blocks)
    prelim_data_dir = Path("/media/visitor/Extreme SSD/data/complete_blocks")

    splits_dir = script_dir / "data_splits" / "1fv5"
    output_dir = script_dir / "trained_models"
    training_script = script_dir / "train_evoformer_ode.py"

    # Check if data directories exist
    valid_data_dirs = []
    for data_dir in data_dirs:
        if data_dir.exists():
            valid_data_dirs.append(str(data_dir))
            print(f"✅ Found data directory: {data_dir}")
        else:
            print(f"⚠️  Data directory not found: {data_dir}")

    if not valid_data_dirs:
        print(f"❌ No valid data directories found!")
        return 1

    if not training_script.exists():
        print(f"❌ Training script not found: {training_script}")
        return 1

    if not splits_dir.exists():
        print(f"❌ Data splits directory not found: {splits_dir}")
        return 1

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_full_ode_with_prelim2"

    # Configuration - simplified for restructured script
    config = {
        'data_dirs': valid_data_dirs,
        'splits_dir': str(splits_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 10000,
        'learning_rate': 1e-3,
        'reduced_cluster_size': 64,
        'hidden_dim': 64,
        'integrator': 'rk4',
        'use_fast_ode': False,
        'max_residues': 450,
        'loss': 'default',
        'use_amp': torch.cuda.is_available(),
        'output_dir': str(output_dir),
        'experiment_name': experiment_name,
        # Enhanced features
        'lr_patience': 3,
        'lr_factor': 0.5,
        'min_lr': 1e-6,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0001,
        'max_time_hours': 17 ,
        # Memory optimizations (aggressive_cleanup is the only configurable one now)
        'aggressive_cleanup': True,
        # NEW: Preliminary training settings
        'enable_preliminary_training': True,  # Set to True to enable
        'prelim_data_dir': str(prelim_data_dir),
        'prelim_block_stride': 4,
        'prelim_max_epochs': 20,
        'prelim_chunk_size': 2,  # Good balance of memory and stability
    }

    # Parse command line arguments
    if 'cpu' in sys.argv:
        config['device'] = 'cpu'
        config['use_amp'] = False
    elif 'cuda' in sys.argv:
        config['device'] = 'cuda'
        config['use_amp'] = torch.cuda.is_available()

    # NEW: Enable preliminary training with command line flag
    if '--with-preliminary' in sys.argv or '--prelim' in sys.argv:
        config['enable_preliminary_training'] = True
        print("🔄 Preliminary training enabled via command line")

    # Disable preliminary training
    if '--no-preliminary' in sys.argv or '--no-prelim' in sys.argv:
        config['enable_preliminary_training'] = False
        print("⏭️  Preliminary training disabled via command line")

    # Quick test mode
    if '--quick-test' in sys.argv:
        config['epochs'] = 3
        config['max_residues'] = 100
        config['reduced_cluster_size'] = 16
        config['prelim_max_epochs'] = 2  # Shorter preliminary training for testing
        print("🔧 Quick test mode: 3 epochs, small proteins, small clusters")

    # Custom preliminary settings via command line
    if '--prelim-stride' in sys.argv:
        try:
            idx = sys.argv.index('--prelim-stride')
            config['prelim_block_stride'] = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Error: --prelim-stride requires an integer argument")
            return 1

    if '--prelim-epochs' in sys.argv:
        try:
            idx = sys.argv.index('--prelim-epochs')
            config['prelim_max_epochs'] = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Error: --prelim-epochs requires an integer argument")
            return 1

    if '--prelim-chunk-size' in sys.argv:
        try:
            idx = sys.argv.index('--prelim-chunk-size')
            config['prelim_chunk_size'] = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("❌ Error: --prelim-chunk-size requires an integer argument")
            return 1

    # Memory optimization flags
    if '--small-memory' in sys.argv:
        config['reduced_cluster_size'] = 32
        config['prelim_chunk_size'] = 2
        config['max_residues'] = 150
        print("💾 Small memory mode: 32 clusters, chunk size 2, ≤150 residues")

    print("🚀 Neural ODE Training Runner - Updated for Restructured Script")
    print(f"📁 Data directories: {valid_data_dirs}")

    if config['enable_preliminary_training']:
        print(f"🔄 Preliminary training enabled:")
        print(f"   Directory: {config['prelim_data_dir']}")
        print(f"   Block stride: {config['prelim_block_stride']}")
        print(f"   Chunk size: {config['prelim_chunk_size']}")
        print(f"   Max epochs: {config['prelim_max_epochs']}")

        # Check if preliminary data directory exists
        if not prelim_data_dir.exists():
            print(f"⚠️  Preliminary data directory not found: {prelim_data_dir}")
            print("   Preliminary training will be skipped")
            config['enable_preliminary_training'] = False
    else:
        print("⏭️  Preliminary training disabled")

    print(f"💻 Device: {config['device']}")
    print(f"🔧 Memory: Aggressive cleanup={config['aggressive_cleanup']}")
    print(f"🧮 Method: Adjoint backpropagation (0→48 blocks only)")
    print(f"📊 Model: {'Fast ODE' if config['use_fast_ode'] else 'Full ODE'}")

    # Build command for restructured script
    cmd = [sys.executable, str(training_script)]
    for key, value in config.items():
        if key == 'data_dirs':
            # Handle multiple data directories
            cmd.extend(['--data_dirs'] + value)
        elif isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    # Show what will be executed
    print(f"\n🔧 Training configuration:")
    print(f"   Main training: {config['epochs']} epochs max")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Time limit: {config.get('max_time_hours', 'None')} hours")
    print(f"   Cluster size: {config['reduced_cluster_size']}")
    if config['enable_preliminary_training']:
        print(f"   Preliminary: {config['prelim_max_epochs']} epochs on stride-{config['prelim_block_stride']} blocks")
        print(f"   Chunk size: {config['prelim_chunk_size']} blocks per chunk")

    try:
        # Start the process with real-time output streaming
        process = subprocess.Popen(
            cmd,
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Stream output to console in real-time
        for line in process.stdout:
            print(line, end='')  # Print to console immediately

        # Wait for process to complete
        result_code = process.wait()

        print("\n" + "=" * 50)
        if result_code == 0:
            print("✅ Training completed successfully!")

            # Show training results if available
            training_log = output_dir / f"{experiment_name}.txt"
            if training_log.exists():
                # Try to extract key results from log
                try:
                    with open(training_log, 'r') as f:
                        content = f.read()

                        # Look for preliminary training results
                        if config['enable_preliminary_training'] and 'PRELIMINARY TRAINING PHASE' in content:
                            print("🔄 Preliminary training phase detected in log")
                            if 'Preliminary training completed' in content:
                                print("✅ Preliminary training completed successfully")

                        # Look for early stopping or final results
                        if '🛑 Early stopping triggered' in content:
                            print("🛑 Training stopped early due to validation criteria")
                        elif 'Best validation loss:' in content:
                            # Try to extract best validation loss
                            lines = content.split('\n')
                            for line in lines:
                                if 'Best validation loss:' in line:
                                    print(f"🏆 {line.strip()}")
                                    break

                        if 'Learning rate reductions:' in content:
                            lines = content.split('\n')
                            for line in lines:
                                if 'Learning rate reductions:' in line:
                                    print(f"📉 {line.strip()}")
                                elif 'Final learning rate:' in line:
                                    print(f"🎛️  {line.strip()}")
                                    break

                        # Look for method confirmation
                        if 'adjoint_0_to_48' in content:
                            print("🧮 Confirmed: Adjoint method used for 0→48 transformation")
                except:
                    pass

            print(f"\n📊 Training log: {output_dir}/{experiment_name}.txt")
            print(f"🤖 Model saved to: {output_dir}/{experiment_name}_final_model.pt")
        else:
            print("❌ Training failed!")
            print(f"🔍 Check the training log for details: {output_dir}/{experiment_name}.txt")

        return result_code

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


if __name__ == "__main__":
    print("Neural ODE Training Runner - Updated for Restructured Script")
    print("Features: Blocks 0→48 only, Adjoint backprop, LR Scheduling, Early Stopping, Memory optimized")
    print("NEW: Updated for simplified configuration structure")
    print("")
    print("Usage:")
    print("  python training_runner.py                          # Standard training (with preliminary)")
    print("  python training_runner.py --no-prelim              # Disable preliminary training")
    print("  python training_runner.py --prelim                 # Enable preliminary training (default)")
    print("  python training_runner.py --quick-test             # Quick 3-epoch test")
    print("  python training_runner.py --prelim-stride 8        # Custom stride")
    print("  python training_runner.py --prelim-epochs 50       # Custom prelim epochs")
    print("  python training_runner.py --prelim-chunk-size 2    # Custom chunk size")
    print("  python training_runner.py --small-memory           # Memory-optimized settings")
    print("  python training_runner.py cpu                      # Force CPU")
    print("  python training_runner.py cuda                     # Force CUDA")
    print("")
    print("Preliminary Training:")
    print("  • Runs BEFORE main 0→48 training")
    print("  • Uses strided intermediate blocks (e.g., 0→4→8→12→16→20→24→28→32→36→40→44→48)")
    print("  • Helps initialize model with intermediate dynamics")
    print("  • Configurable stride length, epoch count, and chunk size")
    print("  • Uses same validation set for early stopping")
    print("  • Same early stopping delta as main training (hardcoded)")
    print("")
    print("Memory Optimization:")
    print("  • Sequential loading always enabled (hardcoded)")
    print("  • Best weights restoration always enabled (hardcoded)")
    print("  • Configurable: cluster_size, chunk_size, aggressive_cleanup")
    print("  • --small-memory flag for memory-constrained GPUs")
    print("")
    print("Data directory search order:")
    print("  1. /media/visitor/Extreme SSD/data/complete_blocks")
    print("  2. /media/visitor/Extreme SSD/data/endpoint_blocks")
    print("  Preliminary: /media/visitor/Extreme SSD/data/complete_blocks")
    print("  (Edit script to add more directories)")
    print("")
    sys.exit(main())