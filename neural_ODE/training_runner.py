#!/usr/bin/env python3
"""
Simplified training runner for the new train_evoformer_ode.py
Uses only blocks 0→48 with adjoint method
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
    data_dir = Path("/media/visitor/Extreme SSD/data/complete_blocks")
    splits_dir = script_dir / "data_splits" / "mini"
    output_dir = script_dir / "trained_models"
    training_script = script_dir / "train_evoformer_ode.py"

    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
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
    experiment_name = f"adjoint_training_{timestamp}"

    # Configuration - simplified for adjoint method
    config = {
        'data_dir': str(data_dir),
        'splits_dir': str(splits_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 1000,
        'learning_rate': 1e-3,
        'reduced_cluster_size': 64,
        'hidden_dim': 64,
        'integrator': 'rk4',
        'use_fast_ode': False,
        'use_amp': torch.cuda.is_available(),
        'output_dir': str(output_dir),
        'experiment_name': experiment_name,
#        'max_residues': 100,
        # Enhanced features
        'lr_patience': 3,
        'lr_factor': 0.5,
        'min_lr': 1e-6,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0001,
        'restore_best_weights': True,
        'max_time_hours': 15.5,
        # Memory optimizations
        'use_sequential_loading': True,  # New memory optimization
        'aggressive_cleanup': True  # New memory optimization
    }

    # Parse command line arguments
    if 'cpu' in sys.argv:
        config['device'] = 'cpu'
        config['use_amp'] = False
    elif 'cuda' in sys.argv:
        config['device'] = 'cuda'
        config['use_amp'] = torch.cuda.is_available()

    # Quick test mode
    if '--quick-test' in sys.argv:
        config['epochs'] = 3
        config['max_residues'] = 100
        config['reduced_cluster_size'] = 16
        print("🔧 Quick test mode: 3 epochs, small proteins, small clusters")

    print("🚀 Neural ODE Training Runner")
    print(f"📁 Data: {data_dir}")
    print(f"💻 Device: {config['device']}")
    print(
        f"🔧 Memory: Sequential loading={config['use_sequential_loading']}, Aggressive cleanup={config['aggressive_cleanup']}")
    print(f"🎯 Method: Adjoint backpropagation (0→48 blocks only)")

    # Build command
    cmd = [sys.executable, str(training_script)]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

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
    print("Neural ODE Training Runner with Memory Optimizations")
    print("Features: Blocks 0→48 only, Adjoint backprop, LR Scheduling, Early Stopping, Memory optimized")
    print("")
    print("Usage:")
    print("  python training_runner.py                  # Default training")
    print("  python training_runner.py --quick-test     # Quick 3-epoch test")
    print("  python training_runner.py cpu              # Force CPU")
    print("  python training_runner.py cuda             # Force CUDA")
    print("")
    print("Memory optimizations:")
    print("  - Sequential tensor loading (avoid 4x memory spike)")
    print("  - Aggressive GPU memory cleanup")
    print("  - Mixed precision training")
    print("  - Immediate tensor deletion")
    print("")
    sys.exit(main())