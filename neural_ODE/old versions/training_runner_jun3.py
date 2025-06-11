#!/usr/bin/env python3
"""
Updated training runner with loss mode selection
Focuses on real-time console output and structured training logs only
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
    output_dir = script_dir / "outputs"
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
    experiment_name = f"training_output_{timestamp}"

    # Configuration
    config = {
        'data_dir': str(data_dir),
        'splits_dir': str(splits_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 20,
        'learning_rate': 1e-3,
        'reduced_cluster_size': 32,
        'hidden_dim': 64,
        'integrator': 'rk4',
        'use_fast_ode': True,
        'use_amp': torch.cuda.is_available(),
        'output_dir': str(output_dir),
        'experiment_name': experiment_name,
        'batch_size': 10,
        'max_residues': 100,
        'loss_mode': 'incremental',  # Default loss mode
        # Enhanced features
        'lr_patience': 2,
        'lr_factor': 0.5,
        'min_lr': 1e-6,
        'early_stopping_patience': 7,
        'early_stopping_min_delta': 0.001,
        'restore_best_weights': True
    }

    # Parse command line arguments
    if 'cpu' in sys.argv:
        config['device'] = 'cpu'
        config['use_amp'] = False
    elif 'cuda' in sys.argv:
        config['device'] = 'cuda'
        config['use_amp'] = torch.cuda.is_available()

    # Adjust batch size if specified
    if '--small-batch' in sys.argv:
        config['batch_size'] = 2
    elif '--large-batch' in sys.argv:
        config['batch_size'] = 20

    # NEW: Loss mode selection
    if '--loss-mode-end-to-end' in sys.argv or '--end-to-end' in sys.argv:
        config['loss_mode'] = 'end_to_end'
        print("🎯 Using end-to-end loss mode (0→48)")
    elif '--loss-mode-incremental' in sys.argv or '--incremental' in sys.argv:
        config['loss_mode'] = 'incremental'
        print("🎯 Using incremental loss mode (original)")

    # Add mode handling
    if '--test' in sys.argv:
        config['mode'] = 'testing'
    else:
        config['mode'] = 'training'  # default - includes validation

    if '--test' in sys.argv:
        # Find available proteins to test with
        proteins = []
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.endswith('_evoformer_blocks'):
                protein_id = item.name.replace('_evoformer_blocks', '')
                proteins.append(protein_id)
        if proteins:
            config['test_single_protein'] = proteins[0]
            config['mode'] = 'single_test'

    # Build command
    cmd = [sys.executable, str(training_script)]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print("🚀 Starting Enhanced Neural ODE Training with Dual Loss")
    print(f"📁 Data: {config['data_dir']}")
    print(f"📂 Splits: {config['splits_dir']}")
    print(f"🎯 Mode: {config['mode']}")
    print(f"💻 Device: {config['device']}")
    print(f"🔧 Config: LR={config['learning_rate']}, Epochs={config['epochs']}")
    print(f"📉 LR Scheduling: patience={config['lr_patience']}, factor={config['lr_factor']}")
    print(f"🛑 Early Stopping: patience={config['early_stopping_patience']}")
    print(f"🎯 Loss Mode: {config['loss_mode']} ({'0→48 loss' if config['loss_mode'] == 'end_to_end' else 'incremental loss'} guides learning)")
    print(f"📊 Dual Reporting: Both incremental and 0→48 losses will be computed and reported")
    print(f"📊 Reports will be saved to: {output_dir}/{experiment_name}.txt")
    print("=" * 50)

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

        # Stream output to console in real-time (no file saving)
        for line in process.stdout:
            print(line, end='')  # Print to console immediately

        # Wait for process to complete
        result_code = process.wait()

        print("\n" + "=" * 50)
        if result_code == 0:
            print("✅ Training completed successfully!")
            print(f"📊 Detailed training report: {output_dir}/{experiment_name}.txt")

            # Show training results if available
            training_log = output_dir / f"{experiment_name}.txt"
            if training_log.exists():
                print(f"📈 Training log: {training_log}")
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

                        # Look for dual loss information
                        if 'Loss Mode:' in content:
                            lines = content.split('\n')
                            for line in lines:
                                if 'Loss Mode:' in line:
                                    print(f"🎯 {line.strip()}")
                                    break
                except:
                    pass
        else:
            print("❌ Training failed!")
            print(f"🔍 Check the training log for details: {output_dir}/{experiment_name}.txt")

        # List key files created (excluding console output)
        print(f"\n📁 Files created in {output_dir}:")
        for file in sorted(output_dir.glob(f"{experiment_name}*")):
            # Skip console output files
            if "_console_output.txt" in file.name:
                continue

            size_mb = file.stat().st_size / 1024 / 1024
            if file.suffix == '.pt':
                print(f"  - {file.name} ({size_mb:.1f} MB) [Model checkpoint]")
            elif file.suffix == '.txt':
                print(f"  - {file.name} ({size_mb:.1f} MB) [Training log]")
            else:
                print(f"  - {file.name} ({size_mb:.1f} MB)")

        return result_code

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


if __name__ == "__main__":
    print("Enhanced Neural ODE Training Runner with Dual Loss")
    print("Features: LR Scheduling, Early Stopping, Real-time Monitoring, Dual Loss Computation")
    print("")
    print("Usage:")
    print("  python training_runner.py                        # Training mode with validation, incremental loss")
    print("  python training_runner.py --end-to-end           # Training mode with end-to-end loss (0→48)")
    print("  python training_runner.py --incremental          # Training mode with incremental loss (default)")
    print("  python training_runner.py --test                 # Testing mode")
    print("  python training_runner.py --small-batch          # Use smaller batch size (2)")
    print("  python training_runner.py --large-batch          # Use larger batch size (20)")
    print("  python training_runner.py cpu                    # Force CPU")
    print("")
    print("🎯 Loss Mode Options:")
    print("  --incremental     : Use incremental loss (original approach, default)")
    print("  --end-to-end      : Use end-to-end loss (0→48 transformation)")
    print("")
    print("🎯 Features:")
    print("  📦 Temporal batching approach (simplified)")
    print("  📉 Automatic learning rate reduction on validation plateau")
    print("  🛑 Early stopping with best model weight restoration")
    print("  📊 Real-time validation monitoring")
    print("  🎯 Dual loss computation: Both incremental and 0→48 losses computed and reported")
    print("  🔄 Loss mode selection: Choose which loss guides learning")
    print("  💾 Structured training logs (no console output files)")
    print("")

    sys.exit(main())