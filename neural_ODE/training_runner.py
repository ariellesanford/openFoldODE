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
    experiment_name = f"adjoint_training_{timestamp}"

    # Configuration - simplified for adjoint method
    config = {
        'data_dir': str(data_dir),
        'splits_dir': str(splits_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 100,
        'learning_rate': 1e-3,
        'reduced_cluster_size': 128,
        'hidden_dim': 128,
        'integrator': 'rk4',
        'use_fast_ode': True,
        'use_amp': torch.cuda.is_available(),
        'output_dir': str(output_dir),
        'experiment_name': experiment_name,
        'max_residues': 1000,
        # Enhanced features
        'lr_patience': 3,
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

    # Adjust cluster size if specified
    if '--small-cluster' in sys.argv:
        config['reduced_cluster_size'] = 16
        print("🔧 Using small cluster size (16)")
    elif '--large-cluster' in sys.argv:
        config['reduced_cluster_size'] = 128
        print("🔧 Using large cluster size (128)")

    # Adjust max residues if specified
    if '--small-proteins' in sys.argv:
        config['max_residues'] = 100
        print("🔧 Using small proteins (≤100 residues)")
    elif '--medium-proteins' in sys.argv:
        config['max_residues'] = 300
        print("🔧 Using medium proteins (≤300 residues)")
    elif '--large-proteins' in sys.argv:
        config['max_residues'] = 500
        print("🔧 Using large proteins (≤500 residues)")

    # Adjust model size if specified
    if '--small-model' in sys.argv:
        config['hidden_dim'] = 32
        print("🔧 Using small model (hidden_dim=32)")
    elif '--large-model' in sys.argv:
        config['hidden_dim'] = 128
        print("🔧 Using large model (hidden_dim=128)")

    # Adjust integrator if specified
    if '--dopri5' in sys.argv:
        config['integrator'] = 'dopri5'
        print("🔧 Using dopri5 integrator")
    elif '--euler' in sys.argv:
        config['integrator'] = 'euler'
        print("🔧 Using euler integrator")

    # Adjust learning rate if specified
    if '--low-lr' in sys.argv:
        config['learning_rate'] = 1e-4
        print("🔧 Using low learning rate (1e-4)")
    elif '--high-lr' in sys.argv:
        config['learning_rate'] = 1e-2
        print("🔧 Using high learning rate (1e-2)")

    # Quick test mode
    if '--quick-test' in sys.argv:
        config['epochs'] = 3
        config['max_residues'] = 100
        config['reduced_cluster_size'] = 16
        print("🔧 Quick test mode: 3 epochs, small proteins, small clusters")

    # Build command
    cmd = [sys.executable, str(training_script)]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print("🚀 Starting Simplified Neural ODE Training with Adjoint Method")
    print(f"📁 Data: {config['data_dir']}")
    print(f"📂 Splits: {config['splits_dir']}")
    print(f"💻 Device: {config['device']}")
    print(f"🔧 Config: LR={config['learning_rate']}, Epochs={config['epochs']}")
    print(f"📦 Model: cluster_size={config['reduced_cluster_size']}, hidden_dim={config['hidden_dim']}")
    print(f"🧮 Method: Adjoint method (automatic in torchdiffeq)")
    print(f"🎯 Loss: Only blocks 0→48 transformation")
    print(f"📏 Max residues: {config['max_residues']}")
    print(f"🔄 Integrator: {config['integrator']}")
    print(f"📉 LR Scheduling: patience={config['lr_patience']}, factor={config['lr_factor']}")
    print(f"🛑 Early Stopping: patience={config['early_stopping_patience']}")
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

        # Stream output to console in real-time
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

                        # Look for method confirmation
                        if 'adjoint_0_to_48' in content:
                            print("🧮 Confirmed: Adjoint method used for 0→48 transformation")
                except:
                    pass
        else:
            print("❌ Training failed!")
            print(f"🔍 Check the training log for details: {output_dir}/{experiment_name}.txt")

        # List key files created
        print(f"\n📁 Files created in {output_dir}:")
        for file in sorted(output_dir.glob(f"{experiment_name}*")):
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
    print("Simplified Neural ODE Training Runner with Adjoint Method")
    print("Features: Blocks 0→48 only, Adjoint backprop, LR Scheduling, Early Stopping")
    print("")
    print("Usage:")
    print("  python training_runner_june3.py                  # Default training")
    print("  python training_runner_june3.py --quick-test     # Quick 3-epoch test")
    print("  python training_runner_june3.py --small-proteins # Use proteins ≤100 residues")
    print("  python training_runner_june3.py --medium-proteins# Use proteins ≤300 residues")
    print("  python training_runner_june3.py --large-proteins # Use proteins ≤500 residues")
    print("  python training_runner_june3.py --small-cluster  # Use cluster_size=16")
    print("  python training_runner_june3.py --large-cluster  # Use cluster_size=128")
    print("  python training_runner_june3.py --small-model    # Use hidden_dim=32")
    print("  python training_runner_june3.py --large-model    # Use hidden_dim=128")
    print("  python training_runner_june3.py --dopri5         # Use dopri5 integrator")
    print("  python training_runner_june3.py --euler          # Use euler integrator")
    print("  python training_runner_june3.py --low-lr         # Use learning_rate=1e-4")
    print("  python training_runner_june3.py --high-lr        # Use learning_rate=1e-2")
    print("  python training_runner_june3.py cpu              # Force CPU")
    print("")
    print("🎯 Key Features:")
    print("  🧮 Adjoint method for memory-efficient backpropagation")
    print("  🎯 Only uses blocks 0 and 48 (simplified loss)")
    print("  📉 Automatic learning rate reduction on validation plateau")
    print("  🛑 Early stopping with best model weight restoration")
    print("  📊 Real-time validation monitoring")
    print("  💾 Structured training logs")
    print("  🔄 Multiple ODE integrator options")
    print("")

    sys.exit(main())