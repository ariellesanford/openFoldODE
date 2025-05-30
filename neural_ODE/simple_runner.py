#!/usr/bin/env python3
"""
Simple runner for the improved Neural ODE training
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
    data_dir = script_dir / "data" / "complete_blocks"
    output_dir = script_dir / "outputs"
    training_script = script_dir / "improved_train_evoformer_ode.py"

    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please make sure you have generated the Evoformer blocks first")
        return 1

    # Check if training script exists
    if not training_script.exists():
        print(f"❌ Training script not found: {training_script}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"training_output_{timestamp}.txt"

    # Configuration for stable training
    config = {
        'data_dir': str(data_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 20,
        'learning_rate': 1e-3,  # Good starting point
        'reduced_cluster_size': 32,  # Small for stability
        'hidden_dim': 64,
        'integrator': 'rk4',  # Stable integrator
        'use_fast_ode': True,
        'use_amp': torch.cuda.is_available(),  # Enable AMP only if CUDA available
        'output_dir': str(output_dir),
        'experiment_name': experiment_name,

        # Choose ONE approach - batching, striding, OR simple (all three options)
        'batch_size': 20,        # Use temporal batching with size 2
        # 'block_stride': 8,      # OR use every 8th block (8 divides 48)
        # Neither batch_size nor block_stride = simple approach (all blocks at once)
        'max_residues': 200,  # Skip proteins larger than 200 residues to avoid OOM
    }

    # Override device if specified in command line
    if 'cpu' in sys.argv:
        config['device'] = 'cpu'
        config['use_amp'] = False
    elif 'cuda' in sys.argv:
        config['device'] = 'cuda'
        config['use_amp'] = torch.cuda.is_available()

    # Override approach if specified
    if '--batch' in sys.argv:
        config['batch_size'] = 2  # Use batching approach
    elif '--stride' in sys.argv:
        config['block_stride'] = 8  # Use stride approach
    # Otherwise use simple approach (neither batch_size nor block_stride)

    # Check for test mode
    if '--test' in sys.argv:
        # Find first protein for testing
        proteins = []
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.endswith('_evoformer_blocks'):
                protein_id = item.name.replace('_evoformer_blocks', '')
                proteins.append(protein_id)

        if proteins:
            config['test_single_protein'] = proteins[0]
            print(f"🧪 Test mode: using protein {proteins[0]}")
        else:
            print("❌ No proteins found for testing")
            return 1

    # Build command
    cmd = [sys.executable, str(training_script)]

    # Add arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print("🚀 Starting Neural ODE Training")
    print(f"📁 Data: {config['data_dir']}")
    print(f"💻 Device: {config['device']}")
    print(f"🔧 Config: LR={config['learning_rate']}, Epochs={config['epochs']}")

    if 'test_single_protein' in config:
        print(f"🧪 Testing single protein: {config['test_single_protein']}")

    print(f"⚡ Command: {' '.join(cmd)}")
    print("=" * 50)

    # Run training
    try:
        result = subprocess.run(cmd, cwd=script_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


if __name__ == "__main__":
    print("Neural ODE Training Runner")
    print("Usage:")
    print("  python simple_runner.py           # Simple approach (all blocks at once)")
    print("  python simple_runner.py --batch   # Batched approach (batch_size=2)")
    print("  python simple_runner.py --stride  # Strided approach (every 8th block)")
    print("  python simple_runner.py --test    # Test on single protein")
    print("  python simple_runner.py cuda      # Force CUDA")
    print("  python simple_runner.py cpu       # Force CPU")
    print("")
    print("Approaches:")
    print("  Simple:   Use all 49 blocks at once (highest memory, best accuracy)")
    print("  Batching: Process 49 blocks in chunks of 2 (medium memory, good accuracy)")
    print("  Striding: Use blocks 0,8,16,24,32,40,48 (lowest memory, fast)")
    print("")

    exit_code = main()

    if exit_code == 0:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed!")

    sys.exit(exit_code)