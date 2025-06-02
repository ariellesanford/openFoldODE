#!/usr/bin/env python3
"""
Fixed simple runner - captures subprocess output properly
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
    output_dir = script_dir / "outputs"
    training_script = script_dir / "improved_train_evoformer_ode.py"

    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1

    if not training_script.exists():
        print(f"❌ Training script not found: {training_script}")
        return 1

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"training_output_{timestamp}"

    # Configuration
    config = {
        'data_dir': str(data_dir),
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
        'batch_size': 20,
        'max_residues': 200,
    }

    # Parse command line arguments
    if 'cpu' in sys.argv:
        config['device'] = 'cpu'
        config['use_amp'] = False
    elif 'cuda' in sys.argv:
        config['device'] = 'cuda'
        config['use_amp'] = torch.cuda.is_available()

    if '--batch' in sys.argv:
        config['batch_size'] = 2
        if 'block_stride' in config:
            del config['block_stride']
    elif '--stride' in sys.argv:
        config['block_stride'] = 8
        if 'batch_size' in config:
            del config['batch_size']

    if '--test' in sys.argv:
        proteins = []
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.endswith('_evoformer_blocks'):
                protein_id = item.name.replace('_evoformer_blocks', '')
                proteins.append(protein_id)
        if proteins:
            config['test_single_protein'] = proteins[0]

    # Build command
    cmd = [sys.executable, str(training_script)]
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
    print(f"📊 Reports will be saved to: {output_dir}/{experiment_name}")
    print("=" * 50)

    # FIXED: Stream output in real-time to console, save to file at end
    console_output_file = output_dir / f"{experiment_name}_console_output.txt"

    try:
        # Start the process with streaming output
        process = subprocess.Popen(
            cmd,
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Collect output while streaming to console
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print to console immediately
            output_lines.append(line)  # Collect for file

        # Wait for process to complete
        result_code = process.wait()

        # Write collected output to file
        with open(console_output_file, 'w') as f:
            f.writelines(output_lines)

        print("\n" + "=" * 50)
        if result_code == 0:
            print("✅ Training completed successfully!")
            print(f"📊 Detailed training report: {output_dir}/{experiment_name}")
            print(f"📄 Console output: {console_output_file}")
        else:
            print("❌ Training failed!")
            print(f"📄 Check console output: {console_output_file}")

        # List all files created
        print(f"\n📁 Files created in {output_dir}:")
        for file in sorted(output_dir.glob(f"{experiment_name}*")):
            print(f"  - {file.name}")

        return result_code

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return 1


if __name__ == "__main__":
    print("Neural ODE Training Runner (Fixed)")
    print("Usage:")
    print("  python fixed_simple_runner.py           # Batched approach (default)")
    print("  python fixed_simple_runner.py --stride  # Strided approach")
    print("  python fixed_simple_runner.py --test    # Test on single protein")
    print("  python fixed_simple_runner.py cpu       # Force CPU")
    print("")
    print("📊 This version properly captures subprocess output")
    print("")

    sys.exit(main())