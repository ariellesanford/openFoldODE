#!/usr/bin/env python3
"""
Automation script for running multiple Neural ODE training experiments
Updated for restructured train_evoformer_ode.py
Runs different configurations sequentially with proper naming and logging
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import torch


def get_experiment_configs():
    """Define all experiment configurations to run"""

    # Base paths - modify these as needed
    script_dir = Path(__file__).parent
    data_dirs = [
        "/media/visitor/Extreme SSD/data/complete_blocks",
        "/media/visitor/Extreme SSD/data/endpoint_blocks",
    ]
    splits_dir = script_dir / "data_splits" / "jumbo"
    output_dir = script_dir / "trained_models"
    prelim_data_dir = "/media/visitor/Extreme SSD/data/complete_blocks"

    # Note: timestamp will be generated when each experiment starts
    configs = [
        {
            'name': 'baseline_no_prelim',
            'description': 'Baseline: Fast ODE, no preliminary training',
            'config': {
                'data_dirs': data_dirs,
                'splits_dir': str(splits_dir),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': 1,
                'learning_rate': 1e-3,
                'reduced_cluster_size': 64,
                'hidden_dim': 64,
                'integrator': 'rk4',
                'use_fast_ode': False,
                'use_amp': torch.cuda.is_available(),
                'output_dir': str(output_dir),
                'experiment_name': None,  # Will be set when experiment starts
                #'max_residues': 50,
                'lr_patience': 3,
                'lr_factor': 0.5,
                'min_lr': 1e-6,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.0001,
                'max_time_hours': 24,
                'aggressive_cleanup': True,
                'enable_preliminary_training': True,
                'prelim_data_dir': str(prelim_data_dir),
                'prelim_block_stride': 8,
                'prelim_max_epochs': 1,
                'prelim_chunk_size': 3,
            }
        },
        {
            'name': 'fast_ode_with_prelim',
            'description': 'Fast ODE with preliminary training',
            'config': {
                'data_dirs': data_dirs,
                'splits_dir': str(splits_dir),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': 1000,
                'learning_rate': 1e-3,
                'reduced_cluster_size': 128,
                'hidden_dim': 64,
                'integrator': 'rk4',
                'use_fast_ode': True,
                'use_amp': torch.cuda.is_available(),
                'output_dir': str(output_dir),
                'experiment_name': None,  # Will be set when experiment starts
                'lr_patience': 3,
                'lr_factor': 0.5,
                'min_lr': 1e-6,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.0001,
                'max_time_hours': 24,
                'aggressive_cleanup': True,
                'enable_preliminary_training': True,
                'prelim_data_dir': str(prelim_data_dir),
                'prelim_block_stride': 8,
                'prelim_max_epochs': 20,
                'prelim_chunk_size': 4,
            }
        },
        {
            'name': 'full_ode_with_prelim_medium',
            'description': 'Full ODE (not fast) with preliminary training - medium size',
            'config': {
                'data_dirs': data_dirs,
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
                'experiment_name': None,  # Will be set when experiment starts
                'lr_patience': 3,
                'lr_factor': 0.5,
                'min_lr': 1e-6,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.0001,
                'max_time_hours': 24,
                'aggressive_cleanup': True,
                'enable_preliminary_training': True,
                'prelim_data_dir': str(prelim_data_dir),
                'prelim_block_stride': 8,
                'prelim_max_epochs': 20,
                'prelim_chunk_size': 4,
            }
        },
        {
            'name': 'full_ode_with_prelim_small',
            'description': 'Full ODE (not fast) with preliminary training - small size',
            'config': {
                'data_dirs': data_dirs,
                'splits_dir': str(splits_dir),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': 1000,
                'learning_rate': 1e-3,
                'reduced_cluster_size': 32,
                'hidden_dim': 32,
                'integrator': 'rk4',
                'use_fast_ode': False,
                'use_amp': torch.cuda.is_available(),
                'output_dir': str(output_dir),
                'experiment_name': None,  # Will be set when experiment starts
                'lr_patience': 3,
                'lr_factor': 0.5,
                'min_lr': 1e-6,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.0001,
                'max_time_hours': 24,
                'aggressive_cleanup': True,
                'enable_preliminary_training': True,
                'prelim_data_dir': str(prelim_data_dir),
                'prelim_block_stride': 8,
                'prelim_max_epochs': 20,
                'prelim_chunk_size': 4,
            }
        },
        {
            'name': 'fast_ode_different_stride',
            'description': 'Fast ODE with preliminary training - different stride',
            'config': {
                'data_dirs': data_dirs,
                'splits_dir': str(splits_dir),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'epochs': 1000,
                'learning_rate': 1e-3,
                'reduced_cluster_size': 64,
                'hidden_dim': 64,
                'integrator': 'rk4',
                'use_fast_ode': True,
                'use_amp': torch.cuda.is_available(),
                'output_dir': str(output_dir),
                'experiment_name': None,  # Will be set when experiment starts
                'lr_patience': 3,
                'lr_factor': 0.5,
                'min_lr': 1e-6,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 0.0001,
                'max_time_hours': 24,
                'aggressive_cleanup': True,
                'enable_preliminary_training': True,
                'prelim_data_dir': str(prelim_data_dir),
                'prelim_block_stride': 12,  # Different stride
                'prelim_max_epochs': 15,
                'prelim_chunk_size': 6,  # Different chunk size
            }
        }
    ]

    return configs


def run_experiment(experiment_config, script_path):
    """Run a single experiment configuration"""

    name = experiment_config['name']
    description = experiment_config['description']
    config = experiment_config['config'].copy()  # Make a copy to avoid modifying original

    print(f"\n🚀 Starting experiment: {name}")
    print(f"📝 Description: {description}")
    print("=" * 80)

    # Generate timestamp for this specific experiment
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['experiment_name'] = f"{experiment_timestamp}_{name}"

    print(f"🏷️  Experiment name: {config['experiment_name']}")

    # Build command
    cmd = [sys.executable, str(script_path)]

    for key, value in config.items():
        if key == 'data_dirs':
            cmd.extend(['--data_dirs'] + value)
        elif isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"🔧 Command: {' '.join(cmd[:5])}... (truncated)")
    print(f"📊 Key settings:")
    print(f"   - Model: {'Fast ODE' if config.get('use_fast_ode') else 'Full ODE'}")
    print(f"   - Hidden dim: {config.get('hidden_dim')}")
    print(f"   - Cluster size: {config.get('reduced_cluster_size')}")
    print(f"   - Preliminary training: {config.get('enable_preliminary_training', False)}")
    if config.get('enable_preliminary_training'):
        print(f"     * Block stride: {config.get('prelim_block_stride')}")
        print(f"     * Chunk size: {config.get('prelim_chunk_size')}")
        print(f"     * Max epochs: {config.get('prelim_max_epochs')}")
    print(f"   - Max time: {config.get('max_time_hours')} hours")
    print(f"   - Timestamp: {experiment_timestamp}")
    print("")

    start_time = time.time()

    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        return_code = process.wait()

        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600

        print(f"\n{'=' * 80}")
        if return_code == 0:
            print(f"✅ Experiment '{name}' completed successfully!")
        else:
            print(f"❌ Experiment '{name}' failed (return code: {return_code})")

        print(f"⏱️  Duration: {duration_hours:.2f} hours")
        print(f"📁 Results: {config['output_dir']}/{config['experiment_name']}")

        return return_code == 0, duration_hours

    except KeyboardInterrupt:
        print(f"\n⏹️  Experiment '{name}' interrupted by user")
        return False, 0
    except Exception as e:
        print(f"\n❌ Error running experiment '{name}': {e}")
        return False, 0


def main():
    """Main automation script"""

    script_dir = Path(__file__).parent
    training_script = script_dir / "train_evoformer_ode.py"

    if not training_script.exists():
        print(f"❌ Training script not found: {training_script}")
        return 1

    # Get all experiment configurations
    experiments = get_experiment_configs()

    print("🤖 Neural ODE Training Automation - Updated for Restructured Script")
    print("=" * 80)
    print(f"📁 Training script: {training_script}")
    print(f"🧪 Total experiments: {len(experiments)}")
    print("")

    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}: {exp['description']}")

    print("")
    response = input("Do you want to run all experiments? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return 0

    # Run all experiments
    results = []
    total_start_time = time.time()

    for i, experiment in enumerate(experiments, 1):
        print(f"\n🔄 Running experiment {i}/{len(experiments)}")

        success, duration = run_experiment(experiment, training_script)
        results.append({
            'name': experiment['name'],
            'description': experiment['description'],
            'success': success,
            'duration_hours': duration
        })

        # Short break between experiments
        if i < len(experiments):
            print(f"\n⏸️  5-second break before next experiment...")
            time.sleep(5)

    # Final summary
    total_duration = (time.time() - total_start_time) / 3600
    successful = sum(1 for r in results if r['success'])

    print(f"\n🎯 AUTOMATION COMPLETE")
    print("=" * 80)
    print(f"⏱️  Total time: {total_duration:.2f} hours")
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {len(experiments) - successful}/{len(experiments)}")
    print("")

    print("📊 Experiment Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['name']} ({result['duration_hours']:.2f}h): {result['description']}")

    if successful > 0:
        print(f"\n📁 Check results in: trained_models/")
        print(f"🔍 Each experiment has its own timestamp:")
        for result in results:
            if result['success']:
                print(f"     ✅ {result['name']} - Look for files with its unique timestamp")

    print(f"\n🔧 Changes made for restructured script:")
    print(f"   - Removed: restore_best_weights, use_sequential_loading")
    print(f"   - Removed: prelim_early_stopping_min_delta, prelim_rtol, prelim_atol")
    print(f"   - Kept configurable: prelim_chunk_size, prelim_block_stride")
    print(f"   - Updated argument structure for new TrainingConfig")
    print(f"   - Each experiment gets its own timestamp when it starts")

    return 0 if successful == len(experiments) else 1


if __name__ == "__main__":
    sys.exit(main())