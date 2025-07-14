#!/usr/bin/env python3
"""
Automation script for running multiple preliminary training comparison experiments
Updated with custom configurations and comprehensive experiment matrix
"""

import os
import sys
import subprocess
import time
import torch
from datetime import datetime
from pathlib import Path


def get_valid_data_dirs():
    """Get valid data directories"""
    data_dirs = [
        Path("/media/visitor/Extreme SSD/data/complete_blocks"),
        Path("/media/visitor/Extreme SSD/data/endpoint_blocks"),
    ]

    valid_data_dirs = []
    for data_dir in data_dirs:
        if data_dir.exists():
            valid_data_dirs.append(str(data_dir))
            print(f"✅ Found data directory: {data_dir}")
        else:
            print(f"⚠️  Data directory not found: {data_dir}")

    if not valid_data_dirs:
        print(f"❌ No valid data directories found!")
        sys.exit(1)

    return valid_data_dirs


def create_experiment_configs():
    """Create experiment configurations for comprehensive preliminary training comparison"""

    # Get paths
    script_dir = Path(__file__).parent
    valid_data_dirs = get_valid_data_dirs()
    prelim_data_dir = Path("/media/visitor/Extreme SSD/data/complete_blocks")
    splits_dir = script_dir / "data_splits" / "jumbo"
    output_dir = script_dir / "trained_models"

    # Base configuration as specified
    base_config = {
        'data_dirs': valid_data_dirs,
        'splits_dir': str(splits_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 10000,
        'learning_rate': 1e-3,
        'reduced_cluster_size': 64,
        'hidden_dim': 64,
        'integrator': 'rk4',
        'use_fast_ode': False,
        'max_residues': 350,
        'loss': 'default',  # Will be overridden per experiment
        'use_amp': torch.cuda.is_available(),
        'output_dir': str(output_dir),
        # Enhanced features
        'lr_patience': 3,
        'lr_factor': 0.5,
        'min_lr': 1e-6,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'max_time_hours': 24,
        # Memory optimizations
        'aggressive_cleanup': True,
        # Preliminary training settings
        'enable_preliminary_training': True,
        'prelim_data_dir': str(prelim_data_dir),
        'prelim_max_epochs': 20,
    }

    # Create experiments as specified
    experiments = []

    # 1. prelim_block_stride 8 with chunk size=1,2,3 and 8 (20 epochs)
    for chunk_size in [1, 2, 3, 8]:
        experiments.append((
            f"Stride8_Chunk{chunk_size}_Epochs20", {
            **base_config,
            'prelim_block_stride': 8,
            'prelim_chunk_size': chunk_size,
            'prelim_max_epochs': 20,
            'loss': 'default',
            'experiment_name': f"stride8_chunk{chunk_size}_epochs20"
        }
        ))

    # 2. prelim_block_stride=12 with chunk size 1,2,4
    for chunk_size in [1, 2, 4]:
        experiments.append((
            f"Stride12_Chunk{chunk_size}_Epochs20", {
            **base_config,
            'prelim_block_stride': 12,
            'prelim_chunk_size': chunk_size,
            'prelim_max_epochs': 20,
            'loss': 'default',
            'experiment_name': f"stride12_chunk{chunk_size}_epochs20"
        }
        ))

    # 3. prelim_block_stride=24 with chunk size= 1,2
    for chunk_size in [1, 2]:
        experiments.append((
            f"Stride24_Chunk{chunk_size}_Epochs20", {
            **base_config,
            'prelim_block_stride': 24,
            'prelim_chunk_size': chunk_size,
            'prelim_max_epochs': 20,
            'loss': 'default',
            'experiment_name': f"stride24_chunk{chunk_size}_epochs20"
        }
        ))

    # 4. prelim_block_stride 8 with chunk_size=3 and loss = 'weighted_row'
    experiments.append((
        "Stride8_Chunk3_WeightedRow", {
        **base_config,
        'prelim_block_stride': 8,
        'prelim_chunk_size': 3,
        'prelim_max_epochs': 20,
        'loss': 'weighted_row',
        'experiment_name': "stride8_chunk3_weighted_row"
    }
    ))

    # 5. prelim_block_stride 8 with loss = 'single_row'
    experiments.append((
        "Stride8_Chunk3_SingleRow", {
        **base_config,
        'prelim_block_stride': 8,
        'prelim_chunk_size': 3,
        'prelim_max_epochs': 20,
        'loss': 'single_row',
        'experiment_name': "stride8_chunk3_single_row"
    }
    ))

    # 6. prelim_block_stride 8 with chunk size=1,2,3 and 8 again (40 epochs)
    for chunk_size in [1, 2, 3, 8]:
        experiments.append((
            f"Stride8_Chunk{chunk_size}_Epochs40", {
            **base_config,
            'prelim_block_stride': 8,
            'prelim_chunk_size': chunk_size,
            'prelim_max_epochs': 40,
            'loss': 'default',
            'experiment_name': f"stride8_chunk{chunk_size}_epochs40"
        }
        ))

    # 7. prelim_block_stride 8 with chunk size=1,2, 16 and with loss = 'weighted_row' and 'single_row'
    for chunk_size in [1, 2, 16]:
        for loss_func in ['weighted_row', 'single_row']:
            experiments.append((
                f"Stride8_Chunk{chunk_size}_{loss_func.title()}", {
                **base_config,
                'prelim_block_stride': 8,
                'prelim_chunk_size': chunk_size,
                'prelim_max_epochs': 20,
                'loss': loss_func,
                'experiment_name': f"stride8_chunk{chunk_size}_{loss_func}"
            }
            ))
    return experiments


def run_training_experiment(name, config):
    """Run a single training experiment with fresh timestamp"""
    print(f"\n🔬 Starting: {name}")
    print(f"📁 Output: {config['output_dir']}")
    print(f"🧬 Loss: {config['loss']}")
    print(f"📏 Max residues: {config['max_residues']}")
    print(f"🔄 Prelim chunk size: {config['prelim_chunk_size']}")
    print(f"📊 Prelim block stride: {config['prelim_block_stride']}")
    print(f"🎯 Prelim max epochs: {config['prelim_max_epochs']}")

    # Generate FRESH timestamp right before training starts
    training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Update the experiment name with fresh timestamp
    base_name = config['experiment_name']
    config['experiment_name'] = f"{training_timestamp}_{base_name}"

    print(f"🏷️  Experiment name: {config['experiment_name']}")

    os.makedirs(config['output_dir'], exist_ok=True)

    # Build command with all parameters
    cmd_args = [
                   sys.executable, 'train_evoformer_ode.py',
                   '--data_dirs'] + config['data_dirs'] + [
                   '--splits_dir', config['splits_dir'],
                   '--device', config['device'],
                   '--epochs', str(config['epochs']),
                   '--learning_rate', str(config['learning_rate']),
                   '--reduced_cluster_size', str(config['reduced_cluster_size']),
                   '--hidden_dim', str(config['hidden_dim']),
                   '--integrator', config['integrator'],
                   '--loss', config['loss'],
                   '--max_residues', str(config['max_residues']),
                   '--output_dir', config['output_dir'],
                   '--experiment_name', config['experiment_name'],  # Now has fresh timestamp
                   '--lr_patience', str(config['lr_patience']),
                   '--lr_factor', str(config['lr_factor']),
                   '--min_lr', str(config['min_lr']),
                   '--early_stopping_patience', str(config['early_stopping_patience']),
                   '--early_stopping_min_delta', str(config['early_stopping_min_delta']),
                   '--max_time_hours', str(config['max_time_hours']),
                   '--prelim_data_dir', config['prelim_data_dir'],
                   '--prelim_block_stride', str(config['prelim_block_stride']),
                   '--prelim_max_epochs', str(config['prelim_max_epochs']),
                   '--prelim_chunk_size', str(config['prelim_chunk_size']),
               ]

    # Add boolean flags
    if config['use_amp']:
        cmd_args.append('--use_amp')
    if config['aggressive_cleanup']:
        cmd_args.append('--aggressive_cleanup')
    if config['enable_preliminary_training']:
        cmd_args.append('--enable_preliminary_training')
    if config['use_fast_ode']:
        cmd_args.append('--use_fast_ode')

    start_time = time.time()

    try:
        # Run the training with real-time output streaming
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Stream output to console in real-time
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print to console immediately
            output_lines.append(line)

        # Wait for process to complete
        result_code = process.wait()

        end_time = time.time()
        duration = end_time - start_time

        # Save logs with updated experiment name
        log_dir = os.path.join(config['output_dir'], 'automation_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{config['experiment_name']}_automation.txt")

        with open(log_file, 'w') as f:
            f.write(f"Experiment: {name}\n")
            f.write(f"Actual experiment name: {config['experiment_name']}\n")
            f.write(f"Started: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ended: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration / 3600:.2f} hours\n")
            f.write(f"Return code: {result_code}\n")
            f.write(f"Command: {' '.join(cmd_args)}\n\n")
            f.write("CONFIGURATION:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nOUTPUT:\n")
            f.write(''.join(output_lines))

        if result_code == 0:
            print(f"✅ {name} completed successfully in {duration / 3600:.2f} hours")
            return True
        else:
            print(f"❌ {name} failed with return code {result_code}")
            print(f"Check logs at: {log_file}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {name} timed out")
        return False
    except Exception as e:
        print(f"💥 {name} crashed: {e}")
        return False


def generate_comparison_report(experiments, results):
    """Generate a comparison report for all experiments"""

    script_dir = Path(__file__).parent
    report_file = script_dir / "trained_models" / f"preliminary_training_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(report_file, 'w') as f:
        f.write("# Preliminary Training Comparison Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiment Overview\n\n")
        f.write("This automated comparison tested different preliminary training strategies:\n")
        f.write("- Block stride variations (8, 12, 24)\n")
        f.write("- Chunk size variations (1, 2, 3, 4, 8)\n")
        f.write("- Epoch variations (20, 40)\n")
        f.write("- Loss function variations (default, weighted_row, single_row)\n\n")

        f.write("## Experiment Results\n\n")
        for i, ((name, config), success) in enumerate(zip(experiments, results), 1):
            status = "✅ SUCCESS" if success else "❌ FAILED"
            f.write(f"**{i}. {name}** - {status}\n")
            f.write(f"   - Loss function: `{config['loss']}`\n")
            f.write(f"   - Block stride: {config['prelim_block_stride']}\n")
            f.write(f"   - Chunk size: {config['prelim_chunk_size']}\n")
            f.write(f"   - Max epochs: {config['prelim_max_epochs']}\n")
            f.write(f"   - Experiment name: `{config['experiment_name']}`\n\n")

        f.write("## Experiment Configurations\n\n")
        f.write("### Base Configuration\n")
        if experiments:
            base_config = experiments[0][1]
            for key, value in base_config.items():
                if key not in ['loss', 'experiment_name', 'prelim_chunk_size', 'prelim_block_stride',
                               'prelim_max_epochs']:
                    f.write(f"- **{key}**: {value}\n")

        f.write("\n### Experiment-Specific Variations\n")
        f.write("| Experiment | Block Stride | Chunk Size | Max Epochs | Loss Function |\n")
        f.write("|------------|--------------|------------|------------|---------------|\n")
        for name, config in experiments:
            f.write(
                f"| {name} | {config['prelim_block_stride']} | {config['prelim_chunk_size']} | {config['prelim_max_epochs']} | {config['loss']} |\n")

        f.write("\n## Analysis Guidelines\n\n")
        f.write("To compare results, examine:\n\n")
        f.write("1. **Preliminary Training Convergence**: Compare how quickly different configurations converge\n")
        f.write("2. **Final Loss Values**: Check both preliminary and main training final losses\n")
        f.write("3. **Training Speed**: Compare epochs/hour for different chunk sizes\n")
        f.write("4. **Memory Usage**: Monitor memory efficiency across configurations\n")
        f.write("5. **Block Stride Impact**: Compare performance across different stride values\n")
        f.write("6. **Chunk Size Impact**: Analyze how chunk size affects learning and memory\n")
        f.write("7. **Loss Function Impact**: Compare default vs weighted_row vs single_row\n\n")

        f.write("## Files to Check\n\n")
        for name, config in experiments:
            f.write(f"### {name}\n")
            f.write(f"- Training log: `{config['output_dir']}/{config['experiment_name']}.txt`\n")
            f.write(f"- Model checkpoint: `{config['output_dir']}/{config['experiment_name']}_final_model.pt`\n")
            f.write(
                f"- Automation log: `{config['output_dir']}/automation_logs/{config['experiment_name']}_automation.txt`\n\n")

    print(f"\n📊 Comparison report generated: {report_file}")


def main():
    """Main automation function"""

    print("🤖 Preliminary Training Comparison Automation")
    print("=" * 65)
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create experiment configurations
    experiments = create_experiment_configs()

    print(f"\n📋 Configured {len(experiments)} experiments:")
    for i, (name, config) in enumerate(experiments, 1):
        print(f"  {i:2d}. {name}")
        print(f"      Stride: {config['prelim_block_stride']}, Chunk: {config['prelim_chunk_size']}, "
              f"Epochs: {config['prelim_max_epochs']}, Loss: '{config['loss']}'")

    # Confirm before starting
    response = input(f"\n🚀 Start all {len(experiments)} experiments? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Aborted by user")
        return

    # Run experiments
    results = []
    total_start = time.time()

    for i, (name, config) in enumerate(experiments):
        print(f"\n📈 Running experiment {i + 1}/{len(experiments)}")
        print(f"=" * 50)
        success = run_training_experiment(name, config)
        results.append(success)

        if not success:
            response = input(f"\n⚠️  {name} failed. Continue with remaining experiments? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Stopping automation due to failure")
                break

        # Brief pause between experiments
        if i < len(experiments) - 1:
            print(f"\n⏸️  Brief pause before next experiment...")
            time.sleep(5)

    total_time = time.time() - total_start

    # Generate report
    generate_comparison_report(experiments, results)

    # Final summary
    successful = sum(results)
    print(f"\n🏁 AUTOMATION COMPLETE")
    print(f"=" * 50)
    print(f"⏱️  Total time: {total_time / 3600:.2f} hours")
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {len(experiments) - successful}/{len(experiments)}")

    if successful > 0:
        print(f"\n📊 Check comparison report and individual experiment logs for results!")
        print(f"📁 Results location: trained_models/")

    return successful == len(experiments)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)