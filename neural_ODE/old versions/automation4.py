#!/usr/bin/env python3
"""
Automated Loss Function Comparison Script
Runs training with all three loss strategies: default, weighted_row, single_row
"""

import os
import sys
import torch
import subprocess
import time
from pathlib import Path
from datetime import datetime


def get_base_config():
    """Base configuration shared across all experiments"""

    # Use same paths as training_runner.py
    script_dir = Path(__file__).parent
    base_data_dir = Path("/media/visitor/Extreme SSD/data")

    valid_data_dirs = []
    for subdir in ["complete_blocks", "endpoint_blocks"]:
        data_path = base_data_dir / subdir
        if data_path.exists():
            valid_data_dirs.append(str(data_path))

    if not valid_data_dirs:
        raise FileNotFoundError(f"No valid data directories found in {base_data_dir}")

    prelim_data_dir = base_data_dir / "complete_blocks"
    splits_dir = script_dir / "data_splits" / "jumbo"  # Same as training_runner
    output_base_dir = script_dir / "trained_models"  # Same as training_runner

    # Create directories
    splits_dir.mkdir(exist_ok=True)
    output_base_dir.mkdir(exist_ok=True)

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
        'max_residues': 300,
        'use_amp': torch.cuda.is_available(),

        # Enhanced features
        'lr_patience': 3,
        'lr_factor': 0.5,
        'min_lr': 1e-6,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0001,
        'max_time_hours': 6,

        # Memory optimizations
        'aggressive_cleanup': True,

        # Preliminary training settings
        'enable_preliminary_training': True,
        'prelim_data_dir': str(prelim_data_dir),
        'prelim_block_stride': 4,
        'prelim_max_epochs': 10,
        'prelim_chunk_size': 3,
    }

    return base_config, output_base_dir


def create_experiment_configs():
    """Create configurations for each loss function experiment"""

    base_config, output_base_dir = get_base_config()

    experiments = []

    # Experiment 1: Default loss (full MSA)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_default = base_config.copy()
    config_default.update({
        'loss': 'default',
        'experiment_name': f'{timestamp}_loss_comparison_default_full_msa',
        'output_dir': str(output_base_dir),
    })
    experiments.append(('Default Loss (Full MSA)', config_default))

    # Experiment 2: Weighted row loss
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_weighted = base_config.copy()
    config_weighted.update({
        'loss': 'weighted_row',
        'experiment_name': f'{timestamp}_loss_comparison_weighted_row_5x_first',
        'output_dir': str(output_base_dir),
    })
    experiments.append(('Weighted Row Loss (5x First Row)', config_weighted))

    # Experiment 3: Single row loss (structure-focused)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_single = base_config.copy()
    config_single.update({
        'loss': 'single_row',
        'experiment_name': f'{timestamp}_loss_comparison_single_row_structure_focused',
        'output_dir': str(output_base_dir),
    })
    experiments.append(('Single Row Loss (Structure-Focused)', config_single))

    return experiments


def run_training_experiment(name, config):
    """Run a single training experiment"""

    print(f"\n{'=' * 80}")
    print(f"🚀 STARTING EXPERIMENT: {name}")
    print(f"📁 Output directory: {config['output_dir']}")
    print(f"🎯 Loss strategy: {config['loss']}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Build command
    cmd = [
        sys.executable, 'train_evoformer_ode.py',  # Adjust script name if needed
        '--config', 'from_dict'  # Special flag to pass config as dict
    ]

    # Alternative: if your script takes individual arguments, build them:
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
                   '--output_dir', config['output_dir'],
                   '--experiment_name', config['experiment_name'],
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
        # Run the training with real-time output streaming (like training_runner)
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

        # Save logs
        log_dir = os.path.join(config['output_dir'], 'automation_logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{config['experiment_name']}_automation.txt")

        with open(log_file, 'w') as f:
            f.write(f"Experiment: {name}\n")
            f.write(f"Started: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ended: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration / 3600:.2f} hours\n")
            f.write(f"Return code: {result_code}\n")
            f.write(f"Command: {' '.join(cmd_args)}\n\n")
            f.write("OUTPUT:\n")
            f.write(''.join(output_lines))

        if result_code == 0:
            print(f"✅ {name} completed successfully in {duration / 3600:.2f} hours")
            return True
        else:
            print(f"❌ {name} failed with return code {result_code}")
            print(f"Check logs at: {log_file}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {name} timed out after {config['max_time_hours']} hours")
        return False
    except Exception as e:
        print(f"💥 {name} crashed with error: {e}")
        return False


def generate_comparison_report(experiments, results):
    """Generate a summary report of all experiments"""

    script_dir = Path(__file__).parent
    report_file = script_dir / "trained_models" / "loss_comparison_report.md"

    with open(report_file, 'w') as f:
        f.write("# Loss Function Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiments Overview\n\n")
        f.write("| Experiment | Loss Strategy | Status | Output Directory |\n")
        f.write("|------------|---------------|--------|-----------------|\n")

        for i, (name, config) in enumerate(experiments):
            status = "✅ Success" if results[i] else "❌ Failed"
            f.write(f"| {name} | `{config['loss']}` | {status} | `{config['output_dir']}` |\n")

        f.write("\n## Loss Strategy Details\n\n")
        f.write("### Default Loss\n")
        f.write("- Uses full MSA representation: `F.mse_loss(pred_m, target_m)`\n")
        f.write("- All MSA rows weighted equally\n")
        f.write("- Most computationally expensive\n\n")

        f.write("### Weighted Row Loss\n")
        f.write("- First MSA row weighted 5x more than others\n")
        f.write("- Normalized weighting (weights sum to 1.0)\n")
        f.write("- Balances structure importance with full MSA training\n\n")

        f.write("### Single Row Loss\n")
        f.write("- Only uses first MSA row: `F.mse_loss(pred_m[0, :, :], target_m[0, :, :])`\n")
        f.write("- Most structure-focused approach\n")
        f.write("- Fastest training and most memory efficient\n\n")

        f.write("## Results Analysis\n\n")
        f.write("Compare the following metrics across experiments:\n\n")
        f.write("1. **Training Loss**: Check final training losses in each experiment's log\n")
        f.write("2. **Validation Loss**: Compare validation performance\n")
        f.write("3. **Structure Quality**: Run structure prediction tests\n")
        f.write("4. **Training Speed**: Compare epochs/hour and memory usage\n")
        f.write("5. **Convergence**: Check how quickly each approach reaches good performance\n\n")

        f.write("## Files to Check\n\n")
        for name, config in experiments:
            f.write(f"### {name}\n")
            f.write(f"- Training log: `{config['output_dir']}/{config['experiment_name']}.txt`\n")
            f.write(f"- Model checkpoint: `{config['output_dir']}/{config['experiment_name']}_best.pt`\n")
            f.write(
                f"- Automation log: `{config['output_dir']}/automation_logs/{config['experiment_name']}_automation.txt`\n\n")

    print(f"\n📊 Comparison report generated: {report_file}")


def main():
    """Main automation function"""

    print("🤖 Loss Function Comparison Automation")
    print("======================================")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create experiment configurations
    experiments = create_experiment_configs()

    print(f"\n📋 Configured {len(experiments)} experiments:")
    for i, (name, config) in enumerate(experiments, 1):
        print(f"  {i}. {name} (loss='{config['loss']}')")

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
        success = run_training_experiment(name, config)
        results.append(success)

        if not success:
            response = input(f"\n⚠️  {name} failed. Continue with remaining experiments? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Stopping automation due to failure")
                break

    total_time = time.time() - total_start

    # Generate report
    generate_comparison_report(experiments, results)

    # Final summary
    successful = sum(results)
    print(f"\n🏁 AUTOMATION COMPLETE")
    print(f"⏱️  Total time: {total_time / 3600:.2f} hours")
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {len(experiments) - successful}/{len(experiments)}")

    if successful > 0:
        print(f"\n📊 Check comparison report and individual experiment logs for results!")

    return successful == len(experiments)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)