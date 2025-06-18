#!/usr/bin/env python3
"""
Simple script to print all information from a trained Neural ODE model file
Usage: python model_info.py <path_to_model.pt>
"""

import torch
import sys
import json
from pathlib import Path
from datetime import datetime


def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    else:
        return f"{seconds / 3600:.1f} hours"


def print_model_info(model_path):
    """Print comprehensive model information"""

    model_path = Path(model_path)

    # File info
    print("=" * 60)
    print("NEURAL ODE MODEL INFORMATION")
    print("=" * 60)
    print(f"📁 File: {model_path.name}")
    print(f"📍 Path: {model_path}")
    print(f"📊 Size: {format_size(model_path.stat().st_size)}")
    print(f"📅 Modified: {datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Checkpoint structure
    print("🔑 CHECKPOINT STRUCTURE")
    print("-" * 30)
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            num_params = len(checkpoint[key])
            print(f"  {key}: {num_params} parameters")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__}")
    print()

    # Configuration
    config = checkpoint.get('config', {})
    if config:
        print("⚙️  MODEL CONFIGURATION")
        print("-" * 30)

        # Core model settings
        print("🤖 Model Architecture:")
        print(f"  Model Type: {'Fast ODE' if config.get('use_fast_ode', False) else 'Full ODE'}")
        print(f"  Hidden Dimension: {config.get('hidden_dim', 'N/A')}")
        print(f"  Cluster Size: {config.get('reduced_cluster_size', 'N/A')}")
        print(f"  Integrator: {config.get('integrator', 'N/A')}")
        print(f"  Model Parameters: {config.get('model_parameters', 'N/A')}")

        print("\n🗂️  Data Settings:")
        data_dirs = config.get('data_dirs', [])
        if isinstance(data_dirs, list):
            for i, data_dir in enumerate(data_dirs):
                print(f"  Data Dir {i + 1}: {Path(data_dir).name}")
        else:
            print(f"  Data Dir: {data_dirs}")
        print(f"  Splits Dir: {Path(str(config.get('splits_dir', 'N/A'))).name}")
        print(f"  Max Residues: {config.get('max_residues', 'None')}")

        print("\n🎯 Training Settings:")
        print(f"  Epochs: {config.get('epochs', 'N/A')}")
        print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Device: {config.get('device', 'N/A')}")
        print(f"  Use AMP: {config.get('use_amp', False)}")
        print(f"  Aggressive Cleanup: {config.get('aggressive_cleanup', False)}")

        # Preliminary training settings
        prelim_enabled = config.get('enable_preliminary_training', False) or config.get('preliminary_training_enabled',
                                                                                        False)
        if prelim_enabled:
            print("\n🔄 Preliminary Training:")
            print(f"  Enabled: {prelim_enabled}")
            print(f"  Data Dir: {Path(str(config.get('prelim_data_dir', 'N/A'))).name}")
            print(f"  Block Stride: {config.get('prelim_block_stride', 'N/A')}")
            print(f"  Max Epochs: {config.get('prelim_max_epochs', 'N/A')}")
            print(f"  Chunk Size: {config.get('prelim_chunk_size', 'N/A')}")
        else:
            print("\n🔄 Preliminary Training: Disabled")

        # Optimization settings
        print("\n📈 Optimization:")
        print(f"  LR Patience: {config.get('lr_patience', 'N/A')}")
        print(f"  LR Factor: {config.get('lr_factor', 'N/A')}")
        print(f"  Min LR: {config.get('min_lr', 'N/A')}")
        print(f"  Early Stop Patience: {config.get('early_stopping_patience', 'N/A')}")
        print(f"  Early Stop Delta: {config.get('early_stopping_min_delta', 'N/A')}")
        print(f"  Max Time Hours: {config.get('max_time_hours', 'None')}")
        print()

    # Training statistics
    training_stats = checkpoint.get('training_stats', {})
    if training_stats:
        print("📊 TRAINING STATISTICS")
        print("-" * 30)

        print("🏁 Completion Status:")
        print(f"  Total Epochs: {training_stats.get('total_epochs', 'N/A')}")
        print(f"  Interrupted by Timeout: {training_stats.get('interrupted_by_timeout', False)}")
        print(f"  Method: {training_stats.get('method', 'N/A')}")

        total_time = training_stats.get('total_time_minutes')
        if total_time:
            print(f"  Total Time: {format_duration(total_time * 60)}")

        print("\n🎯 Performance:")
        best_val_loss = training_stats.get('best_val_loss')
        if best_val_loss:
            print(f"  Best Validation Loss: {best_val_loss:.6f}")
            print(f"  Best Validation Epoch: {training_stats.get('best_val_epoch', 'N/A')}")
        else:
            print(f"  Best Validation Loss: N/A")

        final_val_loss = training_stats.get('final_val_loss')
        if final_val_loss:
            print(f"  Final Validation Loss: {final_val_loss:.6f}")

        early_stopped = training_stats.get('early_stopped')
        if early_stopped is not None:
            print(f"  Early Stopped: {early_stopped}")

        print("\n📉 Learning Rate:")
        print(f"  LR Reductions: {training_stats.get('lr_reductions', 0)}")
        final_lr = training_stats.get('final_lr')
        if final_lr:
            print(f"  Final LR: {final_lr:.2e}")

        # Preliminary training stats
        prelim_enabled = training_stats.get('preliminary_training_enabled', False)
        if prelim_enabled:
            print(f"\n🔄 Preliminary Training: Enabled")
        print()

    # Model state dict analysis
    state_dict = checkpoint.get('model_state_dict', {})
    if state_dict:
        print("🧠 MODEL PARAMETERS")
        print("-" * 30)
        print(f"Total Parameters: {len(state_dict)}")

        # Calculate total parameter count
        total_params = 0
        for name, param in state_dict.items():
            total_params += param.numel()
        print(f"Total Parameter Count: {total_params:,}")

        # Parameter breakdown by component
        components = {}
        for name in state_dict.keys():
            if '.' in name:
                component = name.split('.')[0]
                if component not in components:
                    components[component] = 0
                components[component] += state_dict[name].numel()

        if components:
            print("\nParameter Distribution:")
            for component, count in sorted(components.items()):
                percentage = (count / total_params) * 100
                print(f"  {component}: {count:,} ({percentage:.1f}%)")

        # Show first few parameter shapes for architecture insight
        print("\nFirst 10 Parameter Shapes:")
        for i, (name, param) in enumerate(list(state_dict.items())[:10]):
            print(f"  {name}: {list(param.shape)}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more parameters")
        print()

    # Memory usage estimate
    if state_dict:
        param_memory = sum(param.numel() * 4 for param in state_dict.values())  # Assuming float32
        print("💾 MEMORY ESTIMATES")
        print("-" * 30)
        print(f"Model Parameters: {format_size(param_memory)}")
        print(f"Full Model (with gradients): {format_size(param_memory * 2)}")
        print()

    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python model_info.py <path_to_model.pt>")
        print()
        print("Examples:")
        print("  python model_info.py trained_models/20250615_180436_baseline_final_model.pt")
        print("  python model_info.py /path/to/model.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    if not Path(model_path).exists():
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)

    try:
        print_model_info(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print_model_info("../trained_models/20250617_180948_fast_ode_with_prelim_final_model.pt")
    print_model_info("../trained_models/20250616_180845_full_ode_with_prelim_final_model.pt")
# For running directly in script/IDE:
