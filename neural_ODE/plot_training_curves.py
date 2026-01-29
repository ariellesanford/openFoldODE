#!/usr/bin/env python3
"""Parse training log .txt files and plot loss curves."""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(filepath):
    """Extract epoch/loss data from training log."""
    text = Path(filepath).read_text()
    
    data = {'prelim': [], 'main': []}
    
    # Find preliminary training table (header row then dash separator then data then dash separator)
    prelim_match = re.search(
        r'Preliminary Training Progress:\n-+\nEpoch.*?\n-+\n(.*?)\n-+',
        text, re.DOTALL
    )
    if prelim_match:
        for line in prelim_match.group(1).strip().split('\n'):
            parts = line.split()
            if parts and parts[0].isdigit():
                data['prelim'].append({
                    'epoch': int(parts[0]),
                    'train_loss': float(parts[1]),
                    'val_loss': float(parts[3])
                })
    
    # Find main training table
    main_match = re.search(
        r'Main Training Progress:\n-+\nEpoch.*?\n-+\n(.*?)\n-+',
        text, re.DOTALL
    )
    if main_match:
        for line in main_match.group(1).strip().split('\n'):
            parts = line.split()
            if parts and parts[0].isdigit():
                data['main'].append({
                    'epoch': int(parts[0]),
                    'train_loss': float(parts[1]),
                    'val_loss': float(parts[3])
                })
    
    return data


def plot_curves(data, output_path=None, title=None):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Preliminary training
    if data['prelim']:
        ax = axes[0]
        epochs = [d['epoch'] for d in data['prelim']]
        ax.plot(epochs, [d['train_loss'] for d in data['prelim']], 'b-o', label='Train', markersize=4)
        ax.plot(epochs, [d['val_loss'] for d in data['prelim']], 'r-o', label='Val', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Preliminary Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Main training
    if data['main']:
        ax = axes[1]
        epochs = [d['epoch'] for d in data['main']]
        ax.plot(epochs, [d['train_loss'] for d in data['main']], 'b-', label='Train', linewidth=1.5)
        ax.plot(epochs, [d['val_loss'] for d in data['main']], 'r-', label='Val', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Main Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(15, 60)
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', nargs='?', help='Training log .txt file (optional - will list available if not provided)')
    parser.add_argument('-o', '--output', help='Output image path (default: same name as input with .png)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    trained_models_dir = script_dir / 'trained_models'
    
    if not args.logfile:
        # List available log files
        log_files = sorted(trained_models_dir.glob('*.txt'))
        if not log_files:
            print(f"No .txt files found in {trained_models_dir}")
            exit(1)
        print("Available log files:")
        for i, f in enumerate(log_files, 1):
            print(f"  {i}. {f.name}")
        choice = input("\nEnter number or filename: ").strip()
        if choice.isdigit():
            args.logfile = str(log_files[int(choice) - 1])
        else:
            args.logfile = str(trained_models_dir / choice)
    elif not Path(args.logfile).exists():
        # Try relative to trained_models
        args.logfile = str(trained_models_dir / args.logfile)
    
    data = parse_log_file(args.logfile)
    print(f"Parsed {len(data['prelim'])} preliminary epochs, {len(data['main'])} main epochs")
    
    output = args.output or str(Path(args.logfile).with_suffix('.png'))
    title = Path(args.logfile).stem
    
    plot_curves(data, output, title)

    #python plot_training_curves.py 20250616_180845_full_ode_with_prelim.txt
