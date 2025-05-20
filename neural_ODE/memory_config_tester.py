import os
import argparse
import sys
import subprocess
import torch
import gc
import datetime
import time as time_module
from typing import Dict, List, Any


# Function to get project root directory
def get_project_root():
    """Get the path to the project root directory."""
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The current file should be in the neural_ODE directory, so current_dir is the project root
    return current_dir


def clear_memory():
    """Clear memory safely"""
    gc.collect()

    # Only attempt CUDA operations if CUDA is available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Reset peak memory stats, but handle errors gracefully
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as e:
                # Don't crash if reset_peak_memory_stats fails
                print(f"Warning: Could not reset CUDA peak memory stats: {e}")
                # This might happen if CUDA is technically available but not properly initialized
                # or if running in a WSL environment with limited CUDA support
        except Exception as e:
            print(f"Warning: CUDA error when clearing memory: {e}")
    else:
        print("Note: CUDA not available, skipping CUDA memory operations")


def test_memory_configurations(python_path, script_path, data_dir, output_dir, test_protein=None, use_fast_ode=False,
                               cpu_only=False):
    """
    Test different memory optimization configurations by calling the training script
    with different flag combinations.

    Args:
        python_path: Path to Python interpreter
        script_path: Path to training script
        data_dir: Data directory path
        output_dir: Output directory path
        test_protein: Specific protein ID to test (optional)
        use_fast_ode: Whether to use the fast ODE implementation
        cpu_only: Whether to force CPU-only mode
    """
    # If in CPU-only mode, use more CPU-friendly configurations
    if cpu_only:
        configs = [
            # 1. CPU Baseline - optimized for CPU
            {
                "name": "CPU Baseline",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 32,  # Smaller size for CPU
                    "reduced_hidden_dim": 32,  # Reduced dimensions for CPU
                    "num_time_points": 10,  # Fewer time points for CPU
                    "batch_size": 1,
                    "integrator": "euler",  # Simpler integrator for CPU
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["monitor_memory", "test-single-step", "cpu-only"],
                "no_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration", "clean_memory"]
            },
            # 2. CPU Fast - faster but less accurate
            {
                "name": "CPU Fast",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 16,  # Very small clusters
                    "reduced_hidden_dim": 16,  # Very small hidden dim
                    "num_time_points": 5,  # Minimal time points
                    "batch_size": 1,
                    "integrator": "euler",
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["use_fast_ode", "monitor_memory", "test-single-step", "cpu-only",
                               "reduced_precision_integration"],
                "no_flags": ["use_amp", "use_checkpoint", "clean_memory"]
            }
        ]
    else:
        # Original GPU configurations
        configs = [
            # 1. True Baseline - original values, no optimizations
            {
                "name": "True Baseline (Original values, no optimizations)",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 64,  # Reduced cluster size to avoid OOM
                    "reduced_hidden_dim": 64,  # Reduced hidden dim to avoid OOM
                    "num_time_points": 25,  # Fewer time points
                    "batch_size": 1,
                    "integrator": "rk4",
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["monitor_memory", "test-single-step"],
                "no_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration", "clean_memory", "cpu-only"]
            },

            # 2. AMP Only - only use Automatic Mixed Precision
            {
                "name": "AMP Only",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 64,  # Reduced cluster size
                    "reduced_hidden_dim": 64,  # Reduced hidden dim
                    "num_time_points": 25,  # Fewer time points
                    "batch_size": 1,
                    "integrator": "rk4",
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["use_amp", "monitor_memory", "test-single-step"],
                "no_flags": ["use_checkpoint", "reduced_precision_integration", "clean_memory", "cpu-only"]
            },

            # 3. Checkpoint Only - only use gradient checkpointing
            {
                "name": "Checkpoint Only",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 64,  # Reduced cluster size
                    "reduced_hidden_dim": 64,  # Reduced hidden dim
                    "num_time_points": 25,  # Fewer time points
                    "batch_size": 1,
                    "integrator": "rk4",
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["use_checkpoint", "monitor_memory", "test-single-step"],
                "no_flags": ["use_amp", "reduced_precision_integration", "clean_memory", "cpu-only"]
            },

            # 4. Memory optimized baseline - original values with key memory optimizations
            {
                "name": "Memory Optimized Baseline",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 96,  # Slightly reduced but not too much
                    "reduced_hidden_dim": 96,  # Slightly reduced but not too much
                    "num_time_points": 25,  # Fewer time points for speed
                    "batch_size": 1,
                    "integrator": "rk4",
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["use_amp", "use_checkpoint", "monitor_memory", "test-single-step"],
                "no_flags": ["reduced_precision_integration", "clean_memory", "cpu-only"]
            },

            # 5. Speed-Optimized Configuration
            {
                "name": "Speed-Optimized",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 64,  # Reduced cluster size
                    "reduced_hidden_dim": 64,  # Reduced hidden dim
                    "num_time_points": 15,  # Very few time points for speed
                    "batch_size": 1,
                    "integrator": "rk4",  # RK4 is faster for low precision
                    "gradient_accumulation": 1,
                    "chunk_size": 0,
                },
                "bool_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration", "monitor_memory",
                               "test-single-step", "use_fast_ode"],
                "no_flags": ["clean_memory", "cpu-only"]
            },

            # 6. Balance of Speed and Quality
            {
                "name": "Balanced Speed-Quality",
                "flags": {
                    "memory_split_size": 128,
                    "reduced_cluster_size": 80,  # Moderate cluster size
                    "reduced_hidden_dim": 80,  # Moderate hidden dim
                    "num_time_points": 20,  # Moderate time points
                    "batch_size": 1,
                    "integrator": "rk4",
                    "gradient_accumulation": 1,
                    "chunk_size": 5,  # Use chunking for memory efficiency
                },
                "bool_flags": ["use_amp", "use_checkpoint", "monitor_memory", "test-single-step"],
                "no_flags": ["reduced_precision_integration", "clean_memory", "cpu-only"]
            }
        ]

    results = []

    for i, config in enumerate(configs):
        print(f"\n{'=' * 50}")
        print(f"Testing Configuration {i + 1}/{len(configs)}: {config['name']}")
        print(f"{'=' * 50}")

        # Build command
        cmd = [
            python_path,
            script_path,
            "--data_dir", data_dir,
            "--output_dir", output_dir,
        ]

        # Add test protein if specified
        if test_protein:
            cmd.extend(["--test-protein", test_protein])

        # Add numeric flags
        for flag, value in config["flags"].items():
            cmd.extend([f"--{flag}", str(value)])

        # Add boolean flags for the train_evoformer_ode.py script
        training_bool_flags = []
        for flag in config["bool_flags"]:
            if flag == "use_fast_ode" and flag in cmd:
                # Skip if we've already added this flag to avoid duplication
                continue
            training_bool_flags.append(f"--{flag}")

        # If global use_fast_ode is enabled, add it to all configurations
        if use_fast_ode and "--use_fast_ode" not in training_bool_flags and "use_fast_ode" not in config["no_flags"]:
            training_bool_flags.append("--use_fast_ode")

        # If global cpu_only is enabled, add it to all configurations
        if cpu_only and "--cpu-only" not in training_bool_flags and "cpu-only" not in config["bool_flags"]:
            training_bool_flags.append("--cpu-only")

        # Add all bool flags to the command
        cmd.extend(training_bool_flags)

        # IMPORTANT: Explicitly disable flags that should be off
        for flag in config["no_flags"]:
            cmd.append(f"--no-{flag}")

        # Add test protein information to the command display
        if test_protein:
            if test_protein.lower() == "all":
                print(f"\nRunning command (testing ALL proteins):")
            else:
                print(f"\nRunning command (with test protein: {test_protein}):")
        else:
            print("\nRunning command (testing first protein only):")
        print(" ".join(cmd))

        # Print what should be enabled/disabled
        print(f"\nEnabled flags: {', '.join(config['bool_flags'])}")
        print(f"Disabled flags: {', '.join(config['no_flags'])}")

        # Try to run the configuration
        try:
            # Clear memory before test
            clear_memory()

            start_time = time_module.time()

            # Set timeout based on CPU/GPU mode
            timeout = 600 if cpu_only else 120  # 10 minutes for CPU, 2 minutes for GPU

            # Run training script as subprocess and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            elapsed_time = (time_module.time() - start_time) * 1000  # Convert to ms

            # Print the output to show the === ACTIVE MEMORY OPTIMIZATIONS ===
            print("\n--- Script output ---")
            print(result.stdout)
            if result.stderr:
                print("--- Script errors ---")
                print(result.stderr)
            print("--- End of output ---\n")

            if result.returncode == 0:
                # Parse output to extract loss and memory usage
                output_lines = result.stdout.split('\n')
                loss = None
                max_memory = None
                all_proteins_skipped = True  # Flag to check if all proteins were skipped

                # First look for average loss from all proteins
                for line in output_lines:
                    if "Epoch 1 - Average Loss:" in line:
                        try:
                            # Extract loss from lines like "Epoch 1 - Average Loss: 39.28158187866211"
                            loss_str = line.split("Average Loss:")[-1].strip()
                            loss = float(loss_str)
                            all_proteins_skipped = False  # At least one protein was processed
                            break  # Found the main loss value
                        except:
                            pass

                # If no average loss found, look for individual protein loss
                if loss is None:
                    for line in output_lines:
                        if "Loss:" in line and "Step" in line:
                            try:
                                # Extract loss from lines like "  - Loss: 5888.4352"
                                loss_str = line.split("Loss:")[-1].strip()
                                loss = float(loss_str)
                                all_proteins_skipped = False  # At least one protein was processed
                                break  # Use the first protein's loss
                            except:
                                pass

                # Look for max memory across all proteins
                for line in output_lines:
                    if "Maximum Memory Usage Across All Proteins:" in line:
                        # Look for max memory allocated in the following lines
                        for mem_line in output_lines[output_lines.index(line) + 1:output_lines.index(line) + 3]:
                            if "Max Memory Allocated:" in mem_line:
                                try:
                                    mem_str = mem_line.split(":")[-1].replace("MiB", "").strip()
                                    max_memory = float(mem_str)
                                    break
                                except:
                                    pass

                # If max memory not found from summary, check individual memory stats
                if max_memory is None:
                    for line in output_lines:
                        if "Max Memory Allocated:" in line:
                            try:
                                mem_str = line.split(":")[-1].replace("MiB", "").strip()
                                current_mem = float(mem_str)
                                if max_memory is None or current_mem > max_memory:
                                    max_memory = current_mem
                            except:
                                pass

                # Check if any protein was processed successfully
                for line in output_lines:
                    if "CUDA OOM for protein" not in line and "Step" in line and "Processing protein" in line:
                        all_proteins_skipped = False

                if all_proteins_skipped:
                    # All proteins were skipped due to OOM
                    error_msg = "All proteins were skipped due to OOM"
                    result_dict = {
                        "name": config['name'],
                        "config": config["flags"],
                        "bool_flags": config["bool_flags"],
                        "no_flags": config["no_flags"],
                        "success": False,
                        "error": "All OOM",
                        "max_memory_mb": max_memory if max_memory is not None else 0
                    }
                    results.append(result_dict)
                    print(f"\n✗ FAILED - All proteins skipped due to OOM")
                else:
                    # At least one protein was processed
                    if loss is None:
                        loss = 0.0
                    if max_memory is None:
                        max_memory = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else "N/A (CPU)"

                    result_dict = {
                        "name": config['name'],
                        "config": config["flags"],
                        "bool_flags": config["bool_flags"],
                        "no_flags": config["no_flags"],
                        "success": True,
                        "loss": loss,
                        "max_memory_mb": max_memory,
                        "time_ms": elapsed_time
                    }
                    results.append(result_dict)

                    print(f"\n✓ SUCCESS")
                    print(f"  Loss: {loss:.4f}")
                    if not isinstance(max_memory, str):
                        print(f"  Max Memory: {max_memory:.2f} MB")
                    else:
                        print(f"  Max Memory: {max_memory}")
                    print(f"  Time: {elapsed_time:.2f} ms")

            else:
                # Failed
                error_msg = result.stderr if result.stderr else "Unknown error"
                if torch.cuda.is_available() and "CUDA out of memory" in error_msg:
                    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                    result_dict = {
                        "name": config['name'],
                        "config": config["flags"],
                        "bool_flags": config["bool_flags"],
                        "no_flags": config["no_flags"],
                        "success": False,
                        "error": "OOM",
                        "max_memory_mb": max_memory
                    }
                    results.append(result_dict)
                    print(f"\n✗ FAILED - OOM at {max_memory:.2f} MB")
                else:
                    result_dict = {
                        "name": config['name'],
                        "config": config["flags"],
                        "bool_flags": config["bool_flags"],
                        "no_flags": config["no_flags"],
                        "success": False,
                        "error": error_msg[:200]  # Truncate long errors
                    }
                    results.append(result_dict)
                    print(f"\n✗ FAILED - Error: {error_msg[:200]}")

        except subprocess.TimeoutExpired:
            result_dict = {
                "name": config['name'],
                "config": config["flags"],
                "bool_flags": config["bool_flags"],
                "no_flags": config["no_flags"],
                "success": False,
                "error": "Timeout"
            }
            results.append(result_dict)
            print(f"\n✗ FAILED - Timeout")
        except Exception as e:
            result_dict = {
                "name": config['name'],
                "config": config["flags"],
                "bool_flags": config["bool_flags"],
                "no_flags": config["no_flags"],
                "success": False,
                "error": str(e)
            }
            results.append(result_dict)
            print(f"\n✗ FAILED - Error: {e}")

        clear_memory()

    # Print summary
    summary_text = print_configuration_summary(results, cpu_only)

    return results


def print_configuration_summary(results: List[Dict[str, Any]], cpu_only=False):
    """Print a formatted summary of configuration test results"""
    summary_text = "\n" + "=" * 60 + "\n"
    if cpu_only:
        summary_text += "CONFIGURATION TEST SUMMARY (CPU-ONLY MODE)\n"
    else:
        summary_text += "CONFIGURATION TEST SUMMARY\n"
    summary_text += "=" * 60 + "\n\n"

    # Table header
    header = f"{'Name':<25} {'Status':<10} {'Time (ms)':<12} {'Loss':<10}"
    if not cpu_only:
        header = f"{'Name':<25} {'Status':<10} {'Memory (MB)':<12} {'Time (ms)':<12} {'Loss':<10}"
    summary_text += header + "\n"
    summary_text += "-" * 80 + "\n"

    # Results table
    for result in results:
        name = result['name'][:24]  # Truncate long names
        status = "SUCCESS" if result['success'] else f"FAILED ({result.get('error', 'Unknown')})"
        memory = f"{result.get('max_memory_mb', 0):.2f}" if 'max_memory_mb' in result and not isinstance(
            result.get('max_memory_mb'), str) else "N/A"
        time = f"{result.get('time_ms', 0):.2f}" if 'time_ms' in result else "N/A"
        loss = f"{result.get('loss', 0):.4f}" if 'loss' in result else "N/A"

        if cpu_only:
            row = f"{name:<25} {status:<10} {time:<12} {loss:<10}"
        else:
            row = f"{name:<25} {status:<10} {memory:<12} {time:<12} {loss:<10}"
        summary_text += row + "\n"

    # Find best configurations
    successful_configs = [r for r in results if r['success']]

    if successful_configs:
        summary_text += "\n" + "=" * 60 + "\n"
        if cpu_only:
            summary_text += "BEST CPU CONFIGURATIONS\n"
        else:
            summary_text += "BEST CONFIGURATIONS\n"
        summary_text += "=" * 60 + "\n"

        # Skip memory-based recommendations in CPU mode
        if not cpu_only:
            # Best for memory
            best_memory = min(successful_configs,
                              key=lambda x: x['max_memory_mb'] if not isinstance(x['max_memory_mb'], str) else float(
                                  'inf'))
            summary_text += f"\n🏆 Best Memory Efficiency:\n"
            summary_text += f"   Name: {best_memory['name']}\n"
            if not isinstance(best_memory['max_memory_mb'], str):
                summary_text += f"   Memory: {best_memory['max_memory_mb']:.2f} MB\n"
            else:
                summary_text += f"   Memory: {best_memory['max_memory_mb']}\n"
            summary_text += f"   Time: {best_memory['time_ms']:.2f} ms\n"
            summary_text += f"   Loss: {best_memory['loss']:.4f}\n"

        # Best for speed
        best_speed = min(successful_configs, key=lambda x: x['time_ms'])
        summary_text += f"\n🏆 Best Speed:\n"
        summary_text += f"   Name: {best_speed['name']}\n"
        if not cpu_only and not isinstance(best_speed['max_memory_mb'], str):
            summary_text += f"   Memory: {best_speed['max_memory_mb']:.2f} MB\n"
        summary_text += f"   Time: {best_speed['time_ms']:.2f} ms\n"
        summary_text += f"   Loss: {best_speed['loss']:.4f}\n"

        # Best for loss
        best_loss = min(successful_configs, key=lambda x: x['loss'])
        summary_text += f"\n🏆 Best Loss Performance:\n"
        summary_text += f"   Name: {best_loss['name']}\n"
        if not cpu_only and not isinstance(best_loss['max_memory_mb'], str):
            summary_text += f"   Memory: {best_loss['max_memory_mb']:.2f} MB\n"
        summary_text += f"   Time: {best_loss['time_ms']:.2f} ms\n"
        summary_text += f"   Loss: {best_loss['loss']:.4f}\n"

        # Recommended balance - but skip complex balance calculations in CPU mode
        if not cpu_only:
            # Score = normalized memory + normalized time (lower is better)
            valid_memory_configs = [c for c in successful_configs if not isinstance(c['max_memory_mb'], str)]
            if valid_memory_configs:
                min_mem = min(c['max_memory_mb'] for c in valid_memory_configs)
                max_mem = max(c['max_memory_mb'] for c in valid_memory_configs)
                min_time = min(c['time_ms'] for c in valid_memory_configs)
                max_time = max(c['time_ms'] for c in valid_memory_configs)

                for config in valid_memory_configs:
                    norm_mem = (config['max_memory_mb'] - min_mem) / (max_mem - min_mem) if max_mem > min_mem else 0
                    norm_time = (config['time_ms'] - min_time) / (max_time - min_time) if max_time > min_time else 0
                    config['balance_score'] = norm_mem + norm_time

                best_balance = min(valid_memory_configs, key=lambda x: x['balance_score'])
                summary_text += f"\n🏆 Best Balance (Memory + Speed):\n"
                summary_text += f"   Name: {best_balance['name']}\n"
                summary_text += f"   Memory: {best_balance['max_memory_mb']:.2f} MB\n"
                summary_text += f"   Time: {best_balance['time_ms']:.2f} ms\n"
                summary_text += f"   Loss: {best_balance['loss']:.4f}\n"
    else:
        summary_text += "\n⚠️  No successful configurations found!\n"

    summary_text += "\n" + "=" * 60

    # Print the summary to console
    print(summary_text)

    # Return the text for file writing
    return summary_text


def generate_config_report(results: List[Dict[str, Any]], output_file: str = "memory_optimization_report.txt",
                           cpu_only=False):
    """Generate a detailed report of configuration test results"""
    with open(output_file, 'w') as f:
        if cpu_only:
            f.write("EVOFORMER ODE CPU-ONLY MODE OPTIMIZATION REPORT\n")
        else:
            f.write("EVOFORMER ODE MEMORY OPTIMIZATION REPORT\n")
        f.write("=" * 40 + "\n\n")

        # System information
        f.write("System Information:\n")
        f.write(f"  CUDA Available: {torch.cuda.is_available()}\n")
        if cpu_only:
            f.write(f"  Running in CPU-only mode (CUDA disabled)\n")
        elif torch.cuda.is_available():
            f.write(f"  GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  CUDA Version: {torch.version.cuda}\n")
        f.write(f"  PyTorch Version: {torch.__version__}\n")
        f.write("\n")

        # Get summary text (will also print to console)
        summary_text = print_configuration_summary(results, cpu_only)

        # Write the summary to the file
        f.write(summary_text)
        f.write("\n\n")

        # Detailed results
        f.write("Detailed Configuration Results:\n")
        f.write("-" * 40 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"Configuration {i}: {result['name']}\n")
            f.write(f"  Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")

            # Configuration details
            f.write("  Settings:\n")
            for key, value in result['config'].items():
                f.write(f"    {key}: {value}\n")
            f.write(f"    enabled flags: {', '.join(result.get('bool_flags', []))}\n")
            f.write(f"    disabled flags: {', '.join(result.get('no_flags', []))}\n")

            # Results
            if result['success']:
                f.write(f"  Results:\n")
                f.write(f"    Loss: {result['loss']:.6f}\n")
                if 'max_memory_mb' in result and not isinstance(result['max_memory_mb'], str):
                    f.write(f"    Max Memory: {result['max_memory_mb']:.2f} MB\n")
                else:
                    f.write(f"    Max Memory: CPU mode (no GPU metrics)\n")
                f.write(f"    Time: {result['time_ms']:.2f} ms\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown')}\n")
                if 'max_memory_mb' in result and not isinstance(result['max_memory_mb'], str):
                    f.write(f"    Max Memory at failure: {result['max_memory_mb']:.2f} MB\n")

            f.write("\n")

        f.write("=" * 40 + "\n")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Report generated at: {current_time}\n")

        # Add WSL-specific recommendations if no CUDA available
        if not torch.cuda.is_available() and not cpu_only:
            f.write("\nRecommendations for WSL users:\n")
            f.write("1. If you want to use CUDA with WSL, follow Microsoft's guide at:\n")
            f.write(
                "   https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-4---download-the-linux-kernel-update-package\n\n")
            f.write("2. Install NVIDIA CUDA drivers for WSL:\n")
            f.write("   https://developer.nvidia.com/cuda/wsl\n\n")
            f.write("3. After setting up CUDA for WSL, reinstall PyTorch with CUDA support:\n")
            f.write(
                "   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113\n\n")
            f.write(
                "4. Alternatively, you can continue using CPU-only mode by adding the --cpu-only flag to the run scripts\n\n")

    print(f"Report saved to: {output_file}")


def main():
    """Main function to run configuration testing"""
    parser = argparse.ArgumentParser(description='Test memory configurations for Evoformer ODE')

    # Get the project root directory
    project_root = get_project_root()

    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(project_root, "data"),
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(project_root, "config_test_outputs"),
                        help='Path to output directory')
    parser.add_argument('--python_path', type=str,
                        default=sys.executable,  # Use the current python interpreter
                        help='Path to Python interpreter')
    parser.add_argument('--script_path', type=str,
                        default=os.path.join(project_root, "train_evoformer_ode.py"),
                        help='Path to training script')
    parser.add_argument('--test-protein', type=str, default=None,
                        help='Specific protein ID to test (use "all" to test all proteins)')
    parser.add_argument('--use_fast_ode', action='store_true', default=False,
                        help='Use fast EvoformerODE implementation for all configs')
    parser.add_argument('--no-use_fast_ode', dest='use_fast_ode', action='store_false',
                        help='Use standard EvoformerODE implementation for all configs')
    parser.add_argument('--cpu-only', action='store_true', default=False,
                        help='Force CPU-only mode regardless of CUDA availability')

    args = parser.parse_args()

    # Initialize use_fast_ode from args
    use_fast_ode = args.use_fast_ode

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.cpu_only:
        print("Memory Configuration Tester (CPU-ONLY MODE)")
    else:
        print("Memory Configuration Tester")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Python path: {args.python_path}")
    print(f"Script path: {args.script_path}")

    # Check if USE_FAST_ODE environment variable is set
    use_fast_ode_env = os.environ.get('USE_FAST_ODE', '').lower() == 'true'
    if use_fast_ode_env:
        print("Using FAST ODE implementation (from environment variable)")
        # Override the command line argument
        use_fast_ode = True

    print(f"Using fast ODE implementation: {use_fast_ode}")
    if args.cpu_only:
        print("Force CPU-only mode: YES (CUDA will be disabled even if available)")
    if args.test_protein:
        print(f"Testing protein: {args.test_protein}")
    else:
        print("Testing: First protein in the dataset")

    # Run tests
    results = test_memory_configurations(
        python_path=args.python_path,
        script_path=args.script_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_protein=args.test_protein,
        use_fast_ode=use_fast_ode,
        cpu_only=args.cpu_only
    )

    # Generate report
    report_path = os.path.join(args.output_dir, "memory_optimization_report.txt")
    generate_config_report(results, report_path, args.cpu_only)


if __name__ == "__main__":
    main()