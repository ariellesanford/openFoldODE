import subprocess
import torch
import gc
import datetime
import time as time_module
import os
import sys
import argparse
from typing import Dict, List, Any


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_memory_configurations(python_path, script_path, data_dir, output_dir, test_protein=None, use_fast_ode=False):
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

    Returns:
        List of results for each configuration tested
    """
    # List of configurations to try
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
            "no_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration", "clean_memory"]
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
            "no_flags": ["use_checkpoint", "reduced_precision_integration", "clean_memory"]
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
            "no_flags": ["use_amp", "reduced_precision_integration", "clean_memory"]
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
            "no_flags": ["reduced_precision_integration", "clean_memory"]
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
            "no_flags": ["clean_memory"]
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
            "no_flags": ["reduced_precision_integration", "clean_memory"]
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

            # Run training script as subprocess and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # Reduced timeout to 2 minutes

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

                for line in output_lines:
                    if "Loss:" in line and "Step" in line and loss is None:  # Get first loss value
                        try:
                            # Extract loss from lines like "  - Loss: 5888.4352"
                            loss_str = line.split("Loss:")[-1].strip()
                            loss = float(loss_str)
                            all_proteins_skipped = False  # At least one protein was processed
                        except:
                            pass
                    if "Max Memory Allocated:" in line:
                        try:
                            mem_str = line.split(":")[-1].replace("MiB", "").strip()
                            max_memory = float(mem_str)
                        except:
                            pass
                    # Check if any protein was processed successfully
                    if "CUDA OOM for protein" not in line and "Step" in line and "Processing protein" in line:
                        all_proteins_skipped = False

                # Check for max memory info for all proteins
                for line in output_lines:
                    if "Maximum Memory Usage Across All Proteins:" in line:
                        # Extract the max memory across all proteins
                        for mem_line in output_lines[output_lines.index(line) + 1:output_lines.index(line) + 3]:
                            if "Max Memory Allocated:" in mem_line:
                                try:
                                    mem_str = mem_line.split(":")[-1].replace("MiB", "").strip()
                                    max_memory = float(mem_str)
                                    break
                                except:
                                    pass

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
                        max_memory = torch.cuda.max_memory_allocated() / 1024 ** 2

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
                    print(f"  Max Memory: {max_memory:.2f} MB")
                    print(f"  Time: {elapsed_time:.2f} ms")

            else:
                # Failed
                error_msg = result.stderr if result.stderr else "Unknown error"
                if "CUDA out of memory" in error_msg:
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
    summary_text = print_configuration_summary(results)

    return results


def print_configuration_summary(results: List[Dict[str, Any]]):
    """Print a formatted summary of configuration test results"""
    summary_text = "\n" + "=" * 60 + "\n"
    summary_text += "CONFIGURATION TEST SUMMARY\n"
    summary_text += "=" * 60 + "\n\n"

    # Table header
    summary_text += f"{'Name':<25} {'Status':<10} {'Memory (MB)':<12} {'Time (ms)':<12} {'Loss':<10}\n"
    summary_text += "-" * 80 + "\n"

    # Results table
    for result in results:
        name = result['name'][:24]  # Truncate long names
        status = "SUCCESS" if result['success'] else f"FAILED ({result.get('error', 'Unknown')})"
        memory = f"{result.get('max_memory_mb', 0):.2f}" if 'max_memory_mb' in result else "N/A"
        time = f"{result.get('time_ms', 0):.2f}" if 'time_ms' in result else "N/A"
        loss = f"{result.get('loss', 0):.4f}" if 'loss' in result else "N/A"

        summary_text += f"{name:<25} {status:<10} {memory:<12} {time:<12} {loss:<10}\n"

    # Find best configurations
    successful_configs = [r for r in results if r['success']]

    if successful_configs:
        summary_text += "\n" + "=" * 60 + "\n"
        summary_text += "BEST CONFIGURATIONS\n"
        summary_text += "=" * 60 + "\n"

        # Best for memory
        best_memory = min(successful_configs, key=lambda x: x['max_memory_mb'])
        summary_text += f"\n🏆 Best Memory Efficiency:\n"
        summary_text += f"   Name: {best_memory['name']}\n"
        summary_text += f"   Memory: {best_memory['max_memory_mb']:.2f} MB\n"
        summary_text += f"   Time: {best_memory['time_ms']:.2f} ms\n"
        summary_text += f"   Loss: {best_memory['loss']:.4f}\n"

        # Best for speed
        best_speed = min(successful_configs, key=lambda x: x['time_ms'])
        summary_text += f"\n🏆 Best Speed:\n"
        summary_text += f"   Name: {best_speed['name']}\n"
        summary_text += f"   Memory: {best_speed['max_memory_mb']:.2f} MB\n"
        summary_text += f"   Time: {best_speed['time_ms']:.2f} ms\n"
        summary_text += f"   Loss: {best_speed['loss']:.4f}\n"

        # Best for loss
        best_loss = min(successful_configs, key=lambda x: x['loss'])
        summary_text += f"\n🏆 Best Loss Performance:\n"
        summary_text += f"   Name: {best_loss['name']}\n"
        summary_text += f"   Memory: {best_loss['max_memory_mb']:.2f} MB\n"
        summary_text += f"   Time: {best_loss['time_ms']:.2f} ms\n"
        summary_text += f"   Loss: {best_loss['loss']:.4f}\n"

        # Recommended balance
        # Score = normalized memory + normalized time (lower is better)
        min_mem = min(c['max_memory_mb'] for c in successful_configs)
        max_mem = max(c['max_memory_mb'] for c in successful_configs)
        min_time = min(c['time_ms'] for c in successful_configs)
        max_time = max(c['time_ms'] for c in successful_configs)

        for config in successful_configs:
            norm_mem = (config['max_memory_mb'] - min_mem) / (max_mem - min_mem) if max_mem > min_mem else 0
            norm_time = (config['time_ms'] - min_time) / (max_time - min_time) if max_time > min_time else 0
            config['balance_score'] = norm_mem + norm_time

        best_balance = min(successful_configs, key=lambda x: x['balance_score'])
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


def generate_config_report(results: List[Dict[str, Any]], output_file: str = "memory_optimization_report.txt"):
    """Generate a detailed report of configuration test results"""
    with open(output_file, 'w') as f:
        f.write("EVOFORMER ODE MEMORY OPTIMIZATION REPORT\n")
        f.write("=" * 40 + "\n\n")

        # System information
        f.write("System Information:\n")
        f.write(f"  CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"  GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  CUDA Version: {torch.version.cuda}\n")
            f.write(f"  PyTorch Version: {torch.__version__}\n")
        f.write("\n")

        # Get summary text (will also print to console)
        summary_text = print_configuration_summary(results)

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
                f.write(f"    Max Memory: {result['max_memory_mb']:.2f} MB\n")
                f.write(f"    Time: {result['time_ms']:.2f} ms\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown')}\n")
                if 'max_memory_mb' in result:
                    f.write(f"    Max Memory at failure: {result['max_memory_mb']:.2f} MB\n")

            f.write("\n")

        f.write("=" * 40 + "\n")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Report generated at: {current_time}\n")

    print(f"Report saved to: {output_file}")


def main():
    """Main function to run configuration testing"""
    parser = argparse.ArgumentParser(description='Test memory configurations for Evoformer ODE')
    parser.add_argument('--data_dir', type=str,
                        default="/home/visitor/PycharmProjects/openFold/neural_ODE/data",
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                        default="/home/visitor/PycharmProjects/openFold/neural_ODE/config_test_outputs",
                        help='Path to output directory')
    parser.add_argument('--python_path', type=str,
                        default="/home/visitor/anaconda3/envs/openfold_env/bin/python",
                        help='Path to Python interpreter')
    parser.add_argument('--script_path', type=str,
                        default="train_evoformer_ode.py",
                        help='Path to training script')
    parser.add_argument('--test-protein', type=str, default=None,
                        help='Specific protein ID to test (use "all" to test all proteins)')
    parser.add_argument('--use_fast_ode', action='store_true', default=False,
                        help='Use fast EvoformerODE implementation for all configs')
    parser.add_argument('--no-use_fast_ode', dest='use_fast_ode', action='store_false',
                        help='Use standard EvoformerODE implementation for all configs')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

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
    if args.test_protein:
        print(f"Testing protein: {args.test_protein}")

    # Run tests
    results = test_memory_configurations(
        python_path=args.python_path,
        script_path=args.script_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_protein=args.test_protein,
        use_fast_ode=use_fast_ode
    )

    # Generate report
    report_path = os.path.join(args.output_dir, "memory_optimization_report.txt")
    generate_config_report(results, report_path)


if __name__ == "__main__":
    main()