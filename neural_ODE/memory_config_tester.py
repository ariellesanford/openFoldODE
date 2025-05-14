import subprocess
import torch
import gc
import time
import os
import sys
import argparse
from typing import Dict, List, Any


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_memory_configurations(python_path, script_path, data_dir, output_dir, test_protein=None):
    """
    Test different memory optimization configurations by calling the training script
    with different flag combinations.

    Args:
        python_path: Path to Python interpreter
        script_path: Path to training script
        data_dir: Data directory path
        output_dir: Output directory path
        test_protein: Specific protein ID to test (optional)

    Returns:
        List of results for each configuration tested
    """
    # List of configurations to try
    configs = [
        # Baseline with minimal optimizations - explicitly disable everything
        {
            "name": "Baseline (No optimizations)",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "rk4",
                "gradient_accumulation": 1,
                "chunk_size": 0,
            },
            "bool_flags": ["monitor_memory", "test-single-step"],
            "no_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration", "clean_memory"]
        },

        # Add checkpointing only
        {
            "name": "Checkpointing only",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "rk4",
                "gradient_accumulation": 1,
                "chunk_size": 0,
            },
            "bool_flags": ["use_checkpoint", "monitor_memory", "test-single-step"],
            "no_flags": ["use_amp", "reduced_precision_integration", "clean_memory"]
        },

        # Add mixed precision only
        {
            "name": "Mixed precision only",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "rk4",
                "gradient_accumulation": 1,
                "chunk_size": 0,
            },
            "bool_flags": ["use_amp", "monitor_memory", "test-single-step"],
            "no_flags": ["use_checkpoint", "reduced_precision_integration", "clean_memory"]
        },

        # Checkpointing + mixed precision
        {
            "name": "Checkpointing + AMP",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "rk4",
                "gradient_accumulation": 1,
                "chunk_size": 0,
            },
            "bool_flags": ["use_amp", "use_checkpoint", "monitor_memory", "test-single-step"],
            "no_flags": ["reduced_precision_integration", "clean_memory"]
        },

        # Add gradient accumulation
        {
            "name": "Checkpointing + AMP + Grad accumulation",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "rk4",
                "gradient_accumulation": 4,
                "chunk_size": 0,
            },
            "bool_flags": ["use_amp", "use_checkpoint", "monitor_memory", "test-single-step"],
            "no_flags": ["reduced_precision_integration", "clean_memory"]
        },

        # Add chunking
        {
            "name": "Previous + Chunking",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "rk4",
                "gradient_accumulation": 4,
                "chunk_size": 10,
            },
            "bool_flags": ["use_amp", "use_checkpoint", "monitor_memory", "test-single-step"],
            "no_flags": ["reduced_precision_integration", "clean_memory"]
        },

        # Add reduced precision integration with dopri5
        {
            "name": "Previous + Reduced precision dopri5",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 128,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "dopri5",
                "gradient_accumulation": 4,
                "chunk_size": 10,
            },
            "bool_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration",
                           "monitor_memory", "test-single-step"],
            "no_flags": ["clean_memory"]
        },

        # Full optimization
        {
            "name": "Full optimization (all enabled)",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 64,
                "reduced_hidden_dim": 64,
                "num_time_points": 25,
                "batch_size": 5,
                "integrator": "dopri5",
                "gradient_accumulation": 4,
                "chunk_size": 10,
            },
            "bool_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration",
                           "clean_memory", "monitor_memory", "test-single-step"],
            "no_flags": []
        },

        # Balanced optimization with original hidden_dim
        {
            "name": "Balanced optimization (original hidden_dim)",
            "flags": {
                "memory_split_size": 128,
                "reduced_cluster_size": 96,
                "reduced_hidden_dim": 128,
                "num_time_points": 35,
                "batch_size": 3,
                "integrator": "dopri5",
                "gradient_accumulation": 2,
                "chunk_size": 5,
            },
            "bool_flags": ["use_amp", "use_checkpoint", "reduced_precision_integration",
                           "clean_memory", "monitor_memory", "test-single-step"],
            "no_flags": []
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

        # Add boolean flags
        for flag in config["bool_flags"]:
            cmd.append(f"--{flag}")

        # IMPORTANT: Explicitly disable flags that should be off
        for flag in config["no_flags"]:
            cmd.append(f"--no-{flag}")

        # Add test protein to the command display
        if test_protein:
            print(f"\nRunning command (with test protein: {test_protein}):")
        else:
            print("\nRunning command:")
        print(" ".join(cmd))

        # Print what should be enabled/disabled
        print(f"\nEnabled flags: {', '.join(config['bool_flags'])}")
        print(f"Disabled flags: {', '.join(config['no_flags'])}")

        # Try to run the configuration
        try:
            # Clear memory before test
            clear_memory()

            start_time = time.time()

            # Run training script as subprocess and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

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

                for line in output_lines:
                    if "Loss:" in line and loss is None:  # Get first loss value
                        try:
                            # Extract loss from lines like "  - Loss: 5888.4352"
                            loss_str = line.split("Loss:")[-1].strip()
                            loss = float(loss_str)
                        except:
                            pass
                    if "Max Memory Allocated:" in line:
                        try:
                            mem_str = line.split(":")[-1].replace("MiB", "").strip()
                            max_memory = float(mem_str)
                        except:
                            pass

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
    print_configuration_summary(results)

    return results


def print_configuration_summary(results: List[Dict[str, Any]]):
    """Print a formatted summary of configuration test results"""
    print("\n" + "=" * 60)
    print("CONFIGURATION TEST SUMMARY")
    print("=" * 60)

    # Table header
    print(f"\n{'Name':<25} {'Status':<10} {'Memory (MB)':<12} {'Time (ms)':<12} {'Loss':<10}")
    print("-" * 80)

    # Results table
    for result in results:
        name = result['name'][:24]  # Truncate long names
        status = "SUCCESS" if result['success'] else f"FAILED ({result.get('error', 'Unknown')})"
        memory = f"{result.get('max_memory_mb', 0):.2f}" if 'max_memory_mb' in result else "N/A"
        time = f"{result.get('time_ms', 0):.2f}" if 'time_ms' in result else "N/A"
        loss = f"{result.get('loss', 0):.4f}" if 'loss' in result else "N/A"

        print(f"{name:<25} {status:<10} {memory:<12} {time:<12} {loss:<10}")

    # Find best configurations
    successful_configs = [r for r in results if r['success']]

    if successful_configs:
        print("\n" + "=" * 60)
        print("BEST CONFIGURATIONS")
        print("=" * 60)

        # Best for memory
        best_memory = min(successful_configs, key=lambda x: x['max_memory_mb'])
        print(f"\n🏆 Best Memory Efficiency:")
        print(f"   Name: {best_memory['name']}")
        print(f"   Memory: {best_memory['max_memory_mb']:.2f} MB")
        print(f"   Time: {best_memory['time_ms']:.2f} ms")
        print(f"   Loss: {best_memory['loss']:.4f}")

        # Best for speed
        best_speed = min(successful_configs, key=lambda x: x['time_ms'])
        print(f"\n🏆 Best Speed:")
        print(f"   Name: {best_speed['name']}")
        print(f"   Memory: {best_speed['max_memory_mb']:.2f} MB")
        print(f"   Time: {best_speed['time_ms']:.2f} ms")
        print(f"   Loss: {best_speed['loss']:.4f}")

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
        print(f"\n🏆 Best Balance (Memory + Speed):")
        print(f"   Name: {best_balance['name']}")
        print(f"   Memory: {best_balance['max_memory_mb']:.2f} MB")
        print(f"   Time: {best_balance['time_ms']:.2f} ms")
        print(f"   Loss: {best_balance['loss']:.4f}")
    else:
        print("\n⚠️  No successful configurations found!")

    print("\n" + "=" * 60)


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
        f.write(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

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
                        help='Specific protein ID to test')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("Memory Configuration Tester")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Python path: {args.python_path}")
    print(f"Script path: {args.script_path}")
    if args.test_protein:
        print(f"Testing protein: {args.test_protein}")

    # Run tests
    results = test_memory_configurations(
        python_path=args.python_path,
        script_path=args.script_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_protein=args.test_protein
    )

    # Generate report
    report_path = os.path.join(args.output_dir, "memory_optimization_report.txt")
    generate_config_report(results, report_path)


if __name__ == "__main__":
    main()