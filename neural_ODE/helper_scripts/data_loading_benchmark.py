#!/usr/bin/env python3
"""
Benchmark different data loading strategies to determine optimal approach
Tests: per-protein vs per-timestep vs hybrid approaches
"""

import os
import time
import torch
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import gc
from contextlib import contextmanager


class DataLoadingBenchmark:
    """
    Comprehensive benchmark of different data loading strategies
    """

    def __init__(self, data_dir: str, temp_dir: str, training_script: str):
        self.data_dir = Path(data_dir)
        self.temp_dir = Path(temp_dir)
        self.training_script = Path(training_script)

        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Discover available proteins and timesteps
        self.proteins = self._discover_proteins()
        self.max_timesteps = self._discover_max_timesteps()

        print(f"Benchmark initialized:")
        print(f"  Data directory: {data_dir}")
        print(f"  Proteins found: {len(self.proteins)} - {self.proteins}")
        print(f"  Max timesteps: {self.max_timesteps}")
        print(f"  Temp directory: {temp_dir}")

    def _discover_proteins(self) -> List[str]:
        """Find all available proteins"""
        proteins = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.endswith('_evoformer_blocks'):
                protein_id = item.name.replace('_evoformer_blocks', '')
                proteins.append(protein_id)
        return sorted(proteins)

    def _discover_max_timesteps(self) -> int:
        """Find maximum number of timesteps available"""
        max_timesteps = 0
        for protein in self.proteins:
            protein_dir = self.data_dir / f"{protein}_evoformer_blocks" / "recycle_0"
            if protein_dir.exists():
                m_files = list(protein_dir.glob("m_block_*.pt"))
                max_timesteps = max(max_timesteps, len(m_files))
        return max_timesteps

    @contextmanager
    def timer(self, description: str):
        """Context manager for timing operations"""
        print(f"⏱️  Starting: {description}")
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            print(f"✅ Completed: {description}")
            print(f"   Time: {duration:.2f}s")
            print(f"   Memory change: {memory_delta:.1f}MB")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 2
        else:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 ** 2

    def _clear_temp_data(self):
        """Clear all temporary data"""
        if self.temp_dir.exists():
            for item in self.temp_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    def _copy_protein_data(self, protein_id: str, timesteps: List[int] = None):
        """Copy specific protein data to temp directory"""
        source_dir = self.data_dir / f"{protein_id}_evoformer_blocks"
        target_dir = self.temp_dir / f"{protein_id}_evoformer_blocks"

        if not source_dir.exists():
            raise FileNotFoundError(f"Source data not found: {source_dir}")

        # Create target structure
        target_recycle_dir = target_dir / "recycle_0"
        target_recycle_dir.mkdir(parents=True, exist_ok=True)

        # Copy specific timesteps or all
        source_recycle_dir = source_dir / "recycle_0"

        if timesteps is None:
            # Copy all files
            for file in source_recycle_dir.glob("*.pt"):
                shutil.copy2(file, target_recycle_dir / file.name)
        else:
            # Copy only specific timesteps
            for timestep in timesteps:
                m_file = f"m_block_{timestep}.pt"
                z_file = f"z_block_{timestep}.pt"

                for filename in [m_file, z_file]:
                    source_file = source_recycle_dir / filename
                    if source_file.exists():
                        shutil.copy2(source_file, target_recycle_dir / filename)

        # Copy mask files (needed for training)
        for mask_file in ["msa_mask.pt", "pair_mask.pt"]:
            source_mask = source_recycle_dir / mask_file
            if source_mask.exists():
                shutil.copy2(source_mask, target_recycle_dir / mask_file)

    def _run_training_test(self, test_name: str, test_proteins: List[str] = None,
                           epochs: int = 1, extra_args: List[str] = None) -> Dict:
        """Run a training test and measure performance"""

        if test_proteins is None:
            test_proteins = self.proteins

        # Prepare command
        cmd = [
            "python", str(self.training_script),
            "--data_dir", str(self.temp_dir),
            "--output_dir", str(self.temp_dir / "training_output"),
            "--epochs", str(epochs),
            "--test-single-step",  # Only one step for benchmarking
            "--monitor_memory",
            "--reduced_cluster_size", "32",  # Small for faster testing
            "--reduced_hidden_dim", "32",
            "--num_time_points", "5",  # Minimal for speed
            "--learning_rate", "1e-4"
        ]

        # Add specific proteins
        if len(test_proteins) == 1:
            cmd.extend(["--test-protein", test_proteins[0]])
        elif len(test_proteins) > 1:
            cmd.extend(["--test-protein", "all"])

        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)

        print(f"🚀 Running training test: {test_name}")
        print(f"   Proteins: {test_proteins}")
        print(f"   Command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()

            # Parse results from output
            success = result.returncode == 0
            total_time = end_time - start_time

            # Extract loss and memory from output if successful
            loss = None
            max_memory = None

            if success:
                for line in result.stdout.split('\n'):
                    if "Loss:" in line and "Step" not in line:
                        try:
                            loss = float(line.split("Loss:")[-1].strip())
                            break
                        except:
                            pass
                    elif "Max Memory Allocated:" in line:
                        try:
                            memory_str = line.split(":")[-1].replace("MiB", "").strip()
                            max_memory = float(memory_str)
                        except:
                            pass

            return {
                "test_name": test_name,
                "success": success,
                "total_time": total_time,
                "loss": loss,
                "max_memory_mb": max_memory,
                "num_proteins": len(test_proteins),
                "proteins": test_proteins,
                "stdout": result.stdout if success else None,
                "stderr": result.stderr if not success else None
            }

        except subprocess.TimeoutExpired:
            return {
                "test_name": test_name,
                "success": False,
                "total_time": 300.0,
                "error": "Timeout",
                "num_proteins": len(test_proteins),
                "proteins": test_proteins
            }
        except Exception as e:
            return {
                "test_name": test_name,
                "success": False,
                "error": str(e),
                "num_proteins": len(test_proteins),
                "proteins": test_proteins
            }

    def test_strategy_1_per_protein_sequential(self) -> List[Dict]:
        """Test Strategy 1: Load one protein at a time, train separately"""
        print(f"\n{'=' * 60}")
        print(f"STRATEGY 1: Per-Protein Sequential Loading")
        print(f"{'=' * 60}")

        results = []
        total_overhead_time = 0

        for i, protein_id in enumerate(self.proteins):
            with self.timer(f"Strategy 1 - Protein {i + 1}/{len(self.proteins)}: {protein_id}"):
                # Clear temp data
                self._clear_temp_data()

                # Copy this protein's data
                copy_start = time.time()
                self._copy_protein_data(protein_id)
                copy_time = time.time() - copy_start

                # Run training
                result = self._run_training_test(
                    test_name=f"Strategy1_Protein_{protein_id}",
                    test_proteins=[protein_id]
                )

                result["data_copy_time"] = copy_time
                results.append(result)

                total_overhead_time += copy_time

        # Summary
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            total_training_time = sum(r["total_time"] for r in successful_results)
            avg_time_per_protein = total_training_time / len(successful_results)

            print(f"\n📊 Strategy 1 Results:")
            print(f"   Successful proteins: {len(successful_results)}/{len(results)}")
            print(f"   Total training time: {total_training_time:.2f}s")
            print(f"   Total data copy overhead: {total_overhead_time:.2f}s")
            print(f"   Average time per protein: {avg_time_per_protein:.2f}s")
            print(f"   Overhead percentage: {(total_overhead_time / total_training_time) * 100:.1f}%")

        return results

    def test_strategy_2_all_proteins_at_once(self) -> Dict:
        """Test Strategy 2: Load all proteins at once, train together"""
        print(f"\n{'=' * 60}")
        print(f"STRATEGY 2: All Proteins At Once")
        print(f"{'=' * 60}")

        with self.timer("Strategy 2 - Loading all proteins"):
            # Clear temp data
            self._clear_temp_data()

            # Copy all protein data
            copy_start = time.time()
            for protein_id in self.proteins:
                self._copy_protein_data(protein_id)
            copy_time = time.time() - copy_start

            # Run training on all proteins
            result = self._run_training_test(
                test_name="Strategy2_AllProteins",
                test_proteins=self.proteins
            )

            result["data_copy_time"] = copy_time

            if result["success"]:
                print(f"\n📊 Strategy 2 Results:")
                print(f"   Total time: {result['total_time']:.2f}s")
                print(f"   Data copy overhead: {copy_time:.2f}s")
                print(f"   Overhead percentage: {(copy_time / result['total_time']) * 100:.1f}%")
                print(f"   Time per protein: {result['total_time'] / len(self.proteins):.2f}s")

        return result

    def test_strategy_3_per_timestep_batch(self, timesteps_to_test: List[int] = None) -> List[Dict]:
        """Test Strategy 3: Load one timestep for all proteins, train, repeat"""
        print(f"\n{'=' * 60}")
        print(f"STRATEGY 3: Per-Timestep Batch Loading")
        print(f"{'=' * 60}")

        if timesteps_to_test is None:
            timesteps_to_test = list(range(min(5, self.max_timesteps)))  # Test first 5 timesteps

        results = []
        total_overhead_time = 0

        for timestep in timesteps_to_test:
            with self.timer(f"Strategy 3 - Timestep {timestep}"):
                # Clear temp data
                self._clear_temp_data()

                # Copy this timestep for all proteins
                copy_start = time.time()
                for protein_id in self.proteins:
                    try:
                        self._copy_protein_data(protein_id, timesteps=[timestep])
                    except Exception as e:
                        print(f"Warning: Could not copy timestep {timestep} for {protein_id}: {e}")
                copy_time = time.time() - copy_start

                # Run training
                result = self._run_training_test(
                    test_name=f"Strategy3_Timestep_{timestep}",
                    test_proteins=self.proteins
                )

                result["timestep"] = timestep
                result["data_copy_time"] = copy_time
                results.append(result)

                total_overhead_time += copy_time

        # Summary
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            total_training_time = sum(r["total_time"] for r in successful_results)
            avg_time_per_timestep = total_training_time / len(successful_results)

            print(f"\n📊 Strategy 3 Results:")
            print(f"   Successful timesteps: {len(successful_results)}/{len(results)}")
            print(f"   Total training time: {total_training_time:.2f}s")
            print(f"   Total data copy overhead: {total_overhead_time:.2f}s")
            print(f"   Average time per timestep: {avg_time_per_timestep:.2f}s")
            print(f"   Overhead percentage: {(total_overhead_time / total_training_time) * 100:.1f}%")

        return results

    def run_comprehensive_benchmark(self, output_file: str = None) -> Dict:
        """Run all benchmark strategies and compare results"""
        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE DATA LOADING STRATEGY BENCHMARK")
        print(f"{'=' * 80}")

        benchmark_results = {
            "benchmark_info": {
                "proteins": self.proteins,
                "max_timesteps": self.max_timesteps,
                "data_dir": str(self.data_dir),
                "temp_dir": str(self.temp_dir)
            },
            "strategy_1_per_protein": [],
            "strategy_2_all_at_once": {},
            "strategy_3_per_timestep": []
        }

        try:
            # Test Strategy 1: Per-protein sequential
            benchmark_results["strategy_1_per_protein"] = self.test_strategy_1_per_protein_sequential()

            # Test Strategy 2: All proteins at once
            benchmark_results["strategy_2_all_at_once"] = self.test_strategy_2_all_proteins_at_once()

            # Test Strategy 3: Per-timestep batch
            benchmark_results["strategy_3_per_timestep"] = self.test_strategy_3_per_timestep_batch()

        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            benchmark_results["error"] = str(e)

        # Clean up
        self._clear_temp_data()

        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

        # Print summary
        self._print_benchmark_summary(benchmark_results)

        return benchmark_results

    def _print_benchmark_summary(self, results: Dict):
        """Print a comprehensive summary of benchmark results"""
        print(f"\n{'=' * 80}")
        print(f"BENCHMARK SUMMARY & RECOMMENDATIONS")
        print(f"{'=' * 80}")

        # Strategy 1 summary
        strategy1_results = results["strategy_1_per_protein"]
        strategy1_successful = [r for r in strategy1_results if r["success"]]

        if strategy1_successful:
            strategy1_total_time = sum(r["total_time"] for r in strategy1_successful)
            strategy1_overhead = sum(r.get("data_copy_time", 0) for r in strategy1_successful)
            print(f"\n📊 STRATEGY 1 (Per-Protein Sequential):")
            print(f"   Success rate: {len(strategy1_successful)}/{len(strategy1_results)}")
            print(f"   Total time: {strategy1_total_time:.2f}s")
            print(
                f"   Data loading overhead: {strategy1_overhead:.2f}s ({(strategy1_overhead / strategy1_total_time) * 100:.1f}%)")
            print(f"   Average per protein: {strategy1_total_time / len(strategy1_successful):.2f}s")

        # Strategy 2 summary
        strategy2_result = results["strategy_2_all_at_once"]
        if strategy2_result.get("success"):
            strategy2_time = strategy2_result["total_time"]
            strategy2_overhead = strategy2_result.get("data_copy_time", 0)
            print(f"\n📊 STRATEGY 2 (All Proteins At Once):")
            print(f"   Success: Yes")
            print(f"   Total time: {strategy2_time:.2f}s")
            print(
                f"   Data loading overhead: {strategy2_overhead:.2f}s ({(strategy2_overhead / strategy2_time) * 100:.1f}%)")
            print(f"   Time per protein: {strategy2_time / len(self.proteins):.2f}s")

        # Strategy 3 summary
        strategy3_results = results["strategy_3_per_timestep"]
        strategy3_successful = [r for r in strategy3_results if r["success"]]

        if strategy3_successful:
            strategy3_total_time = sum(r["total_time"] for r in strategy3_successful)
            strategy3_overhead = sum(r.get("data_copy_time", 0) for r in strategy3_successful)
            print(f"\n📊 STRATEGY 3 (Per-Timestep Batch):")
            print(f"   Success rate: {len(strategy3_successful)}/{len(strategy3_results)}")
            print(f"   Total time: {strategy3_total_time:.2f}s")
            print(
                f"   Data loading overhead: {strategy3_overhead:.2f}s ({(strategy3_overhead / strategy3_total_time) * 100:.1f}%)")
            print(f"   Average per timestep: {strategy3_total_time / len(strategy3_successful):.2f}s")

        # Recommendation
        print(f"\n🎯 RECOMMENDATION:")

        # Compare efficiency (time per protein)
        recommendations = []

        if strategy1_successful:
            strategy1_per_protein = strategy1_total_time / len(strategy1_successful)
            recommendations.append(
                ("Strategy 1 (Per-Protein)", strategy1_per_protein, strategy1_overhead / strategy1_total_time))

        if strategy2_result.get("success"):
            strategy2_per_protein = strategy2_time / len(self.proteins)
            recommendations.append(
                ("Strategy 2 (All At Once)", strategy2_per_protein, strategy2_overhead / strategy2_time))

        if strategy3_successful:
            # For strategy 3, estimate time per protein across all timesteps
            avg_timestep_time = strategy3_total_time / len(strategy3_successful)
            estimated_total_time = avg_timestep_time * self.max_timesteps
            strategy3_per_protein = estimated_total_time / len(self.proteins)
            recommendations.append(
                ("Strategy 3 (Per-Timestep)", strategy3_per_protein, strategy3_overhead / strategy3_total_time))

        if recommendations:
            best_strategy = min(recommendations, key=lambda x: x[1])
            print(f"   🏆 WINNER: {best_strategy[0]}")
            print(f"   ⚡ Time per protein: {best_strategy[1]:.2f}s")
            print(f"   📁 Overhead ratio: {best_strategy[2] * 100:.1f}%")

            print(f"\n📋 Full comparison:")
            for name, time_per_protein, overhead_ratio in sorted(recommendations, key=lambda x: x[1]):
                print(f"   {name}: {time_per_protein:.2f}s per protein, {overhead_ratio * 100:.1f}% overhead")


def main():
    parser = argparse.ArgumentParser(description="Benchmark data loading strategies")
    parser.add_argument("--data_dir", required=True, help="Directory containing protein data")
    parser.add_argument("--temp_dir", default="./benchmark_temp", help="Temporary directory for testing")
    parser.add_argument("--training_script", default="./train_evoformer_ode.py", help="Path to training script")
    parser.add_argument("--output", default="./benchmark_results.json", help="Output file for results")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer timesteps")

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = DataLoadingBenchmark(
        data_dir=args.data_dir,
        temp_dir=args.temp_dir,
        training_script=args.training_script
    )

    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(output_file=args.output)

    print(f"\n✅ Benchmark complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()