#!/usr/bin/env python3
"""
Automated Training Suite for Neural ODE Experiments
Runs generate_all_blocks.py first, then systematically tests multiple training configurations
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import torch
import traceback


class AutomatedTrainingSuite:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.data_dir = Path("/media/visitor/Extreme SSD/data/complete_blocks")
        self.splits_dir = self.script_dir / "data_splits" / "mini"
        self.output_dir = self.script_dir / "automated_outputs"
        self.log_file = self.output_dir / f"automation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize log
        self.log("🤖 Automated Training Suite Initialized")
        self.log(f"📁 Output directory: {self.output_dir}")

    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

    def run_command(self, cmd, description, timeout_hours=12):
        """Run a command with timeout and logging"""
        self.log(f"🚀 Starting: {description}")
        self.log(f"📝 Command: {' '.join(cmd)}")

        start_time = time.time()
        timeout_seconds = timeout_hours * 3600

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.script_dir
            )

            # Monitor process with timeout
            output_lines = []
            while True:
                # Check if process finished
                return_code = process.poll()
                if return_code is not None:
                    # Process finished
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        output_lines.append(remaining_output)
                    break

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    self.log(f"⏰ Timeout after {timeout_hours} hours - terminating process")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    return False, f"Timeout after {timeout_hours} hours"

                # Read output
                try:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line)
                        # Show progress for long-running tasks
                        if "✅" in line or "❌" in line or "🎉" in line:
                            self.log(f"   {line.strip()}")
                except:
                    pass

                time.sleep(0.1)

            elapsed_time = time.time() - start_time

            if return_code == 0:
                self.log(f"✅ Completed: {description} ({elapsed_time / 60:.1f} minutes)")
                return True, "Success"
            else:
                self.log(f"❌ Failed: {description} (return code: {return_code})")
                # Log last few lines of output for debugging
                if output_lines:
                    self.log("📄 Last output lines:")
                    for line in output_lines[-10:]:
                        self.log(f"   {line.strip()}")
                return False, f"Process failed with return code {return_code}"

        except Exception as e:
            self.log(f"💥 Exception in {description}: {str(e)}")
            return False, f"Exception: {str(e)}"

    def run_block_generation(self):
        """Run generate_all_blocks.py until completion"""
        self.log("\n" + "=" * 60)
        self.log("🔄 PHASE 1: BLOCK GENERATION")
        self.log("=" * 60)

        generate_script = self.script_dir / "generate_all_blocks.py"

        if not generate_script.exists():
            self.log(f"❌ Block generation script not found: {generate_script}")
            return False

        cmd = [sys.executable, str(generate_script)]
        success, message = self.run_command(cmd, "Block Generation", timeout_hours=24)

        if success:
            self.log("🎉 Block generation completed successfully!")
            return True
        else:
            self.log(f"💀 Block generation failed: {message}")
            return False

    def create_training_configs(self):
        """Create all training configurations to test"""
        base_config = {
            'data_dir': str(self.data_dir),
            'splits_dir': str(self.splits_dir),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epochs': 40,
            'learning_rate': 1e-3,
            'reduced_cluster_size': 32,
            'hidden_dim': 64,
            'integrator': 'rk4',
            'use_fast_ode': True,
            'use_amp': torch.cuda.is_available(),
            'batch_size': 10,
            'max_residues': 200,
            'lr_patience': 3,
            'lr_factor': 0.5,
            'min_lr': 1e-6,
            'early_stopping_patience': 7,
            'early_stopping_min_delta': 0.001,
            'restore_best_weights': True
        }

        configs = [
            {
                'name': 'config_0_default',
                'description': 'Default configuration (baseline)',
                'changes': {}  # No changes - pure default
            },
            {
                'name': 'config_1_slow_ode',
                'description': 'Slow ODE implementation',
                'changes': {'use_fast_ode': False}
            },
            {
                'name': 'config_2_slow_ode_batch5',
                'description': 'Slow ODE with smaller batch size',
                'changes': {'use_fast_ode': False, 'batch_size': 5}
            },
            {
                'name': 'config_3_slow_ode_batch3',
                'description': 'Slow ODE with very small batch size',
                'changes': {'use_fast_ode': False, 'batch_size': 3}
            },
            {
                'name': 'config_4_slow_ode_small_proteins',
                'description': 'Slow ODE with smaller proteins (150 residues)',
                'changes': {'use_fast_ode': False, 'max_residues': 150}
            },
            {
                'name': 'config_5_slow_ode_small_proteins_batch5',
                'description': 'Slow ODE with smaller proteins and batch size',
                'changes': {'use_fast_ode': False, 'max_residues': 150, 'batch_size': 5}
            },
            {
                'name': 'config_6_large_proteins_batch5',
                'description': 'Large proteins (500 residues) with smaller batch',
                'changes': {'max_residues': 500, 'batch_size': 5}
            },
            {
                'name': 'config_7_medium_large_batch5',
                'description': 'Medium-large proteins (400 residues) with smaller batch',
                'changes': {'max_residues': 400, 'batch_size': 5}
            },
            {
                'name': 'config_8_medium_proteins_batch5',
                'description': 'Medium proteins (300 residues) with smaller batch',
                'changes': {'max_residues': 300, 'batch_size': 5}
            },
            {
                'name': 'config_9_large_model',
                'description': 'Large model dimensions',
                'changes': {'reduced_cluster_size': 128, 'hidden_dim': 128}
            },
            {
                'name': 'config_10_large_model_batch5',
                'description': 'Large model dimensions with smaller batch',
                'changes': {'reduced_cluster_size': 128, 'hidden_dim': 128, 'batch_size': 5}
            }
        ]

        # Create full configs
        full_configs = []
        for config_info in configs:
            full_config = base_config.copy()
            full_config.update(config_info['changes'])

            # Add metadata
            full_config['config_name'] = config_info['name']
            full_config['config_description'] = config_info['description']

            full_configs.append(full_config)

        return full_configs

    def run_training_config(self, config, config_index, total_configs):
        """Run a single training configuration"""
        config_name = config['config_name']
        config_desc = config['config_description']

        self.log(f"\n📊 [{config_index}/{total_configs}] Starting: {config_name}")
        self.log(f"📝 Description: {config_desc}")

        # Create experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{config_name}_{timestamp}"
        config['experiment_name'] = experiment_name
        config['output_dir'] = str(self.output_dir)

        # Log key configuration differences
        key_settings = []
        if not config.get('use_fast_ode', True):
            key_settings.append("Slow ODE")
        if config.get('batch_size', 10) != 10:
            key_settings.append(f"batch_size={config['batch_size']}")
        if config.get('max_residues', 200) != 200:
            key_settings.append(f"max_residues={config['max_residues']}")
        if config.get('reduced_cluster_size', 32) != 32:
            key_settings.append(f"cluster_size={config['reduced_cluster_size']}")
        if config.get('hidden_dim', 64) != 64:
            key_settings.append(f"hidden_dim={config['hidden_dim']}")

        self.log(f"🔧 Key settings: {', '.join(key_settings) if key_settings else 'Default'}")

        # Build command
        training_script = self.script_dir / "train_evoformer_ode.py"
        cmd = [sys.executable, str(training_script)]

        for key, value in config.items():
            if key in ['config_name', 'config_description']:
                continue  # Skip metadata

            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])

        # Run training with timeout
        success, message = self.run_command(
            cmd,
            f"Training {config_name}",
            timeout_hours=8  # 8 hour timeout per config
        )

        # Log results
        if success:
            self.log(f"✅ Completed: {config_name}")
            # Try to extract final results
            try:
                log_file = self.output_dir / f"{experiment_name}.txt"
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if 'Best validation loss:' in content:
                            for line in content.split('\n'):
                                if 'Best validation loss:' in line:
                                    self.log(f"🏆 {line.strip()}")
                                    break
            except:
                pass
        else:
            self.log(f"❌ Failed: {config_name} - {message}")

        return success, message

    def run_training_experiments(self):
        """Run all training experiments"""
        self.log("\n" + "=" * 60)
        self.log("🧪 PHASE 2: TRAINING EXPERIMENTS")
        self.log("=" * 60)

        configs = self.create_training_configs()
        total_configs = len(configs)

        self.log(f"📋 Running {total_configs} training configurations:")
        self.log("   0. Default configuration (baseline)")
        self.log("   1. Slow ODE implementation")
        self.log("   2. Slow ODE + batch_size=5")
        self.log("   3. Slow ODE + batch_size=3")
        self.log("   4. Slow ODE + max_residues=150")
        self.log("   5. Slow ODE + max_residues=150 + batch_size=5")
        self.log("   6. max_residues=500 + batch_size=5")
        self.log("   7. max_residues=400 + batch_size=5")
        self.log("   8. max_residues=300 + batch_size=5")
        self.log("   9. Large model (cluster=128, hidden=128)")
        self.log("  10. Large model + batch_size=5")

        results = []
        successful_configs = 0
        failed_configs = 0

        for i, config in enumerate(configs, 1):
            start_time = time.time()

            success, message = self.run_training_config(config, i, total_configs)

            elapsed_time = time.time() - start_time

            result = {
                'config_name': config['config_name'],
                'description': config['config_description'],
                'success': success,
                'message': message,
                'elapsed_time_minutes': elapsed_time / 60,
                'experiment_name': config.get('experiment_name', 'unknown')
            }
            results.append(result)

            if success:
                successful_configs += 1
            else:
                failed_configs += 1

            # Progress update
            self.log(f"📊 Progress: {i}/{total_configs} complete "
                     f"({successful_configs} success, {failed_configs} failed)")

        return results

    def generate_final_report(self, training_results):
        """Generate comprehensive final report"""
        self.log("\n" + "=" * 60)
        self.log("📊 FINAL REPORT")
        self.log("=" * 60)

        successful = [r for r in training_results if r['success']]
        failed = [r for r in training_results if not r['success']]

        self.log(f"✅ Successful configurations: {len(successful)}")
        self.log(f"❌ Failed configurations: {len(failed)}")
        self.log(f"📈 Success rate: {len(successful) / len(training_results) * 100:.1f}%")
        self.log(f"⏱️  Total configs run: {len(training_results)} (including default baseline)")

        total_time = sum(r['elapsed_time_minutes'] for r in training_results)
        self.log(f"🕐 Total training time: {total_time / 60:.1f} hours")

        if successful:
            self.log("\n🏆 SUCCESSFUL EXPERIMENTS:")
            for result in successful:
                self.log(f"   ✅ {result['config_name']}: {result['description']}")
                self.log(f"      Time: {result['elapsed_time_minutes']:.1f} minutes")
                self.log(f"      Output: {result['experiment_name']}")

        if failed:
            self.log("\n💀 FAILED EXPERIMENTS:")
            for result in failed:
                self.log(f"   ❌ {result['config_name']}: {result['description']}")
                self.log(f"      Error: {result['message']}")

        # Save detailed results to JSON
        report_file = self.output_dir / f"automation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_configs': len(training_results),
                    'successful': len(successful),
                    'failed': len(failed),
                    'success_rate': len(successful) / len(training_results) * 100
                },
                'results': training_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        self.log(f"\n📄 Detailed report saved: {report_file}")
        self.log(f"📁 All outputs in: {self.output_dir}")

    def run_full_suite(self):
        """Run the complete automated training suite"""
        suite_start_time = time.time()

        self.log("🚀 AUTOMATED TRAINING SUITE STARTING")
        self.log(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Phase 1: Block Generation
            block_success = self.run_block_generation()

            if not block_success:
                self.log("💀 Stopping suite due to block generation failure")
                return False

            # Phase 2: Training Experiments
            training_results = self.run_training_experiments()

            # Phase 3: Final Report
            self.generate_final_report(training_results)

            total_time = time.time() - suite_start_time
            self.log(f"\n🎉 AUTOMATION SUITE COMPLETED!")
            self.log(f"⏱️  Total time: {total_time / 3600:.1f} hours")
            self.log(f"📊 Results saved in: {self.output_dir}")

            return True

        except KeyboardInterrupt:
            self.log("\n⏹️  Suite interrupted by user")
            return False
        except Exception as e:
            self.log(f"\n💥 Suite failed with exception: {str(e)}")
            self.log(f"🔍 Traceback: {traceback.format_exc()}")
            return False


def main():
    print("🤖 AUTOMATED NEURAL ODE TRAINING SUITE")
    print("=" * 50)
    print("This script will:")
    print("1. 🔄 Run generate_all_blocks.py until completion")
    print("2. 🧪 Test 11 different training configurations (including default)")
    print("3. 📊 Generate comprehensive reports")
    print("")
    print("📋 Training configurations to test:")
    print("   • Default baseline (fast ODE, batch=10, max_res=200)")
    print("   • Slow ODE variations (4 configs)")
    print("   • Different protein sizes (3 configs)")
    print("   • Large model variations (2 configs)")
    print("")
    print("⚠️  This will run for many hours - designed for unattended operation")
    print("")

    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Aborted by user")
        return

    suite = AutomatedTrainingSuite()
    success = suite.run_full_suite()

    if success:
        print("\n✅ Automation suite completed successfully!")
        print(f"📁 Check results in: {suite.output_dir}")
    else:
        print("\n❌ Automation suite failed or was interrupted")
        print(f"📄 Check logs in: {suite.log_file}")


if __name__ == "__main__":
    main()