#!/usr/bin/env python3
"""
Evoformer 48 Iterations Script
Python version with automatic resume from last existing block
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse
import shutil
from typing import Optional, Tuple, List


class EvoformerIterationRunner:
    """Runs multiple Evoformer iterations with automatic resume capability"""

    def __init__(self, protein_id: str, data_base_dir: Optional[str] = None,
                 target_iterations: int = 48, config_preset: str = "model_1_ptm",
                 device: str = "cuda:0"):

        self.protein_id = protein_id
        self.target_iterations = target_iterations
        self.config_preset = config_preset
        self.device = device

        # Set up directory paths
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent

        if data_base_dir:
            self.data_base_dir = Path(data_base_dir)
        else:
            self.data_base_dir = self.script_dir / "data" / "quick_inference_data"

        self.protein_data_dir = self.data_base_dir / f"{protein_id}_evoformer_blocks" / "recycle_0"

        # Find Python interpreter
        self.python_path = self._find_python()

        # Find evoformer iteration script
        self.evoformer_script = self._find_evoformer_script()

        # Statistics
        self.iterations_completed = 0
        self.iterations_skipped = 0
        self.start_time = None
        self.total_time = 0

    def _find_python(self) -> str:
        """Find the appropriate Python interpreter"""
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_python = Path(conda_prefix) / "bin" / "python"
            if conda_python.exists():
                return str(conda_python)
        return shutil.which('python') or sys.executable

    def _find_evoformer_script(self) -> Path:
        """Find the evoformer iteration script"""
        possible_locations = [
            self.project_root / "evoformer_iter" / "run_evoformer_iter.py",
            self.script_dir / "evoformer_iter" / "run_evoformer_iter.py",
            self.script_dir / "run_evoformer_iter.py",
            self.project_root / "run_evoformer_iter.py"
        ]

        for script_path in possible_locations:
            if script_path.exists():
                return script_path

        raise FileNotFoundError(
            f"Could not find evoformer iteration script in any of these locations:\n" +
            "\n".join(f"  - {loc}" for loc in possible_locations)
        )

    def find_existing_blocks(self) -> Tuple[int, int]:
        """Find the last existing m_block and z_block indices"""
        if not self.protein_data_dir.exists():
            raise FileNotFoundError(f"Protein data directory not found: {self.protein_data_dir}")

        # Find all m_block files
        m_blocks = list(self.protein_data_dir.glob("m_block_*.pt"))
        z_blocks = list(self.protein_data_dir.glob("z_block_*.pt"))

        if not m_blocks or not z_blocks:
            raise FileNotFoundError(f"No block files found in {self.protein_data_dir}")

        # Extract indices
        def extract_index(filename: Path) -> int:
            try:
                return int(filename.stem.split('_')[-1])
            except (ValueError, IndexError):
                return -1

        m_indices = [extract_index(f) for f in m_blocks if extract_index(f) >= 0]
        z_indices = [extract_index(f) for f in z_blocks if extract_index(f) >= 0]

        if not m_indices or not z_indices:
            raise ValueError("Could not parse block file indices")

        last_m = max(m_indices)
        last_z = max(z_indices)

        return last_m, last_z

    def validate_existing_blocks(self, m_index: int, z_index: int) -> bool:
        """Validate that the block files exist and indices match"""
        if m_index != z_index:
            print(f"⚠️  Warning: Mismatched indices - m_block: {m_index}, z_block: {z_index}")
            print(f"   Using the lower index: {min(m_index, z_index)}")
            return False

        m_file = self.protein_data_dir / f"m_block_{m_index}.pt"
        z_file = self.protein_data_dir / f"z_block_{z_index}.pt"

        if not m_file.exists() or not z_file.exists():
            print(f"❌ Block files missing: {m_file.name}, {z_file.name}")
            return False

        return True

    def get_starting_block(self) -> int:
        """Determine the starting block index for iterations"""
        try:
            last_m, last_z = self.find_existing_blocks()

            # Use the minimum of the two indices for safety
            last_valid = min(last_m, last_z)

            # Validate the files exist
            if self.validate_existing_blocks(last_valid, last_valid):
                return last_valid
            else:
                # Fall back to previous valid block
                for idx in range(last_valid - 1, -1, -1):
                    if self.validate_existing_blocks(idx, idx):
                        return idx
                raise ValueError("No valid block files found")

        except Exception as e:
            print(f"❌ Error finding existing blocks: {e}")
            raise

    def run_single_iteration(self, current_index: int) -> bool:
        """Run a single Evoformer iteration"""
        m_path = self.protein_data_dir / f"m_block_{current_index}.pt"
        z_path = self.protein_data_dir / f"z_block_{current_index}.pt"
        next_m_path = self.protein_data_dir / f"m_block_{current_index + 1}.pt"
        next_z_path = self.protein_data_dir / f"z_block_{current_index + 1}.pt"

        # Check if next iteration already exists
        if next_m_path.exists() and next_z_path.exists():
            print(f"  ⏭️  Block {current_index + 1} already exists, skipping...")
            self.iterations_skipped += 1
            return True

        # Build command
        cmd = [
            self.python_path,
            str(self.evoformer_script),
            "--m_path", str(m_path),
            "--z_path", str(z_path),
            "--output_dir", str(self.protein_data_dir),
            "--config_preset", self.config_preset,
            "--device", self.device
        ]

        # Set up environment
        env = os.environ.copy()
        pythonpath = str(self.project_root)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{pythonpath}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = pythonpath

        print(f"  ⏳ Running iteration {current_index} → {current_index + 1}...")

        try:
            # Start timer for this iteration
            iter_start = time.perf_counter()

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per iteration
                cwd=str(self.project_root),
                env=env
            )

            iter_time = time.perf_counter() - iter_start

            # Check if successful
            if result.returncode == 0 and next_m_path.exists() and next_z_path.exists():
                print(f"  ✅ Completed in {iter_time:.1f}s")
                self.iterations_completed += 1
                return True
            else:
                print(f"  ❌ Failed (return code: {result.returncode})")
                if result.stderr:
                    print(f"     Error: {result.stderr.strip()[:100]}...")
                return False

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"  💥 Error: {e}")
            return False

    def run_iterations(self) -> bool:
        """Run all iterations from current state to target"""
        print(f"🚀 Evoformer 48 Iterations - {self.protein_id}")
        print(f"📁 Data directory: {self.protein_data_dir}")
        print(f"🎯 Target iterations: {self.target_iterations}")
        print(f"💻 Device: {self.device}")
        print(f"📜 Script: {self.evoformer_script}")
        print("")

        # Find starting point
        try:
            start_index = self.get_starting_block()
            print(f"📍 Starting from block {start_index}")
        except Exception as e:
            print(f"❌ Could not determine starting block: {e}")
            return False

        # Check if already complete
        if start_index >= self.target_iterations:
            print(f"✅ Already complete! Found blocks up to {start_index}")
            return True

        # Run iterations
        self.start_time = time.perf_counter()

        print(f"⏳ Running iterations {start_index + 1} through {self.target_iterations}...")
        print("=" * 60)

        current_index = start_index
        failed_iterations = 0
        max_failures = 3  # Allow up to 3 consecutive failures

        while current_index < self.target_iterations:
            iteration_num = current_index + 1
            print(f"[{iteration_num}/{self.target_iterations}] Processing block {current_index} → {iteration_num}")

            if self.run_single_iteration(current_index):
                current_index += 1
                failed_iterations = 0  # Reset failure counter on success
            else:
                failed_iterations += 1
                if failed_iterations >= max_failures:
                    print(f"❌ Too many consecutive failures ({failed_iterations}), stopping")
                    break
                print(f"⚠️  Failure {failed_iterations}/{max_failures}, continuing...")
                current_index += 1  # Skip this iteration and continue

        # Calculate total time
        self.total_time = time.perf_counter() - self.start_time

        # Print summary
        self._print_summary(start_index, current_index)

        return current_index >= self.target_iterations

    def _print_summary(self, start_index: int, final_index: int):
        """Print completion summary"""
        print("\n" + "=" * 60)
        print("📊 ITERATION SUMMARY")
        print("=" * 60)

        print(f"🏁 Final block index: {final_index}")
        print(f"✅ Iterations completed: {self.iterations_completed}")
        print(f"⏭️  Iterations skipped: {self.iterations_skipped}")
        print(f"⏱️  Total time: {self.total_time:.1f}s")

        if self.iterations_completed > 0:
            avg_time = self.total_time / self.iterations_completed
            print(f"📈 Average time per iteration: {avg_time:.1f}s")

        # Check completion status
        if final_index >= self.target_iterations:
            print(f"🎉 SUCCESS: All {self.target_iterations} iterations completed!")
        else:
            remaining = self.target_iterations - final_index
            print(f"⚠️  INCOMPLETE: {remaining} iterations remaining")

        # Show final files
        final_m = self.protein_data_dir / f"m_block_{final_index}.pt"
        final_z = self.protein_data_dir / f"z_block_{final_index}.pt"

        if final_m.exists() and final_z.exists():
            m_size = final_m.stat().st_size / 1024 / 1024
            z_size = final_z.stat().st_size / 1024 / 1024
            print(f"📁 Final files: m_block_{final_index}.pt ({m_size:.1f}MB), z_block_{final_index}.pt ({z_size:.1f}MB)")

    def list_existing_blocks(self):
        """List all existing block files"""
        if not self.protein_data_dir.exists():
            print(f"❌ Directory not found: {self.protein_data_dir}")
            return

        m_blocks = sorted(self.protein_data_dir.glob("m_block_*.pt"))
        z_blocks = sorted(self.protein_data_dir.glob("z_block_*.pt"))

        print(f"📁 Existing blocks in {self.protein_data_dir}:")
        print(f"   M blocks: {len(m_blocks)} files")
        print(f"   Z blocks: {len(z_blocks)} files")

        # Show range
        if m_blocks and z_blocks:
            def extract_index(f):
                return int(f.stem.split('_')[-1])

            m_indices = [extract_index(f) for f in m_blocks]
            z_indices = [extract_index(f) for f in z_blocks]

            print(f"   M range: {min(m_indices)} - {max(m_indices)}")
            print(f"   Z range: {min(z_indices)} - {max(z_indices)}")

            # Check for gaps
            m_set = set(m_indices)
            z_set = set(z_indices)
            expected = set(range(min(m_indices), max(m_indices) + 1))

            missing_m = expected - m_set
            missing_z = expected - z_set

            if missing_m:
                print(f"   ⚠️  Missing M blocks: {sorted(missing_m)}")
            if missing_z:
                print(f"   ⚠️  Missing Z blocks: {sorted(missing_z)}")


# ===========================================================================================
# CONFIGURATION SECTION - Set your parameters here for PyCharm execution
# ===========================================================================================

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()

# Set these values for direct execution in PyCharm
PROTEIN_ID = "5i5h_A"  # Change this to your protein ID

# Set paths relative to script directory
DATA_DIR = SCRIPT_DIR / "data" / "quick_inference_data"  # Relative to script location
OUTPUT_DIR = SCRIPT_DIR                    # Relative to script location
# Or use None for auto-detection:
# DATA_DIR = None    # Auto-detects as script_dir/data/quick_inference_data
# OUTPUT_DIR = None  # Uses default output handling

# Other settings
TARGET_ITERATIONS = 48
CONFIG_PRESET = "model_1_ptm"
DEVICE = "cuda:0"      # or "cpu" for CPU-only
DRY_RUN = False        # Set to True to see what would happen without running
LIST_BLOCKS = False    # Set to True to just list existing blocks

# ===========================================================================================


def main():
    """Main entry point - works both from command line and direct execution"""

    # Check if running from command line (has arguments) or direct execution
    if len(sys.argv) > 1:
        # Command line mode - use argparse
        parser = argparse.ArgumentParser(
            description='Run Evoformer iterations with auto-resume',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument('protein_id', type=str, help='Protein ID (e.g., 4d0f_A)')
        parser.add_argument('--data-dir', type=str, help='Base data directory')
        parser.add_argument('--iterations', type=int, default=48, help='Target number of iterations')
        parser.add_argument('--config-preset', type=str, default='model_1_ptm', help='OpenFold config preset')
        parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')
        parser.add_argument('--list-blocks', action='store_true', help='List existing blocks and exit')
        parser.add_argument('--dry-run', action='store_true', help='Show what would be done without running')

        args = parser.parse_args()

        # Use command line arguments
        protein_id = args.protein_id
        data_dir = args.data_dir
        target_iterations = args.iterations
        config_preset = args.config_preset
        device = args.device
        list_blocks = args.list_blocks
        dry_run = args.dry_run

    else:
        # Direct execution mode - use configuration section
        print("🔧 Running in PyCharm/Direct execution mode")
        print("   Using configuration from script header")
        print("")

        protein_id = PROTEIN_ID
        data_dir = DATA_DIR
        target_iterations = TARGET_ITERATIONS
        config_preset = CONFIG_PRESET
        device = DEVICE
        list_blocks = LIST_BLOCKS
        dry_run = DRY_RUN

        # Validate required settings
        if not protein_id:
            print("❌ Error: PROTEIN_ID must be set in the configuration section")
            print("   Edit the script and set PROTEIN_ID = 'your_protein_id'")
            return

    # Create runner
    try:
        runner = EvoformerIterationRunner(
            protein_id=protein_id,
            data_base_dir=data_dir,
            target_iterations=target_iterations,
            config_preset=config_preset,
            device=device
        )

        print(f"📋 Configuration:")
        print(f"   Protein ID: {protein_id}")
        print(f"   Data directory: {runner.protein_data_dir}")
        print(f"   Target iterations: {target_iterations}")
        print(f"   Device: {device}")
        print("")

    except Exception as e:
        print(f"❌ Failed to initialize runner: {e}")
        return

    # Handle list blocks option
    if list_blocks:
        runner.list_existing_blocks()
        return

    # Handle dry run
    if dry_run:
        try:
            start_index = runner.get_starting_block()
            print(f"🔍 DRY RUN: Would start from block {start_index}")
            print(f"🎯 Would run iterations {start_index + 1} through {target_iterations}")
            print(f"📊 Total iterations needed: {target_iterations - start_index}")
        except Exception as e:
            print(f"❌ Dry run failed: {e}")
        return

    # Run iterations
    try:
        success = runner.run_iterations()
        if success:
            print("\n🎉 All iterations completed successfully!")
        else:
            print("\n⚠️  Some iterations may have failed - check the summary above")
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()