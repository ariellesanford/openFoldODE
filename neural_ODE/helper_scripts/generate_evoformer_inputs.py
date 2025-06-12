#!/usr/bin/env python3
"""
Modified version - Generate Evoformer inputs for all proteins in training, validation, and testing splits
Uses hardcoded paths for production environment
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EvoformerInputGenerator:
    def __init__(self, config_preset: str = "model_1_ptm", device: str = "cuda:0"):
        # Hardcoded paths
        self.root_dir = Path("/home/visitor/PycharmProjects/openFold")
        self.data_dir = Path("/media/visitor/Extreme SSD/data")
        self.fasta_dir = self.data_dir / "fasta_data"
        self.template_mmcif_dir = self.data_dir / "template_data" / "pdb70_mmcif" / "mmcif_files"
        self.precomputed_alignments = self.data_dir / "alignments"
        self.splits_dir = self.root_dir / "neural_ODE" / "data_splits" / "jumbo"
        self.incomplete_blocks_dir = self.data_dir / "incomplete_blocks"
        self.complete_blocks_dir = self.data_dir / "complete_blocks"

        self.config_preset = config_preset
        self.device = device

        # Find the OpenFold script and parameters file
        self.openfold_script = self.root_dir / "evoformer_init" / "run_evoformer_init.py"
        self.params_file = self.find_params_file()

        # Statistics
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped_incomplete': 0,
            'skipped_complete': 0,
            'errors': []
        }

        self.python_path = self.get_python_path()

    def find_params_file(self) -> Path:
        """Find the parameters file in various possible locations"""
        param_filename = f"params_{self.config_preset}.npz"

        possible_locations = [
            self.root_dir / "evoformer_init" / "openfold" / "resources" / "params" / param_filename,
            self.root_dir / "openfold" / "resources" / "params" / param_filename,
            self.root_dir / "evoformer_init" / param_filename,
            self.root_dir / "evoformer_init" / "resources" / "params" / param_filename,
        ]

        for location in possible_locations:
            if location.exists():
                return location

        # If not found in expected locations, search more broadly
        for path in self.root_dir.rglob(param_filename):
            return path

        raise FileNotFoundError(f"Could not find {param_filename} in any expected location")

    def get_python_path(self) -> str:
        """Get the appropriate Python interpreter path"""
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            return os.path.join(conda_prefix, 'bin', 'python')
        else:
            return 'python'

    def load_protein_splits(self) -> Dict[str, List[str]]:
        """Load protein splits from text files"""
        splits = {}

        for split_name in ['training', 'validation', 'testing']:
            split_file = self.splits_dir / f"{split_name}_chains.txt"

            if split_file.exists():
                with open(split_file, 'r') as f:
                    chains = [line.strip() for line in f if line.strip()]
                splits[split_name] = chains
            else:
                print(f"⚠️  Split file not found: {split_file}")
                splits[split_name] = []

        return splits

    def check_prerequisites(self) -> bool:
        """Check if all required files and directories exist"""
        print("🔍 Checking prerequisites...")

        if not self.openfold_script.exists():
            print(f"❌ OpenFold script not found: {self.openfold_script}")
            return False
        print(f"   ✅ OpenFold script: {self.openfold_script}")

        if not self.template_mmcif_dir.exists():
            print(f"❌ Template directory not found: {self.template_mmcif_dir}")
            return False
        print(f"   ✅ Template directory: {self.template_mmcif_dir}")

        try:
            print(f"   ✅ Parameters file: {self.params_file}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False

        if not self.splits_dir.exists():
            print(f"❌ Splits directory not found: {self.splits_dir}")
            return False
        print(f"   ✅ Splits directory: {self.splits_dir}")

        if not self.precomputed_alignments.exists():
            print(f"❌ Precomputed alignments not found: {self.precomputed_alignments}")
            return False
        print(f"   ✅ Precomputed alignments: {self.precomputed_alignments}")

        try:
            result = subprocess.run([self.python_path, '--version'],
                                    capture_output=True, text=True, check=True)
            print(f"   ✅ Python: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ Python interpreter not found: {self.python_path}")
            return False

        return True

    def setup_protein_directories(self, chain: str) -> Dict[str, Path]:
        """Setup directory structure for a protein"""
        # All outputs go to incomplete_blocks directory
        protein_blocks_dir = self.incomplete_blocks_dir / f"{chain}_evoformer_blocks"

        self.incomplete_blocks_dir.mkdir(exist_ok=True)
        protein_blocks_dir.mkdir(exist_ok=True)

        return {
            'fasta_file': self.fasta_dir / f"{chain}.fasta",
            'alignment_dir': self.precomputed_alignments / chain,
            'protein_blocks_dir': protein_blocks_dir
        }

    def check_existing_blocks(self, protein_blocks_dir: Path) -> bool:
        """Check if Evoformer blocks already exist in incomplete_blocks"""
        recycle_dir = protein_blocks_dir / "recycle_0"
        if recycle_dir.exists():
            m_block_0 = recycle_dir / "m_block_0.pt"
            z_block_0 = recycle_dir / "z_block_0.pt"
            if m_block_0.exists() and z_block_0.exists():
                return True
        return False

    def check_complete_blocks(self, chain: str) -> bool:
        """Check if protein already exists in complete_blocks"""
        complete_protein_dir = self.complete_blocks_dir / f"{chain}_evoformer_blocks"
        if complete_protein_dir.exists():
            recycle_dir = complete_protein_dir / "recycle_0"
            if recycle_dir.exists():
                # Check for final block (block 48)
                m_block_48 = recycle_dir / "m_block_48.pt"
                z_block_48 = recycle_dir / "z_block_48.pt"
                if m_block_48.exists() and z_block_48.exists():
                    return True
        return False

    def prepare_fasta_directory(self, chain: str, fasta_file: Path, temp_fasta_dir: Path) -> bool:
        """Prepare FASTA directory for OpenFold processing"""
        if not fasta_file.exists():
            return False

        temp_fasta_dir.mkdir(parents=True, exist_ok=True)
        target_fasta = temp_fasta_dir / f"{chain}.fasta"

        try:
            with open(fasta_file, 'r') as src, open(target_fasta, 'w') as dst:
                dst.write(src.read())
            return True
        except Exception:
            return False

    def run_openfold_inference(self, chain: str, dirs: Dict[str, Path]) -> bool:
        """Run OpenFold inference to generate Evoformer blocks"""
        temp_fasta_dir = dirs['protein_blocks_dir'] / 'temp_fasta'

        if not self.prepare_fasta_directory(chain, dirs['fasta_file'], temp_fasta_dir):
            return False

        # Use the parent directory to avoid double nesting
        output_parent_dir = dirs['protein_blocks_dir'].parent

        # Build command with explicit parameters path
        cmd = [
            self.python_path,
            str(self.openfold_script),
            str(temp_fasta_dir),
            str(self.template_mmcif_dir),
            '--output_dir', str(output_parent_dir),
            '--use_precomputed_alignments', str(self.precomputed_alignments),
            '--config_preset', self.config_preset,
            '--model_device', self.device,
            '--jax_param_path', str(self.params_file),
            '--save_intermediates',
            '--save_outputs'
        ]

        # Set up environment
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{str(self.root_dir)}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(self.root_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(self.root_dir),
                env=env
            )

            if result.returncode == 0:
                # Check output files
                possible_recycle_dirs = [
                    dirs['protein_blocks_dir'] / "recycle_0",
                    dirs['protein_blocks_dir'] / f"{chain}_evoformer_blocks" / "recycle_0",
                ]

                recycle_dir = None
                for possible_dir in possible_recycle_dirs:
                    if possible_dir.exists():
                        recycle_dir = possible_dir
                        break

                if recycle_dir and recycle_dir.exists():
                    m_block_0 = recycle_dir / "m_block_0.pt"
                    z_block_0 = recycle_dir / "z_block_0.pt"

                    if m_block_0.exists() and z_block_0.exists():
                        # Move files to expected location if they're in the wrong place
                        expected_recycle_dir = dirs['protein_blocks_dir'] / "recycle_0"
                        if recycle_dir != expected_recycle_dir:
                            expected_recycle_dir.mkdir(parents=True, exist_ok=True)

                            import shutil
                            for file_path in recycle_dir.glob("*.pt"):
                                target_path = expected_recycle_dir / file_path.name
                                shutil.copy2(file_path, target_path)

                        # Clean up temporary FASTA directory
                        try:
                            import shutil
                            shutil.rmtree(temp_fasta_dir)
                        except:
                            pass

                        return True

            return False

        except subprocess.TimeoutExpired:
            self.stats['errors'].append(f"{chain}: Timeout after 30 minutes")
            return False
        except Exception as e:
            self.stats['errors'].append(f"{chain}: {str(e)}")
            return False

    def process_protein(self, chain: str) -> Tuple[bool, str]:
        """Process a single protein to generate Evoformer inputs"""

        # First check if already complete (48 blocks)
        if self.check_complete_blocks(chain):
            self.stats['skipped_complete'] += 1
            return True, "already_complete"

        # Setup directories
        dirs = self.setup_protein_directories(chain)

        # Check if already in incomplete_blocks
        if self.check_existing_blocks(dirs['protein_blocks_dir']):
            self.stats['skipped_incomplete'] += 1
            return True, "already_incomplete"

        # Check if input files exist
        if not dirs['fasta_file'].exists():
            self.stats['errors'].append(f"{chain}: FASTA file not found")
            self.stats['failed'] += 1
            return False, "fasta_missing"

        if not dirs['alignment_dir'].exists():
            self.stats['errors'].append(f"{chain}: Alignment directory not found")
            self.stats['failed'] += 1
            return False, "alignment_missing"

        # Run OpenFold inference
        if self.run_openfold_inference(chain, dirs):
            self.stats['successful'] += 1
            return True, "processed"
        else:
            self.stats['failed'] += 1
            return False, "failed"

    def process_all_proteins(self, all_chains: List[str]):
        """Process all proteins in one batch"""
        print(f"\n🔄 Processing all proteins ({len(all_chains)} total)")
        print("=" * 60)

        for i, chain in enumerate(all_chains):
            print(f"[{i + 1}/{len(all_chains)}] {chain}... ", end='', flush=True)

            success, status = self.process_protein(chain)

            if success:
                if status == "already_complete":
                    print("✅ (already complete)")
                elif status == "already_incomplete":
                    print("✅ (already in incomplete)")
                elif status == "processed":
                    print("✅")
                else:
                    print("✅")
            else:
                print("❌")

            # Progress update every 10 proteins
            if (i + 1) % 10 == 0:
                print(f"\n   📊 Progress: {i + 1}/{len(all_chains)} proteins processed")
                print(f"      ✅ Successful: {self.stats['successful']}")
                print(f"      ⏭️  Skipped (complete): {self.stats['skipped_complete']}")
                print(f"      ⏭️  Skipped (incomplete): {self.stats['skipped_incomplete']}")
                print(f"      ❌ Failed: {self.stats['failed']}")

            # Small delay to prevent overwhelming the system
            time.sleep(1)

        print(f"\n✅ All proteins processed")

    def print_final_stats(self):
        """Print final processing statistics"""
        print("\n" + "=" * 60)
        print("📊 FINAL PROCESSING STATISTICS")
        print("=" * 60)

        total = self.stats['successful'] + self.stats['failed'] + self.stats['skipped_incomplete'] + self.stats[
            'skipped_complete']

        print(f"✅ Successful: {self.stats['successful']}")
        print(f"⏭️  Skipped (already complete): {self.stats['skipped_complete']}")
        print(f"⏭️  Skipped (already in incomplete): {self.stats['skipped_incomplete']}")
        print(f"❌ Failed: {self.stats['failed']}")
        if total > 0:
            print(f"📈 Success rate: {self.stats['successful'] / total * 100:.1f}%")
            print(
                f"📈 Already processed rate: {(self.stats['skipped_complete'] + self.stats['skipped_incomplete']) / total * 100:.1f}%")

        if self.stats['errors']:
            print(f"\n❌ Error details:")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                print(f"   - {error}")
            if len(self.stats['errors']) > 10:
                print(f"   ... and {len(self.stats['errors']) - 10} more errors")

        print(f"\n📁 Output structure:")
        if self.incomplete_blocks_dir.exists():
            block_dirs = [d for d in self.incomplete_blocks_dir.iterdir() if
                          d.is_dir() and d.name.endswith('_evoformer_blocks')]
            print(f"  incomplete_blocks/ ({len(block_dirs)} protein directories)")

            # Count successful blocks
            successful_blocks = 0
            for block_dir in block_dirs:
                recycle_dir = block_dir / "recycle_0"
                if recycle_dir.exists():
                    m_block = recycle_dir / "m_block_0.pt"
                    z_block = recycle_dir / "z_block_0.pt"
                    if m_block.exists() and z_block.exists():
                        successful_blocks += 1

            print(f"    └── {successful_blocks} with complete Evoformer blocks")

        if self.complete_blocks_dir.exists():
            complete_block_dirs = [d for d in self.complete_blocks_dir.iterdir() if
                                   d.is_dir() and d.name.endswith('_evoformer_blocks')]
            print(f"  complete_blocks/ ({len(complete_block_dirs)} protein directories)")

            # Count complete blocks (have block 48)
            complete_blocks = 0
            for block_dir in complete_block_dirs:
                recycle_dir = block_dir / "recycle_0"
                if recycle_dir.exists():
                    m_block_48 = recycle_dir / "m_block_48.pt"
                    z_block_48 = recycle_dir / "z_block_48.pt"
                    if m_block_48.exists() and z_block_48.exists():
                        complete_blocks += 1

            print(f"    └── {complete_blocks} with complete 48-block sequences")

    def run(self):
        """Run the complete Evoformer input generation pipeline"""
        print("🚀 Evoformer Input Generator - Hardcoded Paths Mode")
        print(f"📁 Root directory: {self.root_dir}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 FASTA directory: {self.fasta_dir}")
        print(f"📁 Splits directory: {self.splits_dir}")
        print(f"🔧 Config preset: {self.config_preset}")
        print(f"💻 Device: {self.device}")
        print(f"🗂️  Template directory: {self.template_mmcif_dir}")
        print(f"📄 Parameters file: {self.params_file}")

        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites check failed. Please fix the issues above.")
            sys.exit(1)

        # Load protein splits and combine all chains
        print(f"\n📄 Loading protein splits...")
        splits = self.load_protein_splits()

        # Combine all chains from all splits
        all_chains = []
        for split_name, chains in splits.items():
            all_chains.extend(chains)
            print(f"   {split_name}: {len(chains)} chains")

        # Remove duplicates and sort
        all_chains = sorted(list(set(all_chains)))

        print(f"📊 Total unique proteins to process: {len(all_chains)}")

        if len(all_chains) == 0:
            print("❌ No proteins found in splits!")
            sys.exit(1)

        # Process all proteins together
        start_time = time.time()
        self.process_all_proteins(all_chains)
        end_time = time.time()

        # Print final statistics
        self.print_final_stats()

        print(f"\n🎯 Evoformer input generation completed!")
        print(f"⏱️  Total time: {(end_time - start_time) / 60:.1f} minutes")
        print(f"\n💡 Next steps:")
        print(f"   1. Use the generated blocks for neural ODE training")
        print(f"   2. Check failed proteins and investigate issues")
        print(f"   3. The m_block_0.pt and z_block_0.pt files are ready for training")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Evoformer inputs for protein splits - Hardcoded Paths Mode',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config-preset', type=str, default='model_1_ptm',
                        help='OpenFold config preset')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (cuda:0, cpu, etc.)')

    args = parser.parse_args()

    generator = EvoformerInputGenerator(
        config_preset=args.config_preset,
        device=args.device
    )

    # Confirm before starting
    print("⚠️  This will process ALL proteins in your splits (training + validation + testing)")
    print("   This may take several hours depending on the number of proteins.")

    response = input("\nDo you want to continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)

    generator.run()


if __name__ == '__main__':
    main()