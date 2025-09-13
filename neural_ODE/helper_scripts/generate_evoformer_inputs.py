#!/usr/bin/env python3
"""
Generate Evoformer inputs for all proteins in training, validation, and testing splits
Modified to use argparse with sensible defaults instead of hardcoded paths
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
    def __init__(self, data_dir: str, splits_dir: str, config_preset: str = "model_1_ptm",
                 device: str = "cuda:0", template_mmcif_dir: str = None):
        # Use provided directories instead of hardcoded paths
        self.data_dir = Path(data_dir)
        self.fasta_dir = self.data_dir / "fasta_data"
        self.precomputed_alignments = self.data_dir / "alignments"
        self.splits_dir = Path(splits_dir)

        # Output directories
        self.incomplete_blocks_dir = self.data_dir / "incomplete_blocks"
        self.complete_blocks_dir = self.data_dir / "complete_blocks"
        self.endpoint_blocks_dir = self.data_dir / "endpoint_blocks"

        # Set default template directory if not provided
        if template_mmcif_dir is None:
            self.template_mmcif_dir = self.data_dir / "template_data" / "pdb70_mmcif" / "mmcif_files"
        else:
            self.template_mmcif_dir = Path(template_mmcif_dir)

        self.config_preset = config_preset
        self.device = device

        # Find the OpenFold script and root directory
        script_dir = Path(__file__).parent
        # Look for openfold root - try multiple possible locations
        possible_roots = [
            script_dir.parent.parent,  # neural_ODE/helper_scripts/../..
            script_dir.parent,  # neural_ODE/helper_scripts/..
            Path.cwd(),  # Current working directory
        ]

        self.root_dir = None
        for root in possible_roots:
            if (root / "evoformer_init" / "run_evoformer_init.py").exists():
                self.root_dir = root
                break

        if self.root_dir is None:
            raise FileNotFoundError("Could not find OpenFold root directory with evoformer_init")

        self.openfold_script = self.root_dir / "evoformer_init" / "run_evoformer_init.py"

        # Find the parameters file
        self.params_file = self.find_params_file()

        # Statistics
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped_incomplete': 0,
            'skipped_complete': 0,
            'skipped_endpoint': 0,
            'errors': []
        }

        self.python_path = self.get_python_path()

    def find_params_file(self) -> Path:
        """Find the parameters file in various possible locations"""
        param_filename = f"params_{self.config_preset}.npz"

        possible_locations = [
            self.root_dir / "save_intermediates" / "openfold" / "resources" / "params" / param_filename,
            self.root_dir / "openfold" / "resources" / "params" / param_filename
        ]
        print(possible_locations)

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

        if not self.fasta_dir.exists():
            print(f"❌ FASTA directory not found: {self.fasta_dir}")
            return False
        print(f"   ✅ FASTA directory: {self.fasta_dir}")

        try:
            result = subprocess.run([self.python_path, '--version'], capture_output=True, check=True)
            print(f"   ✅ Python: {self.python_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ Python not found: {self.python_path}")
            return False

        return True

    def protein_already_processed(self, protein_id: str) -> Tuple[bool, str]:
        """Check if protein already has complete evoformer blocks"""
        # Check endpoint_blocks first (highest priority)
        endpoint_dir = self.endpoint_blocks_dir / f"{protein_id}_evoformer_blocks" / "recycle_0"
        if endpoint_dir.exists():
            m_block_48 = endpoint_dir / "m_block_48.pt"
            z_block_48 = endpoint_dir / "z_block_48.pt"
            if m_block_48.exists() and z_block_48.exists():
                return True, "endpoint"

        # Check complete_blocks
        complete_dir = self.complete_blocks_dir / f"{protein_id}_evoformer_blocks" / "recycle_0"
        if complete_dir.exists():
            m_block_48 = complete_dir / "m_block_48.pt"
            z_block_48 = complete_dir / "z_block_48.pt"
            if m_block_48.exists() and z_block_48.exists():
                return True, "complete"

        # Check incomplete_blocks for block 0
        incomplete_dir = self.incomplete_blocks_dir / f"{protein_id}_evoformer_blocks" / "recycle_0"
        if incomplete_dir.exists():
            m_block_0 = incomplete_dir / "m_block_0.pt"
            z_block_0 = incomplete_dir / "z_block_0.pt"
            if m_block_0.exists() and z_block_0.exists():
                return True, "incomplete"

        return False, "none"

    def process_protein(self, protein_id: str, split_name: str) -> bool:
        """Process a single protein to generate evoformer inputs"""
        print(f"\n[{split_name.upper()}] Processing {protein_id}")

        # Check if already processed
        is_processed, location = self.protein_already_processed(protein_id)
        if is_processed:
            if location == "endpoint":
                print(f"   ⏭️  Skipping - already in endpoint_blocks")
                self.stats['skipped_endpoint'] += 1
            elif location == "complete":
                print(f"   ⏭️  Skipping - already in complete_blocks")
                self.stats['skipped_complete'] += 1
            elif location == "incomplete":
                print(f"   ⏭️  Skipping - already in incomplete_blocks")
                self.stats['skipped_incomplete'] += 1
            return True

        # Check input files exist
        fasta_file = self.fasta_dir / f"{protein_id}.fasta"
        alignment_dir = self.precomputed_alignments / protein_id

        if not fasta_file.exists():
            print(f"   ❌ FASTA file not found: {fasta_file}")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{protein_id}: FASTA file missing")
            return False

        if not alignment_dir.exists():
            print(f"   ❌ Alignment directory not found: {alignment_dir}")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{protein_id}: Alignment directory missing")
            return False

        # Create directories for protein-specific processing
        temp_fasta_dir = self.data_dir / "temp_fasta" / protein_id
        temp_fasta_dir.mkdir(parents=True, exist_ok=True)

        # Copy FASTA file to temp directory (OpenFold expects directory structure)
        temp_fasta_file = temp_fasta_dir / f"{protein_id}.fasta"
        temp_fasta_file.write_text(fasta_file.read_text())

        # Set up output directory structure
        protein_blocks_dir = self.incomplete_blocks_dir / f"{protein_id}_evoformer_blocks"

        try:
            # Run OpenFold inference to generate initial evoformer blocks
            print(f"   🧬 Running OpenFold inference...")

            cmd = [
                self.python_path,
                str(self.openfold_script),
                str(temp_fasta_dir),
                str(self.template_mmcif_dir),
                '--output_dir', str(self.incomplete_blocks_dir),
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

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(self.root_dir),
                env=env
            )

            if result.returncode == 0:
                # Verify output files were created
                recycle_dir = protein_blocks_dir / "recycle_0"
                m_block_0 = recycle_dir / "m_block_0.pt"
                z_block_0 = recycle_dir / "z_block_0.pt"

                if m_block_0.exists() and z_block_0.exists():
                    print(f"   ✅ Successfully generated evoformer inputs")
                    self.stats['successful'] += 1
                    return True
                else:
                    print(f"   ❌ OpenFold completed but output files missing")
                    self.stats['failed'] += 1
                    self.stats['errors'].append(f"{protein_id}: Output files missing after OpenFold")
                    return False
            else:
                print(f"   ❌ OpenFold failed with return code {result.returncode}")
                if result.stderr:
                    print(f"      Error: {result.stderr[:200]}...")
                self.stats['failed'] += 1
                self.stats['errors'].append(f"{protein_id}: OpenFold failed (code {result.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print(f"   ❌ OpenFold timed out after 30 minutes")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{protein_id}: OpenFold timeout")
            return False
        except Exception as e:
            print(f"   ❌ Error running OpenFold: {e}")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{protein_id}: Exception - {str(e)}")
            return False
        finally:
            # Clean up temp directory
            if temp_fasta_dir.exists():
                import shutil
                shutil.rmtree(temp_fasta_dir)

    def run(self):
        """Run the complete Evoformer input generation pipeline"""
        print("🚀 Evoformer Input Generator")
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
            return

        # Load protein splits
        splits = self.load_protein_splits()
        total_proteins = sum(len(chains) for chains in splits.values())

        if total_proteins == 0:
            print("❌ No proteins found in splits")
            return

        print(f"\n📊 Found {total_proteins} total proteins:")
        for split_name, chains in splits.items():
            print(f"  {split_name}: {len(chains)} proteins")

        # Create output directories
        self.incomplete_blocks_dir.mkdir(parents=True, exist_ok=True)
        self.complete_blocks_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint_blocks_dir.mkdir(parents=True, exist_ok=True)

        # Process all proteins
        start_time = time.time()
        processed_count = 0

        for split_name, protein_list in splits.items():
            for protein_id in protein_list:
                processed_count += 1
                print(f"\n[{processed_count}/{total_proteins}] Processing {protein_id}")

                self.process_protein(protein_id, split_name)

                # Progress update every 5 proteins
                if processed_count % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"\n   📊 Progress: {processed_count}/{total_proteins} proteins processed")
                    print(f"      Time elapsed: {elapsed / 60:.1f} minutes")
                    print(
                        f"      Success rate: {self.stats['successful']}/{processed_count} ({self.stats['successful'] / processed_count * 100:.1f}%)")

        # Print final statistics
        elapsed = time.time() - start_time
        print(f"\n🎯 Final Results:")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⏭️  Skipped (incomplete): {self.stats['skipped_incomplete']}")
        print(f"   ⏭️  Skipped (complete): {self.stats['skipped_complete']}")
        print(f"   ⏭️  Skipped (endpoint): {self.stats['skipped_endpoint']}")
        print(f"   ⏱️  Total time: {elapsed / 60:.1f} minutes")

        if self.stats['errors']:
            print(f"\n❌ Errors encountered:")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                print(f"   - {error}")
            if len(self.stats['errors']) > 10:
                print(f"   ... and {len(self.stats['errors']) - 10} more errors")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Evoformer inputs for protein splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Get default paths relative to script location
    script_dir = Path(__file__).parent
    default_data_dir = script_dir.parent / "data"
    default_splits_dir = script_dir.parent / "data_splits" / "failed"

    parser.add_argument('--data-dir', type=str, default=str(default_data_dir),
                        help='Data directory containing alignments/ and fasta_data/ subdirectories')
    parser.add_argument('--splits-dir', type=str, default=str(default_splits_dir),
                        help='Directory containing training_chains.txt, validation_chains.txt, testing_chains.txt')
    parser.add_argument('--config-preset', type=str, default='model_1_ptm',
                        help='OpenFold config preset')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (cuda:0, cpu, etc.)')
    parser.add_argument('--template-mmcif-dir', type=str, default=None,
                        help='Directory containing template mmCIF files (default: data-dir/template_data/pdb70_mmcif/mmcif_files)')

    args = parser.parse_args()

    generator = EvoformerInputGenerator(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        config_preset=args.config_preset,
        device=args.device,
        template_mmcif_dir=args.template_mmcif_dir
    )

    generator.run()


if __name__ == '__main__':
    main()