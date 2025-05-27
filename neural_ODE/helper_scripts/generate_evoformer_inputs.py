#!/usr/bin/env python3
"""
Generate initial Evoformer inputs for all proteins in training, validation, and testing splits

This script processes each protein by:
1. Finding the FASTA file and alignment data
2. Running OpenFold's inference pipeline to generate initial Evoformer blocks
3. Saving the m_block_0.pt and z_block_0.pt files for neural ODE training

Based on evoformer_init_script.sh but adapted for batch processing

Usage: python generate_evoformer_inputs.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


class EvoformerInputGenerator:
    def __init__(self, data_dir: str, config_preset: str = "model_1_ptm",
                 device: str = "cuda:0", template_mmcif_dir: str = None):
        self.data_dir = Path(data_dir)
        self.config_preset = config_preset
        self.device = device

        # Set default template directory if not provided
        if template_mmcif_dir is None:
            # Assume it's in the OpenFold installation
            script_dir = Path(__file__).parent
            openfold_root = script_dir.parent.parent  # Go up two levels from neural_ODE/helper_scripts to openFold
            self.template_mmcif_dir = openfold_root / "openfold" / "data" / "pdb70_mmcif" / "mmcif_files"
        else:
            self.template_mmcif_dir = Path(template_mmcif_dir)

        # Find the OpenFold script - it's in the openFold root directory
        script_dir = Path(__file__).parent
        openfold_root = script_dir.parent.parent  # Go up two levels from neural_ODE/helper_scripts to openFold
        self.openfold_script = openfold_root / "evoformer_init" / "run_evoformer_init.py"

        # Statistics
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }

        # Get Python path (prefer conda environment)
        self.python_path = self.get_python_path()

    def get_python_path(self) -> str:
        """Get the appropriate Python interpreter path"""
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            return os.path.join(conda_prefix, 'bin', 'python')
        else:
            return 'python'

    def load_protein_splits(self) -> Dict[str, List[str]]:
        """Load protein splits from JSON file"""
        json_file = self.data_dir / 'balanced_protein_splits.json'

        if not json_file.exists():
            print(f"❌ Error: JSON file not found: {json_file}")
            print("Please run create_balanced_protein_splits.py first.")
            sys.exit(1)

        print(f"📄 Loading splits from {json_file}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            splits = {}
            for split_name in ['training', 'validation', 'testing']:
                if split_name in data['splits']:
                    splits[split_name] = data['splits'][split_name]['pdb_chains']
                    print(f"   {split_name}: {len(splits[split_name])} chains")
                else:
                    splits[split_name] = []
                    print(f"   {split_name}: 0 chains (not found)")

            return splits

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON file: {e}")
            sys.exit(1)

    def check_prerequisites(self) -> bool:
        """Check if all required files and directories exist"""
        print("🔍 Checking prerequisites...")

        # Check OpenFold script
        if not self.openfold_script.exists():
            print(f"❌ OpenFold script not found: {self.openfold_script}")
            return False
        print(f"   ✅ OpenFold script: {self.openfold_script}")

        # Check template directory
        if not self.template_mmcif_dir.exists():
            print(f"❌ Template directory not found: {self.template_mmcif_dir}")
            print("   Please download PDB70 templates or specify correct path")
            return False
        print(f"   ✅ Template directory: {self.template_mmcif_dir}")

        # Check Python environment
        try:
            result = subprocess.run([self.python_path, '--version'],
                                    capture_output=True, text=True, check=True)
            print(f"   ✅ Python: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ Python interpreter not found: {self.python_path}")
            return False

        return True

    def setup_protein_directories(self, split: str, chain: str) -> Dict[str, Path]:
        """Setup directory structure for a protein"""
        split_dir = self.data_dir / split

        # Input directories
        fasta_dir = split_dir / 'fasta_data'
        alignments_dir = split_dir / 'alignments'

        # Output directories
        blocks_dir = split_dir / 'blocks'
        protein_blocks_dir = blocks_dir / f"{chain}_evoformer_blocks"

        # Create output directories
        blocks_dir.mkdir(exist_ok=True)
        protein_blocks_dir.mkdir(exist_ok=True)

        return {
            'fasta_dir': fasta_dir,
            'alignments_dir': alignments_dir,
            'blocks_dir': blocks_dir,
            'protein_blocks_dir': protein_blocks_dir,
            'fasta_file': fasta_dir / f"{chain}.fasta",
            'alignment_dir': alignments_dir / chain
        }

    def check_existing_blocks(self, protein_blocks_dir: Path) -> bool:
        """Check if Evoformer blocks already exist"""
        recycle_dir = protein_blocks_dir / "recycle_0"
        if recycle_dir.exists():
            m_block_0 = recycle_dir / "m_block_0.pt"
            z_block_0 = recycle_dir / "z_block_0.pt"
            if m_block_0.exists() and z_block_0.exists():
                return True
        return False

    def prepare_fasta_directory(self, chain: str, fasta_file: Path, temp_fasta_dir: Path) -> bool:
        """Prepare FASTA directory for OpenFold processing"""
        if not fasta_file.exists():
            print(f"      ❌ FASTA file not found: {fasta_file}")
            return False

        # Create temporary FASTA directory structure that OpenFold expects
        temp_fasta_dir.mkdir(parents=True, exist_ok=True)

        # Copy FASTA file to expected location
        target_fasta = temp_fasta_dir / f"{chain}.fasta"

        try:
            # Copy content
            with open(fasta_file, 'r') as src, open(target_fasta, 'w') as dst:
                dst.write(src.read())
            return True
        except Exception as e:
            print(f"      ❌ Error copying FASTA file: {e}")
            return False

    def run_openfold_inference(self, chain: str, dirs: Dict[str, Path]) -> bool:
        """Run OpenFold inference to generate Evoformer blocks"""

        # Create temporary FASTA directory
        temp_fasta_dir = dirs['protein_blocks_dir'] / 'temp_fasta'

        # Prepare FASTA
        if not self.prepare_fasta_directory(chain, dirs['fasta_file'], temp_fasta_dir):
            return False

        print(f"      🧬 Running OpenFold inference...")

        # Build OpenFold command
        cmd = [
            self.python_path,
            str(self.openfold_script),
            str(temp_fasta_dir),  # FASTA directory
            str(self.template_mmcif_dir),  # Template directory
            '--output_dir', str(dirs['protein_blocks_dir']),
            '--use_precomputed_alignments', str(dirs['alignments_dir']),
            '--config_preset', self.config_preset,
            '--model_device', self.device,
            '--save_intermediates',  # This saves the Evoformer blocks
            '--save_outputs'
        ]

        try:
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=dirs['protein_blocks_dir'].parent  # Run from blocks directory
            )

            if result.returncode == 0:
                print(f"      ✅ OpenFold inference completed")

                # Check if the expected files were created
                recycle_dir = dirs['protein_blocks_dir'] / "recycle_0"
                if recycle_dir.exists():
                    m_block_0 = recycle_dir / "m_block_0.pt"
                    z_block_0 = recycle_dir / "z_block_0.pt"

                    if m_block_0.exists() and z_block_0.exists():
                        print(f"      ✅ Evoformer blocks generated successfully")

                        # Clean up temporary FASTA directory
                        try:
                            import shutil
                            shutil.rmtree(temp_fasta_dir)
                        except:
                            pass

                        return True
                    else:
                        print(f"      ❌ Expected block files not found")
                        print(f"         Missing: {m_block_0 if not m_block_0.exists() else z_block_0}")
                else:
                    print(f"      ❌ Recycle directory not created: {recycle_dir}")

            else:
                print(f"      ❌ OpenFold failed with return code {result.returncode}")
                if result.stderr:
                    print(f"         Error: {result.stderr[:200]}...")

            return False

        except subprocess.TimeoutExpired:
            print(f"      ❌ OpenFold timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"      ❌ Error running OpenFold: {e}")
            return False

    def process_protein(self, split: str, chain: str) -> bool:
        """Process a single protein to generate Evoformer inputs"""
        print(f"   🧬 Processing {chain}:")

        # Setup directories
        dirs = self.setup_protein_directories(split, chain)

        # Check if already processed
        if self.check_existing_blocks(dirs['protein_blocks_dir']):
            print(f"      ✅ Evoformer blocks already exist, skipping")
            self.stats['skipped'] += 1
            return True

        # Check if input files exist
        if not dirs['fasta_file'].exists():
            print(f"      ❌ FASTA file not found: {dirs['fasta_file']}")
            self.stats['failed'] += 1
            return False

        if not dirs['alignment_dir'].exists():
            print(f"      ❌ Alignment directory not found: {dirs['alignment_dir']}")
            self.stats['failed'] += 1
            return False

        # Run OpenFold inference
        if self.run_openfold_inference(chain, dirs):
            self.stats['successful'] += 1
            return True
        else:
            self.stats['failed'] += 1
            return False

    def process_split(self, split_name: str, chains: List[str]):
        """Process all proteins in a split"""
        print(f"\n🔄 Processing {split_name} split ({len(chains)} proteins)")
        print("=" * 60)

        for i, chain in enumerate(chains):
            print(f"\n[{i + 1}/{len(chains)}] {split_name}/{chain}")

            self.process_protein(split_name, chain)

            # Progress update every 5 proteins
            if (i + 1) % 5 == 0:
                print(f"\n   📊 Progress: {i + 1}/{len(chains)} proteins processed")
                print(f"      ✅ Successful: {self.stats['successful']}")
                print(f"      ⏭️  Skipped: {self.stats['skipped']}")
                print(f"      ❌ Failed: {self.stats['failed']}")

            # Small delay to prevent overwhelming the system
            time.sleep(2)

        print(f"\n✅ {split_name} split completed")

    def print_final_stats(self):
        """Print final processing statistics"""
        print("\n" + "=" * 60)
        print("📊 FINAL PROCESSING STATISTICS")
        print("=" * 60)

        total = self.stats['successful'] + self.stats['failed'] + self.stats['skipped']

        print(f"✅ Successful: {self.stats['successful']}")
        print(f"⏭️  Skipped (already exists): {self.stats['skipped']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"📈 Success rate: {self.stats['successful'] / max(total, 1) * 100:.1f}%")

        print(f"\n📁 Output structure:")
        for split in ['training', 'validation', 'testing']:
            split_dir = self.data_dir / split / 'blocks'
            if split_dir.exists():
                block_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.endswith('_evoformer_blocks')]
                print(f"  {split}/blocks/ ({len(block_dirs)} protein directories)")

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

    def run(self):
        """Run the complete Evoformer input generation pipeline"""
        print("🚀 Evoformer Input Generator")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🔧 Config preset: {self.config_preset}")
        print(f"💻 Device: {self.device}")
        print(f"🗂️  Template directory: {self.template_mmcif_dir}")

        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites check failed. Please fix the issues above.")
            sys.exit(1)

        # Load protein splits
        splits = self.load_protein_splits()

        # Process each split
        for split_name in ['training', 'validation', 'testing']:
            if splits[split_name]:
                self.process_split(split_name, splits[split_name])

        # Print final statistics
        self.print_final_stats()

        print(f"\n🎯 Evoformer input generation completed!")
        print(f"\n💡 Next steps:")
        print(f"   1. Use the generated blocks for neural ODE training")
        print(f"   2. Check failed proteins and investigate issues")
        print(f"   3. The m_block_0.pt and z_block_0.pt files are ready for training")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Evoformer inputs for protein splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Set default data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '..', 'data')
    default_data_dir = os.path.abspath(default_data_dir)

    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                        help='Data directory containing the splits')
    parser.add_argument('--config-preset', type=str, default='model_1_ptm',
                        help='OpenFold config preset')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference (cuda:0, cpu, etc.)')
    parser.add_argument('--template-mmcif-dir', type=str, default=None,
                        help='Directory containing template mmCIF files')

    args = parser.parse_args()

    # Create and run the generator
    generator = EvoformerInputGenerator(
        data_dir=args.data_dir,
        config_preset=args.config_preset,
        device=args.device,
        template_mmcif_dir=args.template_mmcif_dir
    )

    generator.run()


if __name__ == '__main__':
    main()