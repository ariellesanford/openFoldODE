#!/usr/bin/env python3
"""
Fixed version of generate_evoformer_inputs.py with correct parameter paths
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
                 device: str = "cuda:0", template_mmcif_dir: str = None, debug: bool = False):
        self.data_dir = Path(data_dir)
        self.config_preset = config_preset
        self.device = device
        self.debug = debug

        # Set default template directory if not provided
        if template_mmcif_dir is None:
            script_dir = Path(__file__).parent
            openfold_root = script_dir.parent.parent
            self.template_mmcif_dir = openfold_root / "openfold" / "data" / "pdb70_mmcif" / "mmcif_files"
        else:
            self.template_mmcif_dir = Path(template_mmcif_dir)

        # Find the OpenFold script and root
        script_dir = Path(__file__).parent
        self.openfold_root = script_dir.parent.parent
        self.openfold_script = self.openfold_root / "evoformer_init" / "run_evoformer_init.py"

        # Find the parameters file - check multiple possible locations
        self.params_file = self.find_params_file()

        # Statistics
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }

        self.python_path = self.get_python_path()

    def find_params_file(self) -> Path:
        """Find the parameters file in various possible locations"""
        param_filename = f"params_{self.config_preset}.npz"

        possible_locations = [
            # In evoformer_init directory
            self.openfold_root / "evoformer_init" / "openfold" / "resources" / "params" / param_filename,
            # In main openfold directory
            self.openfold_root / "openfold" / "resources" / "params" / param_filename,
            # In evoformer_init root
            self.openfold_root / "evoformer_init" / param_filename,
            # Search in evoformer_init subdirectories
            self.openfold_root / "evoformer_init" / "resources" / "params" / param_filename,
        ]

        print(f"🔍 Searching for {param_filename}...")

        for location in possible_locations:
            print(f"   Checking: {location}")
            if location.exists():
                print(f"   ✅ Found at: {location}")
                return location
            else:
                print(f"   ❌ Not found")

        # If not found in expected locations, search more broadly
        print(f"   🔍 Searching entire project directory...")
        for path in self.openfold_root.rglob(param_filename):
            print(f"   ✅ Found at: {path}")
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

        if not self.openfold_script.exists():
            print(f"❌ OpenFold script not found: {self.openfold_script}")
            return False
        print(f"   ✅ OpenFold script: {self.openfold_script}")

        if not self.template_mmcif_dir.exists():
            print(f"❌ Template directory not found: {self.template_mmcif_dir}")
            print("   Please download PDB70 templates or specify correct path")
            return False
        print(f"   ✅ Template directory: {self.template_mmcif_dir}")

        try:
            # This will also try to find the params file
            print(f"   ✅ Parameters file: {self.params_file}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False

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

        temp_fasta_dir.mkdir(parents=True, exist_ok=True)
        target_fasta = temp_fasta_dir / f"{chain}.fasta"

        try:
            with open(fasta_file, 'r') as src, open(target_fasta, 'w') as dst:
                dst.write(src.read())
            return True
        except Exception as e:
            print(f"      ❌ Error copying FASTA file: {e}")
            return False

    def test_single_protein(self, chain: str, dirs: Dict[str, Path]) -> bool:
        """Test OpenFold inference on a single protein with detailed debugging"""
        temp_fasta_dir = dirs['protein_blocks_dir'] / 'temp_fasta'

        if not self.prepare_fasta_directory(chain, dirs['fasta_file'], temp_fasta_dir):
            return False

        print(f"      🧬 Running OpenFold inference with full debug output...")

        # Use the parent directory to avoid double nesting
        output_parent_dir = dirs['protein_blocks_dir'].parent

        # Build command with explicit parameters path
        cmd = [
            self.python_path,
            str(self.openfold_script),
            str(temp_fasta_dir),
            str(self.template_mmcif_dir),
            '--output_dir', str(output_parent_dir),
            '--use_precomputed_alignments', str(dirs['alignments_dir']),
            '--config_preset', self.config_preset,
            '--model_device', self.device,
            '--jax_param_path', str(self.params_file),  # Explicitly provide the params file path
            '--save_intermediates',
            '--save_outputs'
        ]

        print(f"      🔧 Command: {' '.join(cmd)}")
        print(f"      📁 Working directory: {self.openfold_root}")
        print(f"      📂 Parameters file: {self.params_file}")

        # Check if input files exist
        print(f"      🔍 Checking input files:")
        print(f"         FASTA file exists: {dirs['fasta_file'].exists()}")
        print(f"         Alignment dir exists: {dirs['alignment_dir'].exists()}")
        if dirs['alignment_dir'].exists():
            alignment_files = list(dirs['alignment_dir'].glob('*'))
            print(f"         Alignment files: {len(alignment_files)} files")
            for f in alignment_files[:5]:  # Show first 5 files
                print(f"           - {f.name}")

        # Set up environment
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{str(self.openfold_root)}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(self.openfold_root)

        print(f"      🌍 PYTHONPATH: {env['PYTHONPATH']}")

        try:
            print(f"      ⏳ Starting OpenFold subprocess...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(self.openfold_root),
                env=env
            )

            print(f"      📊 Return code: {result.returncode}")

            # Always show full output for debugging
            if result.stdout:
                print(f"      📄 STDOUT:")
                print("      " + "\n      ".join(result.stdout.split('\n')))

            if result.stderr:
                print(f"      ❌ STDERR:")
                print("      " + "\n      ".join(result.stderr.split('\n')))

            if result.returncode == 0:
                print(f"      ✅ OpenFold inference completed successfully")

                # Check output files - look in the correct nested directory
                possible_recycle_dirs = [
                    dirs['protein_blocks_dir'] / "recycle_0",  # Direct path
                    dirs['protein_blocks_dir'] / f"{chain}_evoformer_blocks" / "recycle_0",  # Nested path
                ]

                recycle_dir = None
                for possible_dir in possible_recycle_dirs:
                    if possible_dir.exists():
                        recycle_dir = possible_dir
                        print(f"      📁 Found recycle directory: {recycle_dir}")
                        break

                if recycle_dir and recycle_dir.exists():
                    m_block_0 = recycle_dir / "m_block_0.pt"
                    z_block_0 = recycle_dir / "z_block_0.pt"

                    if m_block_0.exists() and z_block_0.exists():
                        print(f"      ✅ Evoformer blocks generated successfully")
                        print(f"         m_block_0.pt: {m_block_0}")
                        print(f"         z_block_0.pt: {z_block_0}")

                        # Move files to expected location if they're in the wrong place
                        expected_recycle_dir = dirs['protein_blocks_dir'] / "recycle_0"
                        if recycle_dir != expected_recycle_dir:
                            print(f"      🔄 Moving files to expected location...")
                            expected_recycle_dir.mkdir(parents=True, exist_ok=True)

                            import shutil
                            for file_path in recycle_dir.glob("*.pt"):
                                target_path = expected_recycle_dir / file_path.name
                                shutil.copy2(file_path, target_path)
                                print(f"         Copied {file_path.name} to {target_path}")

                        return True
                    else:
                        print(f"      ❌ Expected block files not found")
                        print(f"         m_block_0.pt exists: {m_block_0.exists()}")
                        print(f"         z_block_0.pt exists: {z_block_0.exists()}")
                else:
                    print(f"      ❌ No recycle directory found in any expected location")
                    # List what directories were created
                    print(f"      📁 Contents of output directory:")
                    for item in dirs['protein_blocks_dir'].iterdir():
                        print(f"         - {item.name} ({'dir' if item.is_dir() else 'file'})")
            else:
                print(f"      ❌ OpenFold failed with return code {result.returncode}")

            return False

        except subprocess.TimeoutExpired:
            print(f"      ❌ OpenFold timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"      ❌ Error running OpenFold: {e}")
            return False

    def run_debug_single(self, split: str = 'training', limit: int = 1):
        """Run debug mode on a single protein"""
        print("🔍 DEBUG MODE: Testing single protein")

        splits = self.load_protein_splits()

        if not splits[split]:
            print(f"❌ No proteins found in {split} split")
            return

        # Test first protein
        chain = splits[split][0]
        print(f"\n🧪 Testing {split}/{chain}")

        dirs = self.setup_protein_directories(split, chain)

        # Check if already processed
        if self.check_existing_blocks(dirs['protein_blocks_dir']):
            print(f"      ✅ Evoformer blocks already exist, skipping")
            return

        # Run test
        success = self.test_single_protein(chain, dirs)

        if success:
            print(f"✅ Success! The issue has been resolved.")
        else:
            print(f"❌ Still failing. Check the error output above for details.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Evoformer inputs for protein splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode (test single protein)')

    args = parser.parse_args()

    generator = EvoformerInputGenerator(
        data_dir=args.data_dir,
        config_preset=args.config_preset,
        device=args.device,
        template_mmcif_dir=args.template_mmcif_dir,
        debug=args.debug
    )

    if not generator.check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues above.")
        sys.exit(1)

    if args.debug:
        # Run debug mode
        generator.run_debug_single()
    else:
        print("Use --debug flag to test a single protein with full error output")
        print("Example: python generate_evoformer_inputs.py --debug")


if __name__ == '__main__':
    main()