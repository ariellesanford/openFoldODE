#!/usr/bin/env python3
"""
Multi-Method Structure Prediction Script (Cleaned Version)
Generates structure predictions using 4 different methods for a given PDB_ID
Metrics collection handled by comprehensive_metrics module
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import shutil


class StructurePredictionRunner:
    def __init__(self, pdb_id: str):
        self.pdb_id = pdb_id

        # Base directories
        self.script_dir = Path(__file__).parent
        self.root_dir = Path("/home/visitor/PycharmProjects/openFold")
        self.data_dir = Path("/media/visitor/Extreme SSD/data")

        # Configuration
        self.neural_ode_predictions_base_dir = self.data_dir / "post_evoformer_predictions"
        self.use_cuda = True
        self.config_preset = "model_1_ptm"
        self.model_device = "cuda:0"

        # Optional settings
        self.skip_relaxation = False
        self.cif_output = False
        self.save_outputs = False
        self.long_sequence_inference = False
        self.use_deepspeed_attention = False

        # Common paths
        self.fasta_dir = self.data_dir / "fasta_data" / pdb_id
        self.template_mmcif_dir = self.data_dir / "template_data" / "pdb70_mmcif" / "mmcif_files"
        self.precomputed_alignments = self.data_dir / "alignments"

        # Method output directories
        self.predictions_base_dir = self.data_dir / "structure_predictions" / pdb_id
        self.openfold_decon_output_dir = self.predictions_base_dir / "openfold_deconstructed"
        self.openfold_full_output_dir = self.predictions_base_dir / "openfold_0recycles"
        self.half_evoformer_output_dir = self.predictions_base_dir / "half_evoformer"

        # Ground truth data (48th block)
        blocks_dir = blocks_dir = self._find_blocks_directory(pdb_id)
        print(blocks_dir)
        self.ground_truth_msa_path = blocks_dir / "m_block_48.pt"
        self.ground_truth_pair_path = blocks_dir / "z_block_48.pt"

        # Method input files
        self.openfold_decon_msa_path = blocks_dir / "m_block_48.pt"
        self.openfold_decon_pair_path = blocks_dir / "z_block_48.pt"
        self.half_evoformer_msa_path = blocks_dir / "m_block_24.pt"
        self.half_evoformer_pair_path = blocks_dir / "z_block_24.pt"

        # Results tracking
        self.successful_methods = []
        self.failed_methods = []

        self._setup_device()
        self.python_path = self._get_python_path()

    def _find_blocks_directory(self, pdb_id: str) -> Path:
        """Find blocks directory, checking both complete_blocks and incomplete_blocks"""
        possible_dirs = [
            self.data_dir / "complete_blocks" / f"{pdb_id}_evoformer_blocks" / "recycle_0",
            self.data_dir / "endpoint_blocks" / f"{pdb_id}_evoformer_blocks" / "recycle_0",
            self.data_dir / "incomplete_blocks" / f"{pdb_id}_evoformer_blocks" / "recycle_0"
        ]
        print(possible_dirs)
        for blocks_dir in possible_dirs:
            if blocks_dir.exists():
                # Check if required files exist
                if (blocks_dir / "m_block_48.pt").exists() and (blocks_dir / "z_block_48.pt").exists():
                    print(f"✅ Found blocks directory: {blocks_dir}")
                    return blocks_dir

        # If neither directory has the required files, default to complete_blocks
        # (the original behavior - will fail later with appropriate error messages)
        default_dir = self.data_dir / "complete_blocks" / f"{pdb_id}_evoformer_blocks" / "recycle_0"
        print(f"⚠️  No valid blocks directory found, defaulting to: {default_dir}")
        return default_dir

    def _setup_device(self):
        """Set device based on CUDA availability"""
        if not self.use_cuda or not shutil.which("nvidia-smi"):
            self.model_device = "cpu"
        else:
            try:
                subprocess.run(["nvidia-smi"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                self.model_device = "cpu"

    def _get_python_path(self) -> str:
        """Get appropriate Python interpreter path"""
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            return str(Path(conda_prefix) / "bin" / "python")
        return sys.executable

    def discover_neural_ode_predictions(self) -> List[str]:
        """Discover available Neural ODE predictions for the protein"""
        predictions = []

        if not self.neural_ode_predictions_base_dir.exists():
            return predictions

        for pred_dir in self.neural_ode_predictions_base_dir.glob("predictions_*"):
            if pred_dir.is_dir():
                protein_pred_dir = pred_dir / self.pdb_id
                if protein_pred_dir.is_dir():
                    msa_file = protein_pred_dir / "msa_representation.pt"
                    pair_file = protein_pred_dir / "pair_representation.pt"

                    if msa_file.exists() and pair_file.exists():
                        predictions.append(pred_dir.name)

        return predictions

    def output_exists(self, output_dir: Path) -> bool:
        """Check if output already exists (both relaxed and unrelaxed)"""
        relaxed_file = output_dir / f"{self.pdb_id}_model_1_ptm_relaxed.pdb"
        unrelaxed_file = output_dir / f"{self.pdb_id}_model_1_ptm_unrelaxed.pdb"

        return relaxed_file.exists() or unrelaxed_file.exists()

    def check_requirements(self, method_name: str, *required_files) -> bool:
        """Check if required files exist for a method"""
        for file_path in required_files:
            path = Path(file_path)
            if not (path.exists() and (path.is_file() or path.is_dir())):
                print(f"❌ Missing: {file_path}")
                return False
        return True

    def build_optional_args(self) -> List[str]:
        """Build list of optional arguments"""
        args = []

        if self.precomputed_alignments.exists():
            args.extend(["--use_precomputed_alignments", str(self.precomputed_alignments)])

        if self.skip_relaxation:
            args.append("--skip_relaxation")

        if self.cif_output:
            args.append("--cif_output")

        if self.save_outputs:
            args.append("--save_outputs")

        if self.long_sequence_inference:
            args.append("--long_sequence_inference")

        if self.use_deepspeed_attention:
            args.append("--use_deepspeed_evoformer_attention")

        return args

    def run_neural_ode_structure_module(self, pred_name: str) -> bool:
        """Run Neural ODE structure module for a specific prediction"""
        predictions_dir = self.neural_ode_predictions_base_dir / pred_name / self.pdb_id
        msa_path = predictions_dir / "msa_representation.pt"
        pair_path = predictions_dir / "pair_representation.pt"
        output_dir = self.predictions_base_dir / "neuralODE" / pred_name

        print(f"🧬 Running Neural ODE Structure Module ({pred_name})...")
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_path, "run_structure_module.py",
            str(self.fasta_dir), str(self.template_mmcif_dir),
            "--msa_path", str(msa_path),
            "--pair_path", str(pair_path),
            "--output_dir", str(output_dir),
            "--model_device", self.model_device,
            "--config_preset", self.config_preset
        ]
        cmd.extend(self.build_optional_args())

        try:
            subprocess.run(cmd, cwd=self.root_dir / "save_intermediates", check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def run_structure_module(self, method_name: str, msa_path: Path, pair_path: Path, output_dir: Path) -> bool:
        """Run structure module for a given method"""
        print(f"🧬 Running {method_name}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_path, "run_structure_module.py",
            str(self.fasta_dir), str(self.template_mmcif_dir),
            "--msa_path", str(msa_path),
            "--pair_path", str(pair_path),
            "--output_dir", str(output_dir),
            "--model_device", self.model_device,
            "--config_preset", self.config_preset,
        ]
        cmd.extend(self.build_optional_args())

        try:
            subprocess.run(cmd, cwd=self.root_dir / "save_intermediates", check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def run_full_openfold(self, output_dir: Path) -> bool:
        """Run full OpenFold"""
        print("🧬 Running Full OpenFold...")
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_path, "run_pretrained_openfold.py",
            str(self.fasta_dir), str(self.template_mmcif_dir),
            "--output_dir", str(output_dir),
            "--config_preset", self.config_preset,
            "--model_device", self.model_device,
            "--save_intermediates"
        ]
        cmd.extend(self.build_optional_args())

        try:
            subprocess.run(cmd, cwd=self.root_dir / "save_intermediates", check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def run_method_1_neural_ode(self):
        """METHOD 1: Neural ODE Predictions (Auto-discover)"""
        print("\n🎯 METHOD 1: Neural ODE Predictions (Auto-discovering...)")
        print("=========================================")

        neural_ode_predictions = self.discover_neural_ode_predictions()

        if not neural_ode_predictions:
            print(f"⏭️  No Neural ODE predictions found in {self.neural_ode_predictions_base_dir}")
            self.failed_methods.append("Neural ODE (no predictions found)")
            return

        print(f"✅ Found {len(neural_ode_predictions)} Neural ODE predictions:")
        for pred in neural_ode_predictions:
            print(f"   - {pred}")

        neural_ode_processed = 0
        neural_ode_skipped = 0
        neural_ode_failed = 0

        for pred_name in neural_ode_predictions:
            print(f"\n🔄 Processing Neural ODE prediction: {pred_name}")

            output_dir = self.predictions_base_dir / "neuralODE" / pred_name

            # Check if output already exists
            if self.output_exists(output_dir):
                print("   ✅ Structure prediction already exists - skipping")
                neural_ode_skipped += 1
                continue

            # Check requirements
            predictions_dir = self.neural_ode_predictions_base_dir / pred_name / self.pdb_id
            msa_path = predictions_dir / "msa_representation.pt"
            pair_path = predictions_dir / "pair_representation.pt"

            if not self.check_requirements(f"Neural ODE ({pred_name})", msa_path, pair_path,
                                           self.fasta_dir, self.template_mmcif_dir):
                print("   ❌ Missing requirements - skipping")
                neural_ode_failed += 1
                continue

            # Run structure prediction
            if self.run_neural_ode_structure_module(pred_name):
                print(f"   ✅ Successfully processed {pred_name}")
                neural_ode_processed += 1
            else:
                print(f"   ❌ Failed to process {pred_name}")
                neural_ode_failed += 1

        print(f"\n📊 Neural ODE Summary:")
        print(f"   Processed: {neural_ode_processed}")
        print(f"   Skipped (already exists): {neural_ode_skipped}")
        print(f"   Failed: {neural_ode_failed}")

        if neural_ode_processed > 0 or neural_ode_skipped > 0:
            self.successful_methods.append(f"Neural ODE ({neural_ode_processed + neural_ode_skipped} models)")
        if neural_ode_failed > 0:
            self.failed_methods.append(f"Neural ODE ({neural_ode_failed} failed)")

    def run_method_2_openfold_deconstructed(self):
        """METHOD 2: OpenFold Deconstructed (48th Evoformer Block)"""
        print("\n🎯 METHOD 2: OpenFold Deconstructed (48th Evoformer Block)")
        print("=========================================")

        if self.output_exists(self.openfold_decon_output_dir):
            print("✅ OpenFold Deconstructed output already exists - skipping")
            self.successful_methods.append("OpenFold Deconstructed (skipped)")
        elif self.check_requirements("OpenFold Deconstructed",
                                     self.openfold_decon_msa_path, self.openfold_decon_pair_path,
                                     self.fasta_dir, self.template_mmcif_dir):
            if self.run_structure_module("OpenFold Deconstructed",
                                         self.openfold_decon_msa_path,
                                         self.openfold_decon_pair_path,
                                         self.openfold_decon_output_dir):
                self.successful_methods.append("OpenFold Deconstructed")
            else:
                self.failed_methods.append("OpenFold Deconstructed")
        else:
            print("⏭️  Skipping OpenFold Deconstructed - missing requirements")
            self.failed_methods.append("OpenFold Deconstructed (missing files)")

    def run_method_3_full_openfold(self):
        """METHOD 3: Full OpenFold"""
        print("\n🎯 METHOD 3: Full OpenFold")
        print("=========================================")

        full_openfold_predictions_dir = self.openfold_full_output_dir / "predictions"

        if self.output_exists(full_openfold_predictions_dir):
            print("✅ Full OpenFold output already exists - skipping")
            self.successful_methods.append("Full OpenFold (skipped)")
        elif self.check_requirements("Full OpenFold", self.fasta_dir, self.template_mmcif_dir):
            if self.run_full_openfold(self.openfold_full_output_dir):
                self.successful_methods.append("Full OpenFold")
            else:
                self.failed_methods.append("Full OpenFold")
        else:
            print("⏭️  Skipping Full OpenFold - missing requirements")
            self.failed_methods.append("Full OpenFold (missing files)")

    def run_method_4_half_evoformer(self):
        """METHOD 4: Half Evoformer (Block 24 Only)"""
        print("\n🎯 METHOD 4: Half Evoformer (Block 24 Only)")
        print("=========================================")

        if self.output_exists(self.half_evoformer_output_dir):
            print("✅ Half Evoformer output already exists - skipping")
            self.successful_methods.append("Half Evoformer (skipped)")
        elif self.check_requirements("Half Evoformer",
                                     self.half_evoformer_msa_path, self.half_evoformer_pair_path,
                                     self.fasta_dir, self.template_mmcif_dir):
            if self.run_structure_module("Half Evoformer",
                                         self.half_evoformer_msa_path,
                                         self.half_evoformer_pair_path,
                                         self.half_evoformer_output_dir):
                self.successful_methods.append("Half Evoformer")
            else:
                self.failed_methods.append("Half Evoformer")
        else:
            print("⏭️  Skipping Half Evoformer - missing requirements")
            self.failed_methods.append("Half Evoformer (missing files)")

    def print_summary(self):
        """Print final summary"""
        print("\n=========================================")
        print(f"FINAL SUMMARY FOR {self.pdb_id}")
        print("=========================================")

        print(f"✅ Successful methods ({len(self.successful_methods)}):")
        if not self.successful_methods:
            print("   None")
        else:
            for method in self.successful_methods:
                print(f"   - {method}")

        print(f"\n❌ Failed methods ({len(self.failed_methods)}):")
        if not self.failed_methods:
            print("   None")
        else:
            for method in self.failed_methods:
                print(f"   - {method}")

        print("\n🎯 Structure prediction pipeline completed!")
        print("📊 Run comprehensive metrics analysis to compare methods:")
        print(f"   python comprehensive_metrics.py --pdb_id {self.pdb_id}")

    def run_all_methods(self):
        """Run all structure prediction methods"""
        print("=========================================")
        print("MULTI-METHOD STRUCTURE PREDICTION")
        print("=========================================")
        print(f"Protein ID: {self.pdb_id}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Model Device: {self.model_device}")
        print("=========================================")

        # Validate inputs
        if not self.pdb_id or self.pdb_id == "CHANGE_ME":
            print("❌ Error: Please set PDB_ID")
            return False

        if not self.fasta_dir.exists():
            print(f"❌ Error: FASTA directory not found: {self.fasta_dir}")
            return False

        # Run all methods
        self.run_method_1_neural_ode()
        self.run_method_2_openfold_deconstructed()
        self.run_method_3_full_openfold()
        self.run_method_4_half_evoformer()

        # Print summary
        self.print_summary()

        return len(self.successful_methods) > 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Multi-method structure prediction')
    parser.add_argument('--pdb_id', type=str, default="1ujs_A",
                        help='Protein ID to process')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--skip_relaxation', action='store_true',
                        help='Skip amber relaxation')
    parser.add_argument('--cif_output', action='store_true',
                        help='Output in CIF format')
    parser.add_argument('--run_metrics', default='True',
                        help='Run comprehensive metrics analysis after predictions')

    args = parser.parse_args()

    # Create runner
    runner = StructurePredictionRunner(args.pdb_id)

    # Apply command line options
    if args.use_cpu:
        runner.use_cuda = False
        runner.model_device = "cpu"

    runner.skip_relaxation = args.skip_relaxation
    runner.cif_output = args.cif_output

    # Run all methods
    success = runner.run_all_methods()

    # Optionally run metrics analysis
    if args.run_metrics and success:
        print("\n" + "=" * 50)
        print("RUNNING COMPREHENSIVE METRICS ANALYSIS")
        print("=" * 50)

        try:
            # Import and run comprehensive metrics
            from comprehensive_metrics import integrate_comprehensive_metrics

            # Add metrics collection to the runner
            integrate_comprehensive_metrics(runner)

            # Now collect metrics for all successful methods
            runner.method_metrics = {}

            # Collect for Neural ODE methods
            neural_ode_predictions = runner.discover_neural_ode_predictions()
            for pred_name in neural_ode_predictions:
                output_dir = runner.predictions_base_dir / "neuralODE" / pred_name
                if runner.output_exists(output_dir):
                    predictions_dir = runner.neural_ode_predictions_base_dir / pred_name / runner.pdb_id
                    msa_path = predictions_dir / "msa_representation.pt"
                    pair_path = predictions_dir / "pair_representation.pt"

                    metrics = runner.collect_method_metrics(
                        f"Neural ODE ({pred_name})", output_dir, msa_path, pair_path
                    )
                    runner.method_metrics[f"neural_ode_{pred_name}"] = metrics

            # Collect for other methods
            methods_to_check = [
                ("OpenFold Deconstructed", runner.openfold_decon_output_dir,
                 runner.openfold_decon_msa_path, runner.openfold_decon_pair_path),
                ("Full OpenFold", runner.openfold_full_output_dir / "predictions", None, None),
                ("Half Evoformer", runner.half_evoformer_output_dir,
                 runner.half_evoformer_msa_path, runner.half_evoformer_pair_path)
            ]

            for method_name, output_dir, msa_path, pair_path in methods_to_check:
                if runner.output_exists(output_dir):
                    metrics = runner.collect_method_metrics(method_name, output_dir, msa_path, pair_path)
                    method_key = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    runner.method_metrics[method_key] = metrics

            # Generate comprehensive summary
            runner.save_metrics_summary()

        except ImportError as e:
            print(f"⚠️  Could not run metrics analysis: {e}")
            print(f"   Please run: python comprehensive_metrics.py --pdb_id {args.pdb_id}")

    return 0 if success else 1


if __name__ == "__main__":
    print("🧬 Multi-Method Structure Prediction Script")
    print("Generates predictions using Neural ODE, OpenFold Deconstructed, Full OpenFold, and Half Evoformer")
    print("")
    print("Usage:")
    print("  python generate_all_structure_predictions.py                              # Default: 1fv5_A")
    print("  python generate_all_structure_predictions.py --pdb_id 2abc_A              # Custom protein")
    print("  python generate_all_structure_predictions.py --pdb_id 1fv5_A --run_metrics # With metrics")
    print("  python generate_all_structure_predictions.py --use_cpu                     # Force CPU")
    print("")
    print("For comprehensive metrics analysis:")
    print("  python comprehensive_metrics.py --pdb_id 1fv5_A")
    print("")

    sys.exit(main())