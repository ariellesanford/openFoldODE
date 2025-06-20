#!/usr/bin/env python3
"""
Comprehensive metrics collection for protein structure prediction
Includes structural accuracy, representation losses, and OpenFold-style auxiliary losses
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import tempfile
import subprocess
import os
import sys
from datetime import datetime

# Add save_intermediates to path to import OpenFold components
save_intermediates_path = Path(__file__).parent.parent / "save_intermediates"
if not save_intermediates_path.exists():
    save_intermediates_path = Path(__file__).parent.parent.parent / "save_intermediates"
if save_intermediates_path.exists():
    sys.path.insert(0, str(save_intermediates_path))

try:
    from openfold.model.heads import MaskedMSAHead, DistogramHead
    from openfold.utils.loss import masked_msa_loss, distogram_loss
    from openfold.config import model_config
    from openfold.utils.script_utils import load_models_from_command_line
    from openfold.data import data_pipeline, feature_pipeline
    from openfold.utils.tensor_utils import tensor_tree_map
    from openfold.np.protein import from_prediction, to_pdb
    from run_structure_module import generate_single_from_msa
    from run_pretrained_openfold import parse_fasta

    OPENFOLD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import OpenFold components: {e}")
    OPENFOLD_AVAILABLE = False

OPENFOLD_AVAILABLE = False  #I DONT WANT TO GENERATE AUXILLARY METRICS

class ComprehensiveMetricsCollector:
    """Complete metrics collection for structure prediction accuracy"""

    def __init__(self, pdb_id: str, reference_structure_path: Path = None,
                 config_preset: str = "model_1_ptm"):
        self.pdb_id = pdb_id
        self.config_preset = config_preset
        self.reference_coords = None
        self.reference_structure_path = None
        self.has_reference = False
        self.model = None  # Cache the loaded model
        self.device = 'cpu'  # Default device

        if reference_structure_path and reference_structure_path.exists():
            self.has_reference = self._load_reference_structure(reference_structure_path)
        else:
            self._try_auto_find_reference()

        self._init_auxiliary_heads()

    def set_device(self, device: str):
        """Set the device for model computations"""
        self.device = device
        # Reset model to force reload on new device
        self.model = None

    def _load_model(self):
        """Load the OpenFold model for generate_single_from_msa"""
        # if not OPENFOLD_AVAILABLE or self.model is not None:
        #     return

        try:
            config = model_config(self.config_preset)

            # Use same parameter handling as run_structure_module.py
            openfold_checkpoint_path = None
            jax_param_path = None

            # Set up default parameter path if none provided (like run_structure_module.py)
            if jax_param_path is None and openfold_checkpoint_path is None:
                jax_param_path = os.path.join(
                    os.pardir, "save_intermediates", "openfold", "resources", "params",
                    f"params_{self.config_preset}.npz"
                )

            # Use current directory as output_dir
            output_dir = "."

            model_generator = load_models_from_command_line(
                config, self.device, openfold_checkpoint_path, jax_param_path, output_dir
            )
            self.model, _ = next(model_generator)
            print(f"✅ Loaded OpenFold model for single representation generation")

        except Exception as e:
            print(f"⚠️  Could not load OpenFold model: {e}")
            print(f"   Falling back to first MSA row for single representation")
            self.model = None

    def _load_reference_structure(self, ref_path: Path) -> bool:
        """Load reference structure coordinates"""
        coords = self._parse_pdb_coordinates(ref_path)
        if coords is not None and len(coords) > 0:
            self.reference_coords = coords
            self.reference_structure_path = str(ref_path)
            print(f"✅ Loaded reference structure: {ref_path}")
            return True
        return False

    def _try_auto_find_reference(self):
        """Find reference structure in the specific openfold_deconstructed directory"""
        base_path = Path(f"/media/visitor/Extreme SSD/data/structure_predictions/{self.pdb_id}/openfold_deconstructed")

        # Look for both relaxed and unrelaxed PDB files
        possible_files = [
            base_path / f"{self.pdb_id}_model_1_ptm_relaxed.pdb",
            base_path / f"{self.pdb_id}_model_1_ptm_unrelaxed.pdb"
        ]

        for ref_path in possible_files:
            if ref_path.exists():
                self.has_reference = self._load_reference_structure(ref_path)
                if self.has_reference:
                    break

        if not self.has_reference:
            print(f"⚠️  No reference structure found for {self.pdb_id} in openfold_deconstructed")

    def _init_auxiliary_heads(self):
        """Initialize OpenFold auxiliary heads for computing losses"""
        if not OPENFOLD_AVAILABLE:
            self.masked_msa_head = None
            self.distogram_head = None
            return

        config = model_config(self.config_preset)
        self.masked_msa_head = MaskedMSAHead(**config.model.heads.masked_msa)
        self.distogram_head = DistogramHead(**config.model.heads.distogram)
        self.masked_msa_head.eval()
        self.distogram_head.eval()
        print(f"✅ Initialized OpenFold auxiliary heads")

    def _parse_pdb_coordinates(self, pdb_path: Path) -> Optional[np.ndarray]:
        """Parse CA coordinates from PDB file"""
        coords = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[13:16].strip() == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        return np.array(coords) if coords else None

    def load_processed_feature_dict_for_protein(self, fasta_dir: str, pdb_id: str,
                                                precomputed_alignments: str = None) -> Optional[Dict]:
        """Load processed feature dict using same logic as run_structure_module.py"""
        if not OPENFOLD_AVAILABLE:
            return None

        fasta_dir_path = Path(fasta_dir)

        # Find FASTA file
        fasta_files = [f for f in fasta_dir_path.glob("*.fasta")] + [f for f in fasta_dir_path.glob("*.fa")]
        if not fasta_files:
            print(f"⚠️  No FASTA files found in {fasta_dir}")
            return None

        fasta_path = fasta_files[0]

        # Read and parse FASTA
        with open(fasta_path, 'r') as f:
            fasta_data = f.read()

        tags, seqs = parse_fasta(fasta_data)
        tag = tags[0]
        seq = seqs[0]

        # Create temporary FASTA file
        tmp_fasta_path = fasta_dir_path / f"tmp_{os.getpid()}.fasta"
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        try:
            # Set up data pipeline exactly like run_structure_module.py
            config = model_config(self.config_preset)
            data_processor = data_pipeline.DataPipeline(template_featurizer=None)
            feature_processor = feature_pipeline.FeaturePipeline(config.data)

            # Process features exactly like run_structure_module.py
            if precomputed_alignments:
                local_alignment_dir = os.path.join(precomputed_alignments, tag)
                if os.path.exists(local_alignment_dir):
                    feature_dict = data_processor.process_fasta(
                        fasta_path=str(tmp_fasta_path),
                        alignment_dir=local_alignment_dir,
                    )
                else:
                    print(f"⚠️  No precomputed alignments found at {local_alignment_dir}")
                    return None
            else:
                print(f"⚠️  No precomputed alignments provided")
                return None

            # Process features exactly like run_structure_module.py
            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict'
            )

            # Convert to tensors
            processed_feature_dict = tensor_tree_map(
                lambda x: torch.tensor(x),
                processed_feature_dict
            )

            return processed_feature_dict

        finally:
            # Clean up temp file
            if tmp_fasta_path.exists():
                tmp_fasta_path.unlink()

    def extract_comprehensive_plddt(self, pdb_path: Path) -> Dict:
        """Extract comprehensive pLDDT metrics from PDB file"""
        plddt_values = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[13:16].strip() == 'CA':
                    plddt = float(line[60:66])
                    plddt_values.append(plddt)

        if not plddt_values:
            return self._empty_plddt_dict()

        plddt_array = np.array(plddt_values)
        return {
            "avg_plddt": float(np.mean(plddt_array)),
            "median_plddt": float(np.median(plddt_array)),
            "min_plddt": float(np.min(plddt_array)),
            "max_plddt": float(np.max(plddt_array)),
            "std_plddt": float(np.std(plddt_array)),
            "num_residues": len(plddt_array),
            "very_confident_percent": float(np.mean(plddt_array >= 90) * 100),
            "confident_percent": float(np.mean(plddt_array >= 70) * 100),
            "low_confidence_percent": float(np.mean(plddt_array < 50) * 100),
            "plddt_distribution": {
                "very_high": int(np.sum(plddt_array >= 90)),
                "high": int(np.sum((plddt_array >= 70) & (plddt_array < 90))),
                "medium": int(np.sum((plddt_array >= 50) & (plddt_array < 70))),
                "low": int(np.sum(plddt_array < 50))
            }
        }

    def _empty_plddt_dict(self) -> Dict:
        """Return empty pLDDT dictionary when no structure available"""
        return {
            "avg_plddt": None, "median_plddt": None, "min_plddt": None,
            "max_plddt": None, "std_plddt": None, "num_residues": 0,
            "very_confident_percent": 0.0, "confident_percent": 0.0,
            "low_confidence_percent": 0.0,
            "plddt_distribution": {"very_high": 0, "high": 0, "medium": 0, "low": 0}
        }

    def calculate_rmsd(self, pred_pdb_path: Path) -> Optional[float]:
        """Calculate RMSD between predicted and reference structures"""
        if not self.has_reference:
            return None

        pred_coords = self._parse_pdb_coordinates(pred_pdb_path)
        if pred_coords is None:
            return None

        min_len = min(len(self.reference_coords), len(pred_coords))
        ref_subset = self.reference_coords[:min_len]
        pred_subset = pred_coords[:min_len]

        diff = ref_subset - pred_subset
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    def calculate_gdt_ts(self, pred_pdb_path: Path) -> Optional[float]:
        """Calculate GDT-TS score using simple distance-based approach"""
        if not self.has_reference:
            return None

        pred_coords = self._parse_pdb_coordinates(pred_pdb_path)
        if pred_coords is None:
            return None

        min_len = min(len(self.reference_coords), len(pred_coords))
        ref_subset = self.reference_coords[:min_len]
        pred_subset = pred_coords[:min_len]

        distances = np.sqrt(np.sum((ref_subset - pred_subset) ** 2, axis=1))
        thresholds = [1.0, 2.0, 4.0, 8.0]

        gdt_ts = 0.0
        for threshold in thresholds:
            fraction_under_threshold = np.mean(distances <= threshold)
            gdt_ts += fraction_under_threshold

        return float(gdt_ts / 4.0)

    def calculate_tm_score_approximate(self, pred_pdb_path: Path) -> Optional[float]:
        """Approximate TM-score calculation"""
        if not self.has_reference:
            return None

        pred_coords = self._parse_pdb_coordinates(pred_pdb_path)
        if pred_coords is None:
            return None

        min_len = min(len(self.reference_coords), len(pred_coords))
        ref_subset = self.reference_coords[:min_len]
        pred_subset = pred_coords[:min_len]

        distances = np.sqrt(np.sum((ref_subset - pred_subset) ** 2, axis=1))
        L = len(ref_subset)
        d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8 if L > 21 else 0.5

        tm_score = np.mean(1.0 / (1.0 + (distances / d0) ** 2))
        return float(tm_score)

    def compute_detailed_representation_loss(self, msa_path: Path, pair_path: Path,
                                             ground_truth_msa_path: Path, ground_truth_pair_path: Path) -> Dict:
        """Compute detailed representation losses"""
        # Load tensors
        msa_pred = torch.load(msa_path, map_location='cpu')
        pair_pred = torch.load(pair_path, map_location='cpu')
        msa_gt = torch.load(ground_truth_msa_path, map_location='cpu')
        pair_gt = torch.load(ground_truth_pair_path, map_location='cpu')

        # Remove batch dimensions if present
        while len(msa_pred.shape) > 3:
            msa_pred = msa_pred.squeeze(0)
        while len(pair_pred.shape) > 3:
            pair_pred = pair_pred.squeeze(0)
        while len(msa_gt.shape) > 3:
            msa_gt = msa_gt.squeeze(0)
        while len(pair_gt.shape) > 3:
            pair_gt = pair_gt.squeeze(0)

        # Align dimensions
        min_msa_seq = min(msa_pred.shape[0], msa_gt.shape[0])
        min_msa_res = min(msa_pred.shape[1], msa_gt.shape[1])
        min_msa_feat = min(msa_pred.shape[2], msa_gt.shape[2])

        msa_pred_aligned = msa_pred[:min_msa_seq, :min_msa_res, :min_msa_feat]
        msa_gt_aligned = msa_gt[:min_msa_seq, :min_msa_res, :min_msa_feat]

        min_pair_res = min(pair_pred.shape[0], pair_gt.shape[0])
        min_pair_res2 = min(pair_pred.shape[1], pair_gt.shape[1])
        min_pair_feat = min(pair_pred.shape[2], pair_gt.shape[2])

        pair_pred_aligned = pair_pred[:min_pair_res, :min_pair_res2, :min_pair_feat]
        pair_gt_aligned = pair_gt[:min_pair_res, :min_pair_res2, :min_pair_feat]

        # Compute losses
        msa_mse = F.mse_loss(msa_pred_aligned, msa_gt_aligned)
        pair_mse = F.mse_loss(pair_pred_aligned, pair_gt_aligned)

        # Generate single representation using the imported function with proper model
        self._load_model()
        if self.model is not None:
            single_pred = generate_single_from_msa(msa_pred.unsqueeze(0), self.model, self.device).squeeze(0)
            single_gt = generate_single_from_msa(msa_gt.unsqueeze(0), self.model, self.device).squeeze(0)
        else:
            # Fallback: use first MSA row as single representation
            single_pred = msa_pred_aligned[0, :, :]
            single_gt = msa_gt_aligned[0, :, :]

        min_single_res = min(single_pred.shape[0], single_gt.shape[0])
        min_single_feat = min(single_pred.shape[1], single_gt.shape[1])

        single_pred_aligned = single_pred[:min_single_res, :min_single_feat]
        single_gt_aligned = single_gt[:min_single_res, :min_single_feat]

        single_mse = F.mse_loss(single_pred_aligned, single_gt_aligned)

        # Compute cosine similarities
        msa_flat_pred = msa_pred_aligned.flatten()
        msa_flat_gt = msa_gt_aligned.flatten()
        msa_cosine_sim = F.cosine_similarity(msa_flat_pred, msa_flat_gt, dim=0)

        pair_flat_pred = pair_pred_aligned.flatten()
        pair_flat_gt = pair_gt_aligned.flatten()
        pair_cosine_sim = F.cosine_similarity(pair_flat_pred, pair_flat_gt, dim=0)

        # Adaptive loss (MSE + cosine)
        total_adaptive_loss = msa_mse + pair_mse + single_mse + (1 - msa_cosine_sim) + (1 - pair_cosine_sim)

        return {
            "msa_mse_loss": float(msa_mse),
            "pair_mse_loss": float(pair_mse),
            "single_mse_loss": float(single_mse),
            "msa_cosine_similarity": float(msa_cosine_sim),
            "pair_cosine_similarity": float(pair_cosine_sim),
            "total_adaptive_loss": float(total_adaptive_loss),
            "msa_dimensions": list(msa_pred_aligned.shape),
            "pair_dimensions": list(pair_pred_aligned.shape),
            "single_dimensions": list(single_pred_aligned.shape)
        }

    def compute_auxiliary_losses(self, msa_path: Path, pair_path: Path,
                                 ground_truth_msa_path: Path, ground_truth_pair_path: Path,
                                 processed_feature_dict: Optional[Dict] = None) -> Dict:
        """Compute OpenFold auxiliary losses using actual OpenFold components"""
        if not OPENFOLD_AVAILABLE or self.masked_msa_head is None:
            return {"masked_msa_loss": None, "distogram_loss": None}

        # Load representations
        msa_pred = torch.load(msa_path, map_location='cpu')
        pair_pred = torch.load(pair_path, map_location='cpu')

        # Remove batch dimensions
        while len(msa_pred.shape) > 3:
            msa_pred = msa_pred.squeeze(0)
        while len(pair_pred.shape) > 3:
            pair_pred = pair_pred.squeeze(0)

        # Create batch dictionary for heads (add batch dimension)
        batch = {
            'msa': msa_pred.unsqueeze(0),
            'pair': pair_pred.unsqueeze(0),
        }

        with torch.no_grad():
            # Compute logits using OpenFold heads
            masked_msa_logits = self.masked_msa_head(batch['msa'])
            distogram_logits = self.distogram_head(batch['pair'])

            # === MASKED MSA LOSS ===
            if processed_feature_dict and 'aatype' in processed_feature_dict:
                # Get aatype and ensure it has the right shape
                aatype = processed_feature_dict['aatype']
                while len(aatype.shape) > 1:
                    aatype = aatype.squeeze()

                # Create true_msa by expanding to match MSA dimensions
                n_seq = msa_pred.shape[0]
                n_res = aatype.shape[0]
                true_msa = aatype.unsqueeze(0).expand(n_seq, n_res).unsqueeze(0)

                # Use DETERMINISTIC mask for consistent comparison across models
                # Set seed based on protein sequence for reproducibility
                torch.manual_seed(hash(str(aatype.tolist())) % (2 ** 31))
                bert_mask = torch.bernoulli(torch.full_like(true_msa.float(), 0.15)).bool()

                # Compute masked MSA loss
                msa_loss_value = masked_msa_loss(
                    logits=masked_msa_logits,
                    true_msa=true_msa.long(),
                    bert_mask=bert_mask,
                    num_classes=23
                )
            else:
                msa_loss_value = torch.tensor(0.0)

            # === DISTOGRAM LOSS ===
            # Use reference structure coordinates if available, otherwise skip
            if self.has_reference and self.reference_coords is not None:
                print(f"✅ Using reference structure for distogram loss")
                # Use actual reference coordinates
                ref_coords = torch.tensor(self.reference_coords, dtype=torch.float32)

                # Ensure correct shape and match sequence length
                seq_len = pair_pred.shape[0]
                if ref_coords.shape[0] != seq_len:
                    # Pad or truncate to match
                    if ref_coords.shape[0] < seq_len:
                        # Pad with zeros
                        padding = torch.zeros(seq_len - ref_coords.shape[0], 3)
                        ref_coords = torch.cat([ref_coords, padding], dim=0)
                    else:
                        # Truncate
                        ref_coords = ref_coords[:seq_len]

                pseudo_beta = ref_coords.unsqueeze(0)  # Add batch dim: [1, seq_len, 3]
                pseudo_beta_mask = torch.ones(1, seq_len).float()

                distogram_loss_value = distogram_loss(
                    logits=distogram_logits,
                    pseudo_beta=pseudo_beta,
                    pseudo_beta_mask=pseudo_beta_mask
                )
            else:
                # If no reference structure, still use deterministic "dummy" coordinates
                # Make them deterministic based on sequence length and protein ID
                print(f"⚠️  No reference structure available, using dummy coordinates for distogram loss")
                torch.manual_seed(hash(self.pdb_id) % (2 ** 31))
                seq_len = pair_pred.shape[0]
                pseudo_beta = torch.randn(1, seq_len, 3) * 10.0  # Scale for realistic distances
                pseudo_beta_mask = torch.ones(1, seq_len).float()

                distogram_loss_value = distogram_loss(
                    logits=distogram_logits,
                    pseudo_beta=pseudo_beta,
                    pseudo_beta_mask=pseudo_beta_mask
                )

        return {
            "masked_msa_loss": float(msa_loss_value),
            "distogram_loss": float(distogram_loss_value)
        }
    def collect_all_metrics(self, method_name: str, output_dir: Path,
                            msa_path: Path = None, pair_path: Path = None,
                            ground_truth_msa_path: Path = None,
                            ground_truth_pair_path: Path = None,
                            processed_feature_dict: Dict = None) -> Dict:
        """Collect all available metrics for a prediction method"""

        metrics = {
            "method": method_name,
            "pdb_id": self.pdb_id,
            "output_dir": str(output_dir),
            "has_reference_structure": self.has_reference,
            "reference_structure_path": self.reference_structure_path
        }

        # Find PDB file (prefer relaxed, fallback to unrelaxed)
        pdb_files = []
        for suffix in ["_relaxed.pdb", "_unrelaxed.pdb"]:
            pdb_path = output_dir / f"{self.pdb_id}_model_1_ptm{suffix}"
            if pdb_path.exists():
                pdb_files.append((pdb_path, suffix.replace("_", "").replace(".pdb", "")))

        if not pdb_files:
            metrics["structure_file"] = None
            metrics.update(self._empty_plddt_dict())
            metrics["structural_accuracy"] = None
        else:
            pdb_path, structure_type = pdb_files[0]
            metrics["structure_file"] = pdb_path.name
            metrics["structure_type"] = structure_type

            # pLDDT metrics
            plddt_metrics = self.extract_comprehensive_plddt(pdb_path)
            metrics.update(plddt_metrics)

            # Structural accuracy metrics
            if self.has_reference:
                structural_accuracy = {}
                rmsd = self.calculate_rmsd(pdb_path)
                if rmsd is not None:
                    structural_accuracy["rmsd_ca"] = rmsd
                gdt_ts = self.calculate_gdt_ts(pdb_path)
                if gdt_ts is not None:
                    structural_accuracy["gdt_ts"] = gdt_ts
                tm_score = self.calculate_tm_score_approximate(pdb_path)
                if tm_score is not None:
                    structural_accuracy["tm_score_approx"] = tm_score
                metrics["structural_accuracy"] = structural_accuracy
            else:
                metrics["structural_accuracy"] = None

        # Representation loss metrics
        if msa_path and pair_path and ground_truth_msa_path and ground_truth_pair_path:
            representation_metrics = self.compute_detailed_representation_loss(
                msa_path, pair_path, ground_truth_msa_path, ground_truth_pair_path
            )
            metrics.update(representation_metrics)

            # Auxiliary losses
            aux_metrics = self.compute_auxiliary_losses(
                msa_path, pair_path, ground_truth_msa_path, ground_truth_pair_path,
                processed_feature_dict
            )
            metrics.update(aux_metrics)

        return metrics

    def rank_methods_by_all_metrics(self, all_methods: Dict[str, Dict]) -> Dict[str, List[Tuple[str, float]]]:
        """Rank methods by various metrics"""
        rankings = {}

        # Structure quality ranking (by average pLDDT)
        plddt_methods = [(name, metrics.get("avg_plddt"))
                         for name, metrics in all_methods.items()
                         if metrics.get("avg_plddt") is not None]
        if plddt_methods:
            rankings["by_avg_plddt"] = sorted(plddt_methods, key=lambda x: x[1], reverse=True)

        # Representation accuracy ranking
        loss_methods = [(name, metrics.get("total_adaptive_loss"))
                        for name, metrics in all_methods.items()
                        if metrics.get("total_adaptive_loss") is not None]
        if loss_methods:
            rankings["by_total_adaptive_loss"] = sorted(loss_methods, key=lambda x: x[1])

        msa_mse_methods = [(name, metrics.get("msa_mse_loss"))
                           for name, metrics in all_methods.items()
                           if metrics.get("msa_mse_loss") is not None]
        if msa_mse_methods:
            rankings["by_msa_mse_loss"] = sorted(msa_mse_methods, key=lambda x: x[1])

        pair_mse_methods = [(name, metrics.get("pair_mse_loss"))
                            for name, metrics in all_methods.items()
                            if metrics.get("pair_mse_loss") is not None]
        if pair_mse_methods:
            rankings["by_pair_mse_loss"] = sorted(pair_mse_methods, key=lambda x: x[1])

        single_mse_methods = [(name, metrics.get("single_mse_loss"))
                              for name, metrics in all_methods.items()
                              if metrics.get("single_mse_loss") is not None]
        if single_mse_methods:
            rankings["by_single_mse_loss"] = sorted(single_mse_methods, key=lambda x: x[1])


        # MSA similarity ranking
        msa_sim_methods = [(name, metrics.get("msa_cosine_similarity"))
                           for name, metrics in all_methods.items()
                           if metrics.get("msa_cosine_similarity") is not None]
        if msa_sim_methods:
            rankings["by_msa_cosine_similarity"] = sorted(msa_sim_methods, key=lambda x: x[1], reverse=True)

        # Auxiliary loss rankings
        if OPENFOLD_AVAILABLE:
            msa_loss_methods = [(name, metrics.get("masked_msa_loss"))
                                for name, metrics in all_methods.items()
                                if metrics.get("masked_msa_loss") is not None]
            if msa_loss_methods:
                rankings["by_masked_msa_loss"] = sorted(msa_loss_methods, key=lambda x: x[1])

            dist_loss_methods = [(name, metrics.get("distogram_loss"))
                                 for name, metrics in all_methods.items()
                                 if metrics.get("distogram_loss") is not None]
            if dist_loss_methods:
                rankings["by_distogram_loss"] = sorted(dist_loss_methods, key=lambda x: x[1])

        # Structural accuracy rankings (if reference available)
        if self.has_reference:
            rmsd_methods = [(name, metrics["structural_accuracy"].get("rmsd_ca"))
                            for name, metrics in all_methods.items()
                            if metrics.get("structural_accuracy") and metrics["structural_accuracy"].get(
                    "rmsd_ca") is not None]
            if rmsd_methods:
                rankings["by_rmsd"] = sorted(rmsd_methods, key=lambda x: x[1])

            gdt_methods = [(name, metrics["structural_accuracy"].get("gdt_ts"))
                           for name, metrics in all_methods.items()
                           if metrics.get("structural_accuracy") and metrics["structural_accuracy"].get(
                    "gdt_ts") is not None]
            if gdt_methods:
                rankings["by_gdt_ts"] = sorted(gdt_methods, key=lambda x: x[1], reverse=True)

            tm_methods = [(name, metrics["structural_accuracy"].get("tm_score_approx"))
                          for name, metrics in all_methods.items()
                          if metrics.get("structural_accuracy") and metrics["structural_accuracy"].get(
                    "tm_score_approx") is not None]
            if tm_methods:
                rankings["by_tm_score"] = sorted(tm_methods, key=lambda x: x[1], reverse=True)

        return rankings

    def print_comprehensive_summary(self, all_metrics: Dict[str, Dict], rankings: Dict[str, List[Tuple[str, float]]]):
        """Print comprehensive summary of all metrics"""
        print(f"\n🧬 COMPREHENSIVE METRICS SUMMARY for {self.pdb_id}")
        print("=" * 60)

        # Structure Quality Rankings
        if "by_avg_plddt" in rankings:
            print(f"\n🏆 STRUCTURE QUALITY RANKING (Average pLDDT):")
            for i, (method, plddt) in enumerate(rankings["by_avg_plddt"], 1):
                confidence = "🟢 Very High" if plddt >= 90 else "🟡 High" if plddt >= 70 else "🟠 Medium" if plddt >= 50 else "🔴 Low"
                print(f"  {i}. {method}: {plddt:.1f} pLDDT {confidence}")

        # Representation Accuracy Rankings
        if "by_total_adaptive_loss" in rankings:
            print(f"\n🎯 REPRESENTATION ACCURACY RANKING (Adaptive Loss):")
            for i, (method, loss) in enumerate(rankings["by_total_adaptive_loss"], 1):
                print(f"  {i}. {method}: {loss:.6f} adaptive loss")

        if "by_msa_cosine_similarity" in rankings:
            print(f"\n🧬 MSA REPRESENTATION SIMILARITY:")
            for i, (method, sim) in enumerate(rankings["by_msa_cosine_similarity"], 1):
                quality = "Excellent" if sim >= 0.95 else "Good" if sim >= 0.90 else "Fair" if sim >= 0.80 else "Poor"
                print(f"  {i}. {method}: {sim:.4f} cosine similarity ({quality})")

        # Auxiliary Loss Rankings
        if "by_masked_msa_loss" in rankings:
            print(f"\n🎭 MASKED MSA LOSS RANKING:")
            for i, (method, loss) in enumerate(rankings["by_masked_msa_loss"], 1):
                print(f"  {i}. {method}: {loss:.6f} masked MSA loss")

        if "by_distogram_loss" in rankings:
            print(f"\n📐 DISTOGRAM LOSS RANKING:")
            for i, (method, loss) in enumerate(rankings["by_distogram_loss"], 1):
                print(f"  {i}. {method}: {loss:.6f} distogram loss")

        # Structural Accuracy Rankings
        if self.has_reference:
            if "by_rmsd" in rankings:
                print(f"\n📏 STRUCTURAL RMSD RANKING (vs Reference):")
                for i, (method, rmsd) in enumerate(rankings["by_rmsd"], 1):
                    quality = "🟢 Excellent" if rmsd <= 2.0 else "🟡 Good" if rmsd <= 4.0 else "🟠 Fair" if rmsd <= 6.0 else "🔴 Poor"
                    print(f"  {i}. {method}: {rmsd:.2f} Å {quality}")

            if "by_gdt_ts" in rankings:
                print(f"\n🎲 GDT-TS RANKING (vs Reference):")
                for i, (method, gdt) in enumerate(rankings["by_gdt_ts"], 1):
                    quality = "🟢 Excellent" if gdt >= 0.8 else "🟡 Good" if gdt >= 0.6 else "🟠 Fair" if gdt >= 0.4 else "🔴 Poor"
                    print(f"  {i}. {method}: {gdt:.3f} {quality}")

            if "by_tm_score" in rankings:
                print(f"\n🔄 TM-SCORE RANKING (vs Reference):")
                for i, (method, tm) in enumerate(rankings["by_tm_score"], 1):
                    quality = "🟢 Same fold" if tm >= 0.5 else "🟠 Different fold"
                    quality += " (High sim)" if tm >= 0.8 else " (Med sim)" if tm >= 0.6 else " (Low sim)"
                    print(f"  {i}. {method}: {tm:.3f} {quality}")

        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Total methods analyzed: {len(all_metrics)}")
        print(f"   Methods with structure quality: {len(rankings.get('by_avg_plddt', []))}")
        print(f"   Methods with representation loss: {len(rankings.get('by_total_adaptive_loss', []))}")
        print(f"   Methods with auxiliary losses: {len(rankings.get('by_masked_msa_loss', []))}")
        if self.has_reference:
            print(f"   Methods with structural accuracy: {len(rankings.get('by_rmsd', []))}")
            print(f"   Reference structure: {self.reference_structure_path}")


def integrate_comprehensive_metrics(runner_instance):
    """Integrate comprehensive metrics into an existing runner instance"""

    runner_instance.comprehensive_metrics = ComprehensiveMetricsCollector(
        pdb_id=runner_instance.pdb_id,
        reference_structure_path=None
    )

    def enhanced_collect_method_metrics(method_name: str, output_dir: Path,
                                        msa_path: Path = None, pair_path: Path = None) -> Dict:
        processed_feature_dict = None
        if hasattr(runner_instance, 'fasta_dir') and runner_instance.fasta_dir:
            precomputed_alignments = getattr(runner_instance, 'precomputed_alignments', None)
            processed_feature_dict = runner_instance.comprehensive_metrics.load_processed_feature_dict_for_protein(
                runner_instance.fasta_dir,
                runner_instance.pdb_id,
                precomputed_alignments
            )

        return runner_instance.comprehensive_metrics.collect_all_metrics(
            method_name, output_dir, msa_path, pair_path,
            getattr(runner_instance, 'ground_truth_msa_path', None),
            getattr(runner_instance, 'ground_truth_pair_path', None),
            processed_feature_dict
        )

    runner_instance.collect_method_metrics = enhanced_collect_method_metrics

    def comprehensive_save_metrics_summary():
        if not hasattr(runner_instance, 'method_metrics') or not runner_instance.method_metrics:
            return

        summary_file = getattr(runner_instance, 'predictions_base_dir',
                               Path('.')) / "comprehensive_metrics_analysis.json"
        rankings = runner_instance.comprehensive_metrics.rank_methods_by_all_metrics(
            runner_instance.method_metrics
        )

        summary = {
            "protein_id": runner_instance.pdb_id,
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "comprehensive_structure_prediction_metrics",
            "has_reference_structure": runner_instance.comprehensive_metrics.has_reference,
            "reference_structure_path": runner_instance.comprehensive_metrics.reference_structure_path,
            "ground_truth_files": {
                "msa_block_48": str(getattr(runner_instance, 'ground_truth_msa_path', 'None')),
                "pair_block_48": str(getattr(runner_instance, 'ground_truth_pair_path', 'None'))
            },
            "method_metrics": runner_instance.method_metrics,
            "rankings": rankings
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n📊 Comprehensive metrics saved to: {summary_file}")
        runner_instance.comprehensive_metrics.print_comprehensive_summary(
            runner_instance.method_metrics, rankings
        )

    runner_instance.save_metrics_summary = comprehensive_save_metrics_summary


def create_comprehensive_metrics_collector(pdb_id: str) -> ComprehensiveMetricsCollector:
    """Create a standalone metrics collector without runner instance dependency"""
    return ComprehensiveMetricsCollector(pdb_id=pdb_id)


def analyze_method_predictions(pdb_id: str, methods_config: Dict[str, Dict]) -> Dict:
    """
    Standalone function to analyze multiple method predictions

    Args:
        pdb_id: Protein ID to analyze
        methods_config: Dict with method_name -> {output_dir, msa_path, pair_path, ground_truth_msa_path, ground_truth_pair_path}

    Returns:
        Dict with comprehensive analysis results
    """
    collector = ComprehensiveMetricsCollector(pdb_id=pdb_id)

    all_metrics = {}
    for method_name, config in methods_config.items():
        # Load processed feature dict if needed for auxiliary losses
        processed_feature_dict = None
        if config.get('fasta_dir') and config.get('precomputed_alignments'):
            processed_feature_dict = collector.load_processed_feature_dict_for_protein(
                config['fasta_dir'], pdb_id, config['precomputed_alignments']
            )

        metrics = collector.collect_all_metrics(
            method_name=method_name,
            output_dir=Path(config['output_dir']),
            msa_path=Path(config.get('msa_path')) if config.get('msa_path') else None,
            pair_path=Path(config.get('pair_path')) if config.get('pair_path') else None,
            ground_truth_msa_path=Path(config.get('ground_truth_msa_path')) if config.get(
                'ground_truth_msa_path') else None,
            ground_truth_pair_path=Path(config.get('ground_truth_pair_path')) if config.get(
                'ground_truth_pair_path') else None,
            processed_feature_dict=processed_feature_dict
        )
        all_metrics[method_name] = metrics

    rankings = collector.rank_methods_by_all_metrics(all_metrics)
    collector.print_comprehensive_summary(all_metrics, rankings)

    return {
        'metrics': all_metrics,
        'rankings': rankings,
        'summary': {
            'protein_id': pdb_id,
            'has_reference': collector.has_reference,
            'reference_path': collector.reference_structure_path
        }
    }


if __name__ == "__main__":
    print("🧬 Comprehensive Structure Prediction Metrics")
    print("=" * 50)
    print("This module provides comprehensive metrics for structure prediction analysis.")
    print("\nKey features:")
    print("• Finds reference structures in openfold_deconstructed directories")
    print("• Computes structural accuracy (RMSD, GDT-TS, TM-score)")
    print("• Analyzes confidence scores (pLDDT distributions)")
    print("• Evaluates representation losses (MSA, pair, single)")
    print("• Computes OpenFold auxiliary losses (masked MSA, distogram)")
    print("• Integrates with existing prediction pipelines")
    print("\nUsage examples:")
    print("  # Standalone usage:")
    print("  collector = ComprehensiveMetricsCollector('1fv5_A')")
    print("  metrics = collector.collect_all_metrics(...)")
    print("")
    print("  # Integration with runner:")
    print("  integrate_comprehensive_metrics(runner_instance)")
    print("")
    print("  # Batch analysis:")
    print("  results = analyze_method_predictions('1fv5_A', methods_config)")