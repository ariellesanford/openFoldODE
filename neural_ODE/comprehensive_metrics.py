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

# Add save_intermediates to path to import OpenFold components
# Adjust path based on where this script is run from
save_intermediates_path = Path(__file__).parent.parent / "save_intermediates"
if not save_intermediates_path.exists():
    # If running from neural_ODE directory, go up one level
    save_intermediates_path = Path(__file__).parent.parent.parent / "save_intermediates"
if save_intermediates_path.exists():
    sys.path.insert(0, str(save_intermediates_path))
else:
    print(f"⚠️  Warning: save_intermediates not found at expected paths")

try:
    # Import actual OpenFold heads and loss functions
    from openfold.model.heads import MaskedMSAHead, DistogramHead
    from openfold.utils.loss import masked_msa_loss, distogram_loss
    from openfold.config import model_config
    from openfold.utils.script_utils import load_models_from_command_line

    # Import the actual generate_single_from_msa function
    from run_structure_module import generate_single_from_msa

    OPENFOLD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import OpenFold components: {e}")
    print(f"   OpenFold auxiliary losses will be skipped")
    OPENFOLD_AVAILABLE = False


class ComprehensiveMetricsCollector:
    """Complete metrics collection for structure prediction accuracy"""

    def __init__(self, pdb_id: str, reference_structure_path: Path = None,
                 config_preset: str = "model_1_ptm"):
        self.pdb_id = pdb_id
        self.config_preset = config_preset
        self.reference_coords = None
        self.reference_structure_path = None
        self.has_reference = False

        # Try to load reference structure
        if reference_structure_path and reference_structure_path.exists():
            self.has_reference = self._load_reference_structure(reference_structure_path)
        else:
            self._try_auto_find_reference()

        # Initialize auxiliary heads for OpenFold-style losses
        self._init_auxiliary_heads()

    def _load_reference_structure(self, ref_path: Path) -> bool:
        """Load reference structure coordinates"""
        try:
            coords = self._parse_pdb_coordinates(ref_path)
            if coords is not None and len(coords) > 0:
                self.reference_coords = coords
                self.reference_structure_path = str(ref_path)
                print(f"✅ Loaded reference structure: {ref_path}")
                return True
        except Exception as e:
            print(f"❌ Error loading reference structure: {e}")
        return False

    def _try_auto_find_reference(self):
        """Try to automatically find reference structure"""
        possible_locations = [
            Path(f"/media/visitor/Extreme SSD/data/reference_structures/{self.pdb_id}.pdb"),
            Path(f"/media/visitor/Extreme SSD/data/reference_structures/{self.pdb_id[:4]}.pdb"),
            Path(f"./reference_structures/{self.pdb_id}.pdb"),
            Path(f"./reference_structures/{self.pdb_id[:4]}.pdb"),
        ]

        for ref_path in possible_locations:
            if ref_path.exists():
                self.has_reference = self._load_reference_structure(ref_path)
                if self.has_reference:
                    break

        if not self.has_reference:
            print(f"⚠️  No reference structure found for {self.pdb_id}")
            print(f"   Structural accuracy metrics will be skipped")

    def _init_auxiliary_heads(self):
        """Initialize actual OpenFold auxiliary heads for computing losses"""
        if not OPENFOLD_AVAILABLE:
            self.masked_msa_head = None
            self.distogram_head = None
            return

        try:
            # Load OpenFold config to get the correct head parameters
            config = model_config(self.config_preset)

            # Initialize actual OpenFold heads with config parameters
            self.masked_msa_head = MaskedMSAHead(**config.model.heads.masked_msa)
            self.distogram_head = DistogramHead(**config.model.heads.distogram)

            # Set to eval mode
            self.masked_msa_head.eval()
            self.distogram_head.eval()

            print(f"✅ Initialized OpenFold auxiliary heads")

        except Exception as e:
            print(f"⚠️  Could not initialize OpenFold auxiliary heads: {e}")
            self.masked_msa_head = None
            self.distogram_head = None

    def _parse_pdb_coordinates(self, pdb_path: Path) -> Optional[np.ndarray]:
        """Parse CA coordinates from PDB file"""
        coords = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[13:16].strip() == 'CA':
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])

            if coords:
                return np.array(coords)
        except Exception:
            pass
        return None

    # ==================== STRUCTURAL ACCURACY METRICS ====================

    def _kabsch_align(self, ref_coords: np.ndarray, pred_coords: np.ndarray) -> np.ndarray:
        """Align structures using Kabsch algorithm"""
        H = pred_coords.T @ ref_coords
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return pred_coords @ R

    def calculate_rmsd(self, predicted_pdb: Path) -> Optional[float]:
        """Calculate RMSD between predicted and reference structures"""
        if self.reference_coords is None:
            return None

        pred_coords = self._parse_pdb_coordinates(predicted_pdb)
        if pred_coords is None:
            return None

        # Align sequences (assume same length and order)
        min_len = min(len(self.reference_coords), len(pred_coords))
        ref_aligned = self.reference_coords[:min_len]
        pred_aligned = pred_coords[:min_len]

        # Center both structures
        ref_centered = ref_aligned - np.mean(ref_aligned, axis=0)
        pred_centered = pred_aligned - np.mean(pred_aligned, axis=0)

        try:
            aligned_pred = self._kabsch_align(ref_centered, pred_centered)
            rmsd = np.sqrt(np.mean(np.sum((ref_centered - aligned_pred) ** 2, axis=1)))
            return float(rmsd)
        except Exception:
            rmsd = np.sqrt(np.mean(np.sum((ref_centered - pred_centered) ** 2, axis=1)))
            return float(rmsd)

    def calculate_gdt_ts(self, predicted_pdb: Path, thresholds: List[float] = [1.0, 2.0, 4.0, 8.0]) -> Optional[float]:
        """Calculate GDT-TS (Global Distance Test - Total Score)"""
        if self.reference_coords is None:
            return None

        pred_coords = self._parse_pdb_coordinates(predicted_pdb)
        if pred_coords is None:
            return None

        min_len = min(len(self.reference_coords), len(pred_coords))
        ref_coords = self.reference_coords[:min_len]
        pred_coords = pred_coords[:min_len]

        # Center and align structures
        ref_centered = ref_coords - np.mean(ref_coords, axis=0)
        pred_centered = pred_coords - np.mean(pred_coords, axis=0)

        try:
            aligned_pred = self._kabsch_align(ref_centered, pred_centered)
            distances = np.sqrt(np.sum((ref_centered - aligned_pred) ** 2, axis=1))

            gdt_scores = []
            for threshold in thresholds:
                fraction_under_threshold = np.mean(distances <= threshold)
                gdt_scores.append(fraction_under_threshold)

            return float(np.mean(gdt_scores))
        except Exception:
            return None

    def calculate_tm_score_approximate(self, predicted_pdb: Path) -> Optional[float]:
        """Calculate approximate TM-score"""
        if self.reference_coords is None:
            return None

        pred_coords = self._parse_pdb_coordinates(predicted_pdb)
        if pred_coords is None:
            return None

        min_len = min(len(self.reference_coords), len(pred_coords))
        ref_coords = self.reference_coords[:min_len]
        pred_coords = pred_coords[:min_len]
        L_target = len(ref_coords)

        # Center and align
        ref_centered = ref_coords - np.mean(ref_coords, axis=0)
        pred_centered = pred_coords - np.mean(pred_coords, axis=0)

        try:
            aligned_pred = self._kabsch_align(ref_centered, pred_centered)
            distances = np.sqrt(np.sum((ref_centered - aligned_pred) ** 2, axis=1))

            # TM-score normalization
            d0 = 1.24 * (L_target - 15) ** (1 / 3) - 1.8
            if d0 <= 0:
                d0 = 0.5

            tm_score = np.mean(1.0 / (1.0 + (distances / d0) ** 2))
            return float(tm_score)
        except Exception:
            return None

    # ==================== CONFIDENCE AND pLDDT METRICS ====================

    def extract_comprehensive_plddt(self, pdb_path: Path) -> Dict[str, float]:
        """Extract comprehensive pLDDT statistics from PDB file"""
        if not pdb_path.exists():
            return self._empty_plddt_dict()

        plddt_scores = []
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[13:16].strip() == 'CA':
                        bfactor_str = line[60:66].strip()
                        if bfactor_str:
                            plddt_scores.append(float(bfactor_str))

            if not plddt_scores:
                return self._empty_plddt_dict()

            plddt_array = np.array(plddt_scores)

            # Confidence categories (AlphaFold conventions)
            very_high = np.mean(plddt_array >= 90)
            confident = np.mean(plddt_array >= 70)
            low = np.mean(plddt_array >= 50)
            very_low = np.mean(plddt_array < 50)

            return {
                "avg_plddt": float(np.mean(plddt_array)),
                "min_plddt": float(np.min(plddt_array)),
                "max_plddt": float(np.max(plddt_array)),
                "std_plddt": float(np.std(plddt_array)),
                "q25_plddt": float(np.percentile(plddt_array, 25)),
                "q75_plddt": float(np.percentile(plddt_array, 75)),
                "num_residues": len(plddt_scores),
                "very_high_confidence_fraction": very_high,
                "confident_fraction": confident,
                "low_confidence_fraction": low - confident,
                "very_low_confidence_fraction": very_low,
                "confidence_entropy": self._calculate_confidence_entropy(
                    [very_high, confident - very_high, low - confident, very_low]),
                "confidence_uniformity": float(np.std(plddt_array) / (np.mean(plddt_array) + 1e-10))
            }
        except Exception as e:
            print(f"   ⚠️  Error extracting pLDDT from {pdb_path}: {e}")
            return self._empty_plddt_dict()

    def _empty_plddt_dict(self) -> Dict[str, float]:
        """Return empty pLDDT dictionary"""
        return {k: None for k in [
            "avg_plddt", "min_plddt", "max_plddt", "std_plddt", "q25_plddt", "q75_plddt",
            "num_residues", "very_high_confidence_fraction", "confident_fraction",
            "low_confidence_fraction", "very_low_confidence_fraction",
            "confidence_entropy", "confidence_uniformity"
        ]}

    def _calculate_confidence_entropy(self, probs: List[float]) -> float:
        """Calculate entropy of confidence distribution"""
        return float(-np.sum([p * np.log(p + 1e-10) for p in probs if p > 0]))

    # ==================== REPRESENTATION LOSS METRICS ====================

    def compute_detailed_representation_loss(self, msa_path: Path, pair_path: Path,
                                             ground_truth_msa_path: Path, ground_truth_pair_path: Path) -> Dict[
        str, float]:
        """Compute comprehensive representation loss metrics"""
        if not all(path.exists() for path in [msa_path, pair_path, ground_truth_msa_path, ground_truth_pair_path]):
            return self._empty_representation_loss_dict()

        try:
            # Load tensors
            pred_msa = torch.load(msa_path, map_location='cpu')
            pred_pair = torch.load(pair_path, map_location='cpu')
            true_msa = torch.load(ground_truth_msa_path, map_location='cpu')
            true_pair = torch.load(ground_truth_pair_path, map_location='cpu')

            # Report tensor dimensions
            metrics = {
                "msa_dimensions": list(pred_msa.shape),
                "pair_dimensions": list(pred_pair.shape),
                "ground_truth_msa_dimensions": list(true_msa.shape),
                "ground_truth_pair_dimensions": list(true_pair.shape)
            }

            # Normalize dimensions (remove batch if present)
            if pred_msa.dim() == 4: pred_msa = pred_msa.squeeze(0)
            if pred_pair.dim() == 4: pred_pair = pred_pair.squeeze(0)
            if true_msa.dim() == 4: true_msa = true_msa.squeeze(0)
            if true_pair.dim() == 4: true_pair = true_pair.squeeze(0)

            # Match cluster sizes for MSA
            min_clusters = min(pred_msa.shape[0], true_msa.shape[0])
            pred_msa = pred_msa[:min_clusters]
            true_msa = true_msa[:min_clusters]

            # Individual MSA and pair losses
            msa_mse = F.mse_loss(pred_msa, true_msa).item()
            pair_mse = F.mse_loss(pred_pair, true_pair).item()
            msa_mae = F.l1_loss(pred_msa, true_msa).item()
            pair_mae = F.l1_loss(pred_pair, true_pair).item()

            # Cosine similarities
            msa_cos_sim = F.cosine_similarity(pred_msa.flatten(), true_msa.flatten(), dim=0).item()
            pair_cos_sim = F.cosine_similarity(pred_pair.flatten(), true_pair.flatten(), dim=0).item()

            # Adaptive loss (OpenFold style)
            msa_variance = true_msa.var().item() + 1e-8
            pair_variance = true_pair.var().item() + 1e-8
            msa_scaled = msa_mse / msa_variance
            pair_scaled = pair_mse / pair_variance
            total_adaptive_loss = msa_scaled + pair_scaled

            # Single representation loss (using actual OpenFold method)
            pred_single, true_single = self._compute_single_representations(pred_msa, true_msa)
            if pred_single is not None and true_single is not None:
                single_mse = F.mse_loss(pred_single, true_single).item()
                single_mae = F.l1_loss(pred_single, true_single).item()
                single_cos_sim = F.cosine_similarity(pred_single.flatten(), true_single.flatten(), dim=0).item()
            else:
                single_mse = single_mae = single_cos_sim = None

            metrics.update({
                # Individual tensor losses
                "msa_mse_loss": msa_mse,
                "pair_mse_loss": pair_mse,
                "msa_mae_loss": msa_mae,
                "pair_mae_loss": pair_mae,

                # Cosine similarities
                "msa_cosine_similarity": msa_cos_sim,
                "pair_cosine_similarity": pair_cos_sim,
                "average_cosine_similarity": (msa_cos_sim + pair_cos_sim) / 2,

                # Adaptive losses
                "msa_adaptive_loss": msa_scaled,
                "pair_adaptive_loss": pair_scaled,
                "total_adaptive_loss": total_adaptive_loss,

                # Single representation losses
                "single_mse_loss": single_mse,
                "single_mae_loss": single_mae,
                "single_cosine_similarity": single_cos_sim,

                # Variances for context
                "msa_variance": msa_variance,
                "pair_variance": pair_variance
            })

            return metrics

        except Exception as e:
            print(f"   ⚠️  Error computing representation loss: {e}")
            return self._empty_representation_loss_dict()

    def _empty_representation_loss_dict(self) -> Dict[str, float]:
        """Return empty representation loss dictionary"""
        keys = [
            "msa_dimensions", "pair_dimensions", "ground_truth_msa_dimensions", "ground_truth_pair_dimensions",
            "msa_mse_loss", "pair_mse_loss", "msa_mae_loss", "pair_mae_loss",
            "msa_cosine_similarity", "pair_cosine_similarity", "average_cosine_similarity",
            "msa_adaptive_loss", "pair_adaptive_loss", "total_adaptive_loss",
            "single_mse_loss", "single_mae_loss", "single_cosine_similarity",
            "msa_variance", "pair_variance"
        ]

    def _compute_single_representations(self, pred_msa: torch.Tensor, true_msa: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute single representations using actual OpenFold generate_single_from_msa method"""
        if not OPENFOLD_AVAILABLE:
            # Fallback to simple first row extraction
            return pred_msa[0], true_msa[0]

        try:
            # Load a lightweight model just to get the linear layer
            config = model_config(self.config_preset)

            # Create a minimal model instance to get the evoformer linear layer
            # This is expensive but gives us the actual single representation computation
            model_generator = load_models_from_command_line(
                config, 'cpu', None, None, '/tmp'
            )
            model, _ = next(model_generator)

            # Use the actual generate_single_from_msa function
            # Add batch dimension if needed
            if pred_msa.dim() == 3:
                pred_msa = pred_msa.unsqueeze(0)
            if true_msa.dim() == 3:
                true_msa = true_msa.unsqueeze(0)

            pred_single = generate_single_from_msa(pred_msa, model, 'cpu')
            true_single = generate_single_from_msa(true_msa, model, 'cpu')

            # Remove batch dimension if added
            if pred_single.dim() == 3:
                pred_single = pred_single.squeeze(0)
            if true_single.dim() == 3:
                true_single = true_single.squeeze(0)

            return pred_single, true_single

        except Exception as e:
            print(f"   ⚠️  Error computing single representations with model: {e}")
            print(f"   Falling back to first MSA row approximation")
            # Fallback to simple approach
            return pred_msa[0] if pred_msa.dim() == 3 else pred_msa.squeeze(0)[0], \
                true_msa[0] if true_msa.dim() == 3 else true_msa.squeeze(0)[0]

    def _empty_representation_loss_dict(self) -> Dict[str, float]:
        """Return empty representation loss dictionary"""
        keys = [
            "msa_dimensions", "pair_dimensions", "ground_truth_msa_dimensions", "ground_truth_pair_dimensions",
            "msa_mse_loss", "pair_mse_loss", "msa_mae_loss", "pair_mae_loss",
            "msa_cosine_similarity", "pair_cosine_similarity", "average_cosine_similarity",
            "msa_adaptive_loss", "pair_adaptive_loss", "total_adaptive_loss",
            "single_mse_loss", "single_mae_loss", "single_cosine_similarity",
            "msa_variance", "pair_variance"
        ]
        return {k: None for k in keys}

    # ==================== OPENFOLD-STYLE AUXILIARY LOSSES ====================

    def compute_openfold_auxiliary_losses(self, msa_tensor: torch.Tensor, pair_tensor: torch.Tensor,
                                          batch_dict: Dict = None) -> Dict[str, float]:
        """Compute actual OpenFold auxiliary losses using real heads and loss functions"""
        if not OPENFOLD_AVAILABLE or self.masked_msa_head is None or self.distogram_head is None:
            return {"masked_msa_loss": None, "distogram_loss": None}

        try:
            losses = {}

            # Ensure tensors have correct dimensions
            if msa_tensor.dim() == 3:
                msa_tensor = msa_tensor.unsqueeze(0)  # Add batch dim
            if pair_tensor.dim() == 3:
                pair_tensor = pair_tensor.unsqueeze(0)  # Add batch dim

            # 1. MASKED MSA LOSS
            if batch_dict and all(k in batch_dict for k in ["true_msa", "bert_mask"]):
                try:
                    with torch.no_grad():
                        # Compute masked MSA logits using actual OpenFold head
                        masked_msa_logits = self.masked_msa_head(msa_tensor)

                        # Use actual OpenFold masked_msa_loss function
                        msa_loss = masked_msa_loss(
                            logits=masked_msa_logits,
                            true_msa=batch_dict["true_msa"],
                            bert_mask=batch_dict["bert_mask"],
                            num_classes=23,  # 20 amino acids + 3 special tokens
                            eps=1e-8
                        )
                        losses["masked_msa_loss"] = msa_loss.item()

                except Exception as e:
                    print(f"   ⚠️  Error computing masked MSA loss: {e}")
                    losses["masked_msa_loss"] = None
            else:
                losses["masked_msa_loss"] = None

            # 2. DISTOGRAM LOSS
            if batch_dict and all(k in batch_dict for k in ["pseudo_beta", "pseudo_beta_mask"]):
                try:
                    with torch.no_grad():
                        # Compute distogram logits using actual OpenFold head
                        distogram_logits = self.distogram_head(pair_tensor)

                        # Use actual OpenFold distogram_loss function
                        dist_loss = distogram_loss(
                            logits=distogram_logits,
                            pseudo_beta=batch_dict["pseudo_beta"],
                            pseudo_beta_mask=batch_dict["pseudo_beta_mask"],
                            min_bin=2.3125,
                            max_bin=21.6875,
                            no_bins=64,
                            eps=1e-6
                        )
                        losses["distogram_loss"] = dist_loss.item()

                except Exception as e:
                    print(f"   ⚠️  Error computing distogram loss: {e}")
                    losses["distogram_loss"] = None
            else:
                losses["distogram_loss"] = None

            return losses

        except Exception as e:
            print(f"   ⚠️  Error computing OpenFold auxiliary losses: {e}")
            return {"masked_msa_loss": None, "distogram_loss": None}

    def create_proper_batch_dict_from_processed_features(self, processed_feature_dict: Dict,
                                                         msa_tensor: torch.Tensor) -> Dict:
        """Create proper batch dictionary from actual processed features used during training"""
        if not OPENFOLD_AVAILABLE:
            return {}

        batch_dict = {}

        try:
            # Get dimensions
            if msa_tensor.dim() == 3:
                msa_tensor = msa_tensor.unsqueeze(0)
            batch_size, n_seq, n_res, c_m = msa_tensor.shape

            # 1. TRUE MSA - Use the actual aatype from processed features
            if "aatype" in processed_feature_dict:
                aatype = processed_feature_dict["aatype"]

                # Ensure proper dimensions
                if aatype.dim() == 1:
                    aatype = aatype.unsqueeze(0)  # Add batch dim

                # Create true_msa: first row is the target sequence, others can be masked
                true_msa = torch.zeros(batch_size, n_seq, n_res, dtype=torch.long)

                # First row (sequence 0) gets the actual target sequence
                true_msa[0, 0, :] = aatype[0, :n_res]

                # For other sequences, we'd need the actual MSA, but we can approximate
                # by repeating the target sequence (not ideal but better than random)
                for seq_idx in range(1, min(n_seq, 10)):  # Limit to avoid memory issues
                    true_msa[0, seq_idx, :] = aatype[0, :n_res]

                batch_dict["true_msa"] = true_msa

                # 2. BERT MASK - Use actual training mask pattern if available
                if "bert_mask" in processed_feature_dict:
                    bert_mask = processed_feature_dict["bert_mask"]
                    if bert_mask.dim() == 2:
                        bert_mask = bert_mask.unsqueeze(0)  # Add batch dim
                    batch_dict["bert_mask"] = bert_mask.float()
                else:
                    # Create a more realistic mask pattern
                    # Mask 15% but avoid masking special positions
                    bert_mask = torch.zeros_like(true_msa, dtype=torch.float32)

                    # Only mask the first few sequences and avoid padding
                    if "seq_mask" in processed_feature_dict:
                        seq_mask = processed_feature_dict["seq_mask"]
                        if seq_mask.dim() == 1:
                            seq_mask = seq_mask.unsqueeze(0)

                        # Only mask valid residues (seq_mask == 1)
                        for seq_idx in range(min(n_seq, 5)):  # Only first few sequences
                            valid_positions = seq_mask[0] == 1
                            n_valid = valid_positions.sum().item()
                            n_mask = int(0.15 * n_valid)

                            if n_mask > 0:
                                valid_indices = torch.where(valid_positions)[0]
                                mask_indices = valid_indices[torch.randperm(len(valid_indices))[:n_mask]]
                                bert_mask[0, seq_idx, mask_indices] = 1.0

                    batch_dict["bert_mask"] = bert_mask

            # 3. PSEUDO BETA - Use actual coordinates if available
            if "pseudo_beta" in processed_feature_dict:
                pseudo_beta = processed_feature_dict["pseudo_beta"]

                # Ensure proper dimensions
                if pseudo_beta.dim() == 2:
                    pseudo_beta = pseudo_beta.unsqueeze(0)  # Add batch dim

                batch_dict["pseudo_beta"] = pseudo_beta

                # Create mask based on seq_mask
                if "seq_mask" in processed_feature_dict:
                    seq_mask = processed_feature_dict["seq_mask"]
                    if seq_mask.dim() == 1:
                        seq_mask = seq_mask.unsqueeze(0)
                    batch_dict["pseudo_beta_mask"] = seq_mask.float()
                else:
                    batch_dict["pseudo_beta_mask"] = torch.ones(batch_size, n_res)

            elif "all_atom_positions" in processed_feature_dict:
                # Compute pseudo_beta from all_atom_positions
                try:
                    from openfold.np import residue_constants
                    all_atom_positions = processed_feature_dict["all_atom_positions"]
                    aatype = processed_feature_dict["aatype"]

                    if all_atom_positions.dim() == 3:
                        all_atom_positions = all_atom_positions.unsqueeze(0)
                    if aatype.dim() == 1:
                        aatype = aatype.unsqueeze(0)

                    # Compute pseudo-beta positions (CA for glycine, CB for others)
                    pseudo_beta = self._compute_pseudo_beta_from_positions(
                        all_atom_positions, aatype
                    )
                    batch_dict["pseudo_beta"] = pseudo_beta

                    # Create mask
                    if "seq_mask" in processed_feature_dict:
                        seq_mask = processed_feature_dict["seq_mask"]
                        if seq_mask.dim() == 1:
                            seq_mask = seq_mask.unsqueeze(0)
                        batch_dict["pseudo_beta_mask"] = seq_mask.float()
                    else:
                        batch_dict["pseudo_beta_mask"] = torch.ones(batch_size, n_res)

                except ImportError:
                    print(f"   ⚠️  Could not compute pseudo_beta from coordinates")
                    # Fall back to dummy positions
                    batch_dict["pseudo_beta"] = torch.randn(batch_size, n_res, 3)
                    batch_dict["pseudo_beta_mask"] = torch.ones(batch_size, n_res)
            else:
                # Last resort: dummy positions (will give meaningless distogram loss)
                print(f"   ⚠️  No coordinate data available, using dummy pseudo_beta")
                batch_dict["pseudo_beta"] = torch.randn(batch_size, n_res, 3)
                batch_dict["pseudo_beta_mask"] = torch.ones(batch_size, n_res)

            return batch_dict

        except Exception as e:
            print(f"   ⚠️  Error creating batch dict from processed features: {e}")
            return {}

    def _compute_pseudo_beta_from_positions(self, all_atom_positions: torch.Tensor,
                                            aatype: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-beta positions from all-atom coordinates"""
        try:
            from openfold.np import residue_constants

            batch_size, n_res, n_atoms, _ = all_atom_positions.shape
            pseudo_beta = torch.zeros(batch_size, n_res, 3)

            for b in range(batch_size):
                for i in range(n_res):
                    residue_type = aatype[b, i].item()

                    if residue_type < len(residue_constants.restypes):
                        resname = residue_constants.restypes[residue_type]

                        if resname == 'G':  # Glycine - use CA
                            ca_idx = residue_constants.atom_order.get('CA', 1)
                            pseudo_beta[b, i] = all_atom_positions[b, i, ca_idx]
                        else:  # Other residues - use CB
                            cb_idx = residue_constants.atom_order.get('CB', 4)
                            pseudo_beta[b, i] = all_atom_positions[b, i, cb_idx]
                    else:
                        # Unknown residue type, use CA
                        ca_idx = residue_constants.atom_order.get('CA', 1)
                        pseudo_beta[b, i] = all_atom_positions[b, i, ca_idx]

            return pseudo_beta

        except Exception as e:
            print(f"   ⚠️  Error computing pseudo_beta: {e}")
            # Return dummy positions
            return torch.randn(all_atom_positions.shape[0], all_atom_positions.shape[1], 3)

    # ==================== COMPREHENSIVE METRICS COLLECTION ====================

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
            # Use first available PDB file
            pdb_path, structure_type = pdb_files[0]
            metrics["structure_file"] = pdb_path.name
            metrics["structure_type"] = structure_type

            # 1. Comprehensive pLDDT metrics
            plddt_metrics = self.extract_comprehensive_plddt(pdb_path)
            metrics.update(plddt_metrics)

            # 2. Structural accuracy metrics (if reference available)
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

        # 3. Representation loss metrics (if inputs provided)
        if msa_path and pair_path and ground_truth_msa_path and ground_truth_pair_path:
            representation_metrics = self.compute_detailed_representation_loss(
                msa_path, pair_path, ground_truth_msa_path, ground_truth_pair_path
            )
            metrics.update(representation_metrics)

        # 4. OpenFold auxiliary losses (if tensors and processed features available)
        if msa_path and pair_path and processed_feature_dict:
            try:
                msa_tensor = torch.load(msa_path, map_location='cpu')
                pair_tensor = torch.load(pair_path, map_location='cpu')

                # Remove batch dimensions if present
                if msa_tensor.dim() == 4: msa_tensor = msa_tensor.squeeze(0)
                if pair_tensor.dim() == 4: pair_tensor = pair_tensor.squeeze(0)

                # Create proper batch dict using actual processed features
                aux_batch_dict = self.create_proper_batch_dict_from_processed_features(
                    processed_feature_dict, msa_tensor
                )

                # Compute actual OpenFold auxiliary losses
                auxiliary_losses = self.compute_openfold_auxiliary_losses(
                    msa_tensor, pair_tensor, aux_batch_dict
                )
                metrics.update(auxiliary_losses)

            except Exception as e:
                print(f"   ⚠️  Could not compute OpenFold auxiliary losses: {e}")
                metrics.update({"masked_msa_loss": None, "distogram_loss": None})
        else:
            metrics.update({"masked_msa_loss": None, "distogram_loss": None})

        return metrics

    # ==================== RANKING AND SUMMARY ====================

    def rank_methods_by_all_metrics(self, all_metrics: Dict[str, Dict]) -> Dict[str, List]:
        """Rank methods by all available metrics"""
        rankings = {}

        # Rank by structure quality (pLDDT)
        methods_with_plddt = [(name, data["avg_plddt"]) for name, data in all_metrics.items()
                              if data.get("avg_plddt") is not None]
        if methods_with_plddt:
            methods_with_plddt.sort(key=lambda x: x[1], reverse=True)
            rankings["by_avg_plddt"] = methods_with_plddt

        # Rank by representation accuracy
        metrics_to_rank = [
            ("by_total_adaptive_loss", "total_adaptive_loss", False),  # Lower is better
            ("by_msa_cosine_similarity", "msa_cosine_similarity", True),  # Higher is better
            ("by_pair_cosine_similarity", "pair_cosine_similarity", True),  # Higher is better
            ("by_single_mse_loss", "single_mse_loss", False),  # Lower is better
        ]

        for rank_name, metric_key, reverse in metrics_to_rank:
            methods_with_metric = [(name, data[metric_key]) for name, data in all_metrics.items()
                                   if data.get(metric_key) is not None]
            if methods_with_metric:
                methods_with_metric.sort(key=lambda x: x[1], reverse=reverse)
                rankings[rank_name] = methods_with_metric

        # Rank by structural accuracy (if available)
        if any(data.get("structural_accuracy") for data in all_metrics.values()):
            structural_metrics = [
                ("by_rmsd", "rmsd_ca", False),  # Lower is better
                ("by_gdt_ts", "gdt_ts", True),  # Higher is better
                ("by_tm_score", "tm_score_approx", True),  # Higher is better
            ]

            for rank_name, metric_key, reverse in structural_metrics:
                methods_with_metric = [(name, data["structural_accuracy"][metric_key])
                                       for name, data in all_metrics.items()
                                       if (data.get("structural_accuracy") and
                                           data["structural_accuracy"].get(metric_key) is not None)]
                if methods_with_metric:
                    methods_with_metric.sort(key=lambda x: x[1], reverse=reverse)
                    rankings[rank_name] = methods_with_metric

        return rankings

    def print_comprehensive_summary(self, all_metrics: Dict[str, Dict], rankings: Dict[str, List]):
        """Print comprehensive summary of all metrics"""
        print(f"\n📊 COMPREHENSIVE ACCURACY ANALYSIS FOR {self.pdb_id}")
        print("=" * 80)

        # Structure Quality Rankings
        if "by_avg_plddt" in rankings:
            print(f"\n🏆 STRUCTURE QUALITY RANKING (pLDDT):")
            for i, (method, plddt) in enumerate(rankings["by_avg_plddt"], 1):
                confidence = self._get_confidence_category(plddt)
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

        # Structural Accuracy Rankings (if reference available)
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

        # Individual tensor analysis
        print(f"\n📋 DETAILED TENSOR ANALYSIS:")
        for method_name, metrics in all_metrics.items():
            if metrics.get("msa_dimensions"):
                print(f"\n  {method_name}:")
                print(f"    MSA tensor: {metrics['msa_dimensions']} (shape)")
                print(f"    Pair tensor: {metrics['pair_dimensions']} (shape)")
                if metrics.get("msa_mse_loss") is not None:
                    print(f"    MSA MSE loss: {metrics['msa_mse_loss']:.6f}")
                    print(f"    Pair MSE loss: {metrics['pair_mse_loss']:.6f}")
                    print(f"    Single repr. MSE: {metrics['single_mse_loss']:.6f}")
                if metrics.get("masked_msa_loss") is not None:
                    print(f"    Masked MSA loss: {metrics['masked_msa_loss']:.6f}")
                if metrics.get("distogram_loss") is not None:
                    print(f"    Distogram loss: {metrics['distogram_loss']:.6f}")

        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Total methods analyzed: {len(all_metrics)}")
        print(f"   Methods with structure quality: {len(rankings.get('by_avg_plddt', []))}")
        print(f"   Methods with representation loss: {len(rankings.get('by_total_adaptive_loss', []))}")
        if self.has_reference:
            print(f"   Methods with structural accuracy: {len(rankings.get('by_rmsd', []))}")
            print(f"   Reference structure: {self.reference_structure_path}")
        else:
            print(f"   Structural accuracy: Not available (no reference structure)")

        # Best overall method
        if "by_avg_plddt" in rankings and rankings["by_avg_plddt"]:
            best_method = rankings["by_avg_plddt"][0][0]
            best_metrics = all_metrics[best_method]
            print(f"\n🥇 BEST OVERALL METHOD: {best_method}")
            print(f"   Structure Quality: {best_metrics.get('avg_plddt', 'N/A')} pLDDT")
            if best_metrics.get('total_adaptive_loss') is not None:
                print(f"   Representation Loss: {best_metrics['total_adaptive_loss']:.6f}")
            if best_metrics.get('structural_accuracy'):
                struct_acc = best_metrics['structural_accuracy']
                if struct_acc.get('rmsd_ca'):
                    print(f"   RMSD: {struct_acc['rmsd_ca']:.2f} Å")
                if struct_acc.get('gdt_ts'):
                    print(f"   GDT-TS: {struct_acc['gdt_ts']:.3f}")
                if struct_acc.get('tm_score_approx'):
                    print(f"   TM-Score: {struct_acc['tm_score_approx']:.3f}")

    def _get_confidence_category(self, plddt: float) -> str:
        """Get confidence category emoji for pLDDT score"""
        if plddt >= 90:
            return "🟢 (Very High)"
        elif plddt >= 70:
            return "🟡 (High)"
        elif plddt >= 50:
            return "🟠 (Low)"
        else:
            return "🔴 (Very Low)"


# ==================== HELPER FUNCTIONS FOR BATCH DATA ====================

def load_processed_feature_dict_for_protein(fasta_dir: Path, pdb_id: str,
                                            precomputed_alignments_dir: Path = None) -> Optional[Dict]:
    """Load and process features just like the structure prediction pipeline does"""
    if not OPENFOLD_AVAILABLE:
        return None

    try:
        # Import what we need for feature processing
        from openfold.data import data_pipeline, feature_pipeline
        from openfold.config import model_config
        from openfold.utils.tensor_utils import tensor_tree_map
        import tempfile

        # Find FASTA file
        fasta_files = list(fasta_dir.glob("*.fasta")) + list(fasta_dir.glob("*.fa"))
        if not fasta_files:
            return None

        fasta_path = fasta_files[0]

        # Read FASTA content
        with open(fasta_path, 'r') as f:
            fasta_data = f.read()

        # Parse FASTA (simple parsing)
        lines = fasta_data.strip().split('\n')
        sequence = ""
        for line in lines:
            if not line.startswith('>'):
                sequence += line.strip()

        # Create temporary FASTA file with clean header
        tmp_fasta_path = fasta_dir / f"tmp_{os.getpid()}.fasta"
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{pdb_id}\n{sequence}")

        try:
            # Set up data pipeline (simplified)
            config = model_config("model_1_ptm")
            data_processor = data_pipeline.DataPipeline(template_featurizer=None)
            feature_processor = feature_pipeline.FeaturePipeline(config.data)

            # Process features
            if precomputed_alignments_dir and precomputed_alignments_dir.exists():
                feature_dict = data_processor.process_fasta(
                    fasta_path=str(tmp_fasta_path),
                    alignment_dir=str(precomputed_alignments_dir / pdb_id),
                )
            else:
                # Try without alignments (will be limited)
                feature_dict = data_processor.process_fasta(
                    fasta_path=str(tmp_fasta_path),
                    alignment_dir=None,
                )

            # Process features like the main pipeline
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

    except Exception as e:
        print(f"   ⚠️  Error loading processed features: {e}")
        return None


def create_proper_batch_dict_from_feature_dict(feature_dict: Dict,
                                               aatype_tensor: torch.Tensor) -> Dict:
    """Create proper batch dictionary from processed features for auxiliary losses"""
    batch_dict = {}

    try:
        # Get dimensions
        n_res = aatype_tensor.shape[0] if aatype_tensor.dim() == 1 else aatype_tensor.shape[1]

        # 1. True MSA from aatype
        if aatype_tensor.dim() == 1:
            true_msa = aatype_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and seq dims
        else:
            true_msa = aatype_tensor.unsqueeze(0)  # Add batch dim

        batch_dict["true_msa"] = true_msa.long()

        # 2. Create BERT mask (15% masking probability)
        bert_mask = torch.bernoulli(torch.full_like(true_msa.float(), 0.15))
        batch_dict["bert_mask"] = bert_mask

        # 3. Pseudo beta from feature dict if available
        if "pseudo_beta" in feature_dict:
            pseudo_beta = torch.tensor(feature_dict["pseudo_beta"])
            if pseudo_beta.dim() == 2:
                pseudo_beta = pseudo_beta.unsqueeze(0)  # Add batch dim
            batch_dict["pseudo_beta"] = pseudo_beta

            pseudo_beta_mask = torch.ones(pseudo_beta.shape[:-1])
            batch_dict["pseudo_beta_mask"] = pseudo_beta_mask
        else:
            # Create dummy pseudo_beta if not available
            batch_dict["pseudo_beta"] = torch.randn(1, n_res, 3)
            batch_dict["pseudo_beta_mask"] = torch.ones(1, n_res)

        return batch_dict

    except Exception as e:
        print(f"⚠️  Error creating batch dict from features: {e}")
        return {}


# ==================== INTEGRATION FUNCTIONS ====================

def integrate_comprehensive_metrics(runner_instance):
    """
    Integration function to add comprehensive metrics to the main prediction runner

    This replaces the simple metrics collection with comprehensive analysis
    """

    # Initialize comprehensive metrics collector
    runner_instance.comprehensive_metrics = ComprehensiveMetricsCollector(
        runner_instance.pdb_id
    )

    # Override the original metrics collection method
    # Override the original metrics collection method to include processed features
    def enhanced_collect_method_metrics(method_name: str, output_dir: Path,
                                        msa_path: Path = None, pair_path: Path = None) -> Dict:
        """Comprehensive version of collect_method_metrics with actual feature processing"""

        # Load processed features (this is the key to getting real auxiliary losses)
        processed_feature_dict = None
        if hasattr(runner_instance, 'fasta_dir') and runner_instance.fasta_dir:
            precomputed_alignments = getattr(runner_instance, 'precomputed_alignments', None)
            processed_feature_dict = load_processed_feature_dict_for_protein(
                runner_instance.fasta_dir,
                runner_instance.pdb_id,
                precomputed_alignments
            )

        return runner_instance.comprehensive_metrics.collect_all_metrics(
            method_name, output_dir, msa_path, pair_path,
            runner_instance.ground_truth_msa_path, runner_instance.ground_truth_pair_path,
            processed_feature_dict
        )

    runner_instance.collect_method_metrics = enhanced_collect_method_metrics

    # Override the metrics summary methods
    def comprehensive_save_metrics_summary():
        """Comprehensive version with all metrics and rankings"""
        if not runner_instance.method_metrics:
            return

        summary_file = runner_instance.predictions_base_dir / "comprehensive_metrics_analysis.json"

        # Generate comprehensive rankings
        rankings = runner_instance.comprehensive_metrics.rank_methods_by_all_metrics(
            runner_instance.method_metrics
        )

        # Create comprehensive summary
        summary = {
            "protein_id": runner_instance.pdb_id,
            "generated_at": runner_instance._get_timestamp(),
            "analysis_type": "comprehensive_structure_prediction_metrics",
            "has_reference_structure": runner_instance.comprehensive_metrics.has_reference,
            "reference_structure_path": runner_instance.comprehensive_metrics.reference_structure_path,
            "ground_truth_files": {
                "msa_block_48": str(runner_instance.ground_truth_msa_path),
                "pair_block_48": str(runner_instance.ground_truth_pair_path)
            },
            "methods_analyzed": runner_instance.method_metrics,
            "rankings_by_metric": rankings,
            "summary_statistics": compute_comprehensive_summary_stats(runner_instance.method_metrics, rankings),
            "metric_definitions": {
                "plddt": "Per-residue confidence score (0-100, higher=better)",
                "rmsd": "Root Mean Square Deviation in Angstroms (lower=better)",
                "gdt_ts": "Global Distance Test Total Score (0-1, higher=better)",
                "tm_score": "Template Modeling score (0-1, >0.5=same fold, higher=better)",
                "adaptive_loss": "MSA+pair loss scaled by variance (lower=better)",
                "cosine_similarity": "Representation similarity (0-1, higher=better)",
                "masked_msa_loss": "BERT-style MSA reconstruction loss (lower=better)",
                "distogram_loss": "Distance prediction loss (lower=better)"
            }
        }

        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\n📊 Comprehensive metrics analysis saved: {summary_file}")
            runner_instance.comprehensive_metrics.print_comprehensive_summary(
                runner_instance.method_metrics, rankings
            )

        except Exception as e:
            print(f"❌ Failed to save comprehensive metrics: {e}")

    # Replace the summary methods
    runner_instance.save_metrics_summary = comprehensive_save_metrics_summary

    # Replace simple summary stats
    def compute_comprehensive_summary_stats(all_metrics: Dict, rankings: Dict) -> Dict:
        """Compute comprehensive summary statistics"""
        stats = {
            "total_methods": len(all_metrics),
            "methods_with_structure_quality": len(rankings.get("by_avg_plddt", [])),
            "methods_with_representation_loss": len(rankings.get("by_total_adaptive_loss", [])),
            "methods_with_structural_accuracy": len(rankings.get("by_rmsd", [])),
        }

        # Best performances across all metrics
        if "by_avg_plddt" in rankings and rankings["by_avg_plddt"]:
            stats["best_plddt_method"] = rankings["by_avg_plddt"][0][0]
            stats["best_plddt_score"] = rankings["by_avg_plddt"][0][1]

        if "by_total_adaptive_loss" in rankings and rankings["by_total_adaptive_loss"]:
            stats["best_representation_method"] = rankings["by_total_adaptive_loss"][0][0]
            stats["best_representation_loss"] = rankings["by_total_adaptive_loss"][0][1]

        if "by_rmsd" in rankings and rankings["by_rmsd"]:
            stats["best_structural_method"] = rankings["by_rmsd"][0][0]
            stats["best_rmsd"] = rankings["by_rmsd"][0][1]

        # Distribution of confidence levels
        plddt_scores = [m["avg_plddt"] for m in all_metrics.values() if m.get("avg_plddt") is not None]
        if plddt_scores:
            stats["confidence_distribution"] = {
                "very_high_confidence_methods": len([p for p in plddt_scores if p >= 90]),
                "high_confidence_methods": len([p for p in plddt_scores if 70 <= p < 90]),
                "low_confidence_methods": len([p for p in plddt_scores if 50 <= p < 70]),
                "very_low_confidence_methods": len([p for p in plddt_scores if p < 50])
            }

        return stats

    runner_instance._compute_enhanced_summary_stats = lambda: compute_comprehensive_summary_stats(
        runner_instance.method_metrics,
        runner_instance.comprehensive_metrics.rank_methods_by_all_metrics(runner_instance.method_metrics)
    )


# ==================== USAGE EXAMPLE ====================

def example_usage():
    """
    Example of how to use the comprehensive metrics collector
    """

    # Initialize
    collector = ComprehensiveMetricsCollector(
        pdb_id="1fv5_A",
        reference_structure_path=Path("/path/to/reference/1fv5_A.pdb")
    )

    # Collect metrics for a single method
    metrics = collector.collect_all_metrics(
        method_name="Neural ODE",
        output_dir=Path("/path/to/output"),
        msa_path=Path("/path/to/msa.pt"),
        pair_path=Path("/path/to/pair.pt"),
        ground_truth_msa_path=Path("/path/to/ground_truth_msa.pt"),
        ground_truth_pair_path=Path("/path/to/ground_truth_pair.pt")
    )

    # Print metrics
    print(json.dumps(metrics, indent=2))

    # For multiple methods, collect all and rank
    all_methods_metrics = {
        "neural_ode": metrics,
        # ... add other methods
    }

    rankings = collector.rank_methods_by_all_metrics(all_methods_metrics)
    collector.print_comprehensive_summary(all_methods_metrics, rankings)


if __name__ == "__main__":
    print("🧬 Comprehensive Structure Prediction Metrics")
    print("=" * 50)
    print("This module provides:")
    print("• Structural accuracy: RMSD, GDT-TS, TM-score")
    print("• Confidence analysis: pLDDT distributions")
    print("• Representation losses: MSA, pair, single")
    print("• Tensor analysis: dimensions, individual losses")
    print("• Auxiliary losses: masked MSA, distogram")
    print("• Comprehensive rankings and summaries")
    print("")
    print("Usage:")
    print("  from comprehensive_metrics import integrate_comprehensive_metrics")
    print("  integrate_comprehensive_metrics(your_runner_instance)")
    print("")