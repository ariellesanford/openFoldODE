#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
import time
import torch
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

from openfold.config import model_config
from openfold.data import feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.utils.script_utils import (
    load_models_from_command_line, parse_fasta, prep_output, relax_protein, update_timings
)
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import protein


def load_evoformer_outputs(msa_path, pair_path):
    """Load Evoformer outputs from individual file paths."""

    if not os.path.exists(pair_path):
        raise FileNotFoundError(f"Pair representation file not found: {pair_path}")

    if not os.path.exists(msa_path):
        raise FileNotFoundError(f"MSA representation file not found: {msa_path}")

    logger.info(f"Loading pair representation from {pair_path}")
    pair = torch.load(pair_path, map_location='cpu')
    logger.info(f"Loaded pair representation: {pair.shape}")

    logger.info(f"Loading MSA representation from {msa_path}")
    msa = torch.load(msa_path, map_location='cpu')
    logger.info(f"Loaded MSA representation: {msa.shape}")

    return {
        'single': None,
        'pair': pair,
        'msa': msa
    }


def generate_single_from_msa(msa_representation, model, device):
    """Generate single representation from MSA using the model's linear layer."""
    logger.info(f"Generating single representation from MSA: {msa_representation.shape}")

    linear_layer = model.evoformer.linear
    msa_representation = msa_representation.to(device)

    # Extract first sequence: [batch, N_seq, N_res, C_m] -> [batch, N_res, C_m]
    first_msa_row = msa_representation[..., 0, :, :]

    with torch.no_grad():
        single_representation = linear_layer(first_msa_row)

    logger.info(f"Generated single representation: {single_representation.shape}")
    return single_representation


def squeeze_all_dims(tensor):
    """Remove all singleton dimensions from a tensor."""
    if torch.is_tensor(tensor):
        while tensor.dim() > 0 and 1 in tensor.shape:
            tensor = tensor.squeeze()
        return tensor
    return tensor


def process_protein(args, config, model, data_processor, feature_processor):
    """Process a single protein."""

    logger.info(f"Processing protein with:")
    logger.info(f"  Pair path: {args.pair_path}")
    logger.info(f"  MSA path: {args.msa_path}")

    # Load Evoformer outputs
    evoformer_outputs = load_evoformer_outputs(
        msa_path=args.msa_path,
        pair_path=args.pair_path
    )

    # Load and process FASTA exactly like run_pretrained_openfold.py
    fasta_files = [f for f in os.listdir(args.fasta_dir) if f.endswith(('.fasta', '.fa'))]
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {args.fasta_dir}")

    fasta_path = os.path.join(args.fasta_dir, fasta_files[0])

    with open(fasta_path, 'r') as f:
        fasta_data = f.read()

    tags, seqs = parse_fasta(fasta_data)
    tag = tags[0]
    seq = seqs[0]

    # Create temporary FASTA file
    tmp_fasta_path = os.path.join(args.fasta_dir, f"tmp_{os.getpid()}.fasta")
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    try:
        # Process features exactly like run_pretrained_openfold.py lines 89-99
        local_alignment_dir = os.path.join(args.use_precomputed_alignments, tag)

        if os.path.exists(local_alignment_dir):
            logger.info(f"Using precomputed alignments from {local_alignment_dir}")
            feature_dict = data_processor.process_fasta(
                fasta_path=tmp_fasta_path,
                alignment_dir=local_alignment_dir,
            )
        else:
            raise FileNotFoundError(f"No precomputed alignments found at {local_alignment_dir}")
    finally:
        if os.path.exists(tmp_fasta_path):
            os.remove(tmp_fasta_path)

    # Process features exactly like run_pretrained_openfold.py
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict'
    )

    logger.info(f"Processed feature shapes:")
    logger.info(f"  AAtype: {processed_feature_dict['aatype'].shape}")
    logger.info(f"  Seq mask: {processed_feature_dict['seq_mask'].shape}")

    # Convert to tensors and move to device
    processed_feature_dict = tensor_tree_map(
        lambda x: torch.tensor(x).to(args.model_device),
        processed_feature_dict
    )

    # Remove recycling dimension - structure module expects [N] not [N, 1]
    if len(processed_feature_dict["aatype"].shape) == 2:
        processed_feature_dict["aatype"] = processed_feature_dict["aatype"].squeeze(-1)
    if len(processed_feature_dict["seq_mask"].shape) == 2:
        processed_feature_dict["seq_mask"] = processed_feature_dict["seq_mask"].squeeze(-1)

    logger.info(f"After removing recycling dimension:")
    logger.info(f"  AAtype: {processed_feature_dict['aatype'].shape}")
    logger.info(f"  Seq mask: {processed_feature_dict['seq_mask'].shape}")

    # Generate single representation if needed
    if evoformer_outputs['single'] is not None:
        single = evoformer_outputs['single'].to(args.model_device)
        logger.info(f"Using existing single representation: {single.shape}")
    else:
        logger.info("Generating single representation from MSA")
        single = generate_single_from_msa(evoformer_outputs['msa'], model, args.model_device)

    pair = evoformer_outputs['pair'].to(args.model_device)
    logger.info(f"Pair representation shape: {pair.shape}")

    # Remove batch dimension if present
    if len(single.shape) == 3 and single.shape[0] == 1:
        single = single.squeeze(0)
        logger.info(f"Removed batch dim from single: {single.shape}")
    if len(pair.shape) == 4 and pair.shape[0] == 1:
        pair = pair.squeeze(0)
        logger.info(f"Removed batch dim from pair: {pair.shape}")

    # Create outputs dict exactly like model.py does - include MSA if available
    outputs = {
        'single': single,
        'pair': pair,
    }

    # Add MSA if we have it (some structure module components might need it)
    if evoformer_outputs['msa'] is not None:
        msa = evoformer_outputs['msa'].to(args.model_device)
        if len(msa.shape) == 4 and msa.shape[0] == 1:
            msa = msa.squeeze(0)
        # Limit MSA size to avoid memory issues
        if msa.shape[0] > 512:
            msa = msa[:512]
        outputs['msa'] = msa[:, :processed_feature_dict["aatype"].shape[0], :]  # Match sequence length
        logger.info(f"Added MSA to outputs: {outputs['msa'].shape}")

    logger.info(f"Final tensor shapes before structure module:")
    logger.info(f"  Single: {outputs['single'].shape}")
    logger.info(f"  Pair: {outputs['pair'].shape}")
    logger.info(f"  AAtype: {processed_feature_dict['aatype'].shape}")
    logger.info(f"  Seq mask: {processed_feature_dict['seq_mask'].shape}")

    # Verify the sequence lengths match
    single_seq_len = outputs['single'].shape[0]
    pair_seq_len = outputs['pair'].shape[0]
    aatype_seq_len = processed_feature_dict['aatype'].shape[0]

    logger.info(f"Sequence lengths: single={single_seq_len}, pair={pair_seq_len}, aatype={aatype_seq_len}")

    if not (single_seq_len == pair_seq_len == aatype_seq_len):
        raise ValueError(
            f"Sequence length mismatch: single={single_seq_len}, pair={pair_seq_len}, aatype={aatype_seq_len}")

    # Check the channel dimensions
    logger.info(f"Channel dimensions:")
    logger.info(f"  Single channels: {outputs['single'].shape[-1]} (expected: {config.model.evoformer_stack.c_s})")
    logger.info(f"  Pair channels: {outputs['pair'].shape[-1]} (expected: {config.model.evoformer_stack.c_z})")

    # Run structure module exactly like the model does
    logger.info("Running structure module")
    try:
        with torch.no_grad():
            t = time.perf_counter()
            outputs["sm"] = model.structure_module(
                outputs,
                processed_feature_dict["aatype"],
                mask=processed_feature_dict["seq_mask"].to(dtype=single.dtype),
            )
            inference_time = time.perf_counter() - t
            logger.info(f"Structure module inference time: {inference_time:.2f}s")
    except Exception as e:
        logger.error(f"Structure module failed with error: {e}")
        logger.error(f"Error type: {type(e)}")
        raise

    # Add plddt from auxiliary heads if available, otherwise create dummy
    try:
        logger.info("Running auxiliary heads to get plddt...")
        with torch.no_grad():
            # Prepare the exact structure that aux_heads expects
            outputs_for_aux = {
                'single': outputs['single'],  # Keep as tensor
                'pair': outputs['pair'],  # Keep as tensor
                'sm': outputs['sm'],  # Keep as tensor
            }

            # Add MSA if available (some aux heads might need it)
            if 'msa' in outputs:
                outputs_for_aux['msa'] = outputs['msa']

            logger.info(f"Calling aux_heads with keys: {list(outputs_for_aux.keys())}")
            aux_output = model.aux_heads(outputs_for_aux)

        logger.info(f"Keys in aux_output: {list(aux_output.keys())}")

        # Add aux outputs to main outputs dict
        for key, value in aux_output.items():
            if torch.is_tensor(value):
                outputs[key] = value  # Keep as tensor for now
            else:
                outputs[key] = value

        if "plddt" in outputs:
            logger.info(f"Successfully generated plddt: {outputs['plddt'].shape}")

    except Exception as e:
        logger.error(f"Failed to run aux_heads: {e}")
        logger.error(f"This means no confidence scores will be available")
        raise  # Don't create dummy, just fail

    # Add the final processing that creates final_atom_positions, etc.
    # This is what run_pretrained_openfold.py does around lines 355-362
    from openfold.utils.feats import atom14_to_atom37

    # Create proper batch dict with all required fields for atom14_to_atom37
    batch_for_atom_conversion = {
        'aatype': processed_feature_dict['aatype']
    }

    # Add required residue mappings if not present
    if 'residx_atom14_to_atom37' not in processed_feature_dict:
        seq_len = len(processed_feature_dict['aatype'])

        # Create residue mappings based on amino acid types
        residx_atom14_to_atom37 = torch.zeros((seq_len, 14), dtype=torch.long)
        residx_atom37_to_atom14 = torch.zeros((seq_len, 37), dtype=torch.long)
        atom37_atom_exists = torch.zeros((seq_len, 37), dtype=torch.bool)

        for i, aa_idx in enumerate(processed_feature_dict['aatype']):
            if isinstance(aa_idx, torch.Tensor):
                aa_idx = aa_idx.item()
            if aa_idx < 20:  # Valid amino acid
                # Use residue constants to get proper mappings
                restype_atom14_to_atom37 = protein.residue_constants.restype_atom14_to_atom37[aa_idx]
                restype_atom37_to_atom14 = protein.residue_constants.restype_atom37_to_atom14[aa_idx]
                restype_atom37_mask = protein.residue_constants.restype_atom37_mask[aa_idx]

                residx_atom14_to_atom37[i] = torch.tensor(restype_atom14_to_atom37)
                residx_atom37_to_atom14[i] = torch.tensor(restype_atom37_to_atom14)
                atom37_atom_exists[i] = torch.tensor(restype_atom37_mask, dtype=torch.bool)

        batch_for_atom_conversion['residx_atom14_to_atom37'] = residx_atom14_to_atom37
        batch_for_atom_conversion['residx_atom37_to_atom14'] = residx_atom37_to_atom14
        batch_for_atom_conversion['atom37_atom_exists'] = atom37_atom_exists
    else:
        batch_for_atom_conversion['residx_atom14_to_atom37'] = processed_feature_dict['residx_atom14_to_atom37']
        batch_for_atom_conversion['residx_atom37_to_atom14'] = processed_feature_dict['residx_atom37_to_atom14']
        batch_for_atom_conversion['atom37_atom_exists'] = processed_feature_dict.get('atom37_atom_exists',
                                                                                     torch.ones((len(
                                                                                         processed_feature_dict[
                                                                                             'aatype']), 37),
                                                                                         dtype=torch.bool))

    # Add final_atom_positions (convert from atom14 to atom37 format)
    outputs["final_atom_positions"] = atom14_to_atom37(
        outputs["sm"]["positions"][-1], batch_for_atom_conversion
    )

    # Add final_atom_mask
    outputs["final_atom_mask"] = batch_for_atom_conversion['atom37_atom_exists']

    # Add final_affine_tensor
    outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

    # **CRITICAL FIX**: Remove ALL singleton dimensions from arrays before converting to numpy
    logger.info("Removing singleton dimensions from all outputs...")

    def remove_singleton_dims(x):
        if torch.is_tensor(x):
            # Keep squeezing until no more singleton dimensions
            original_shape = x.shape
            while x.dim() > 0 and 1 in x.shape:
                x = x.squeeze()
            if len(original_shape) != len(x.shape):
                logger.info(f"Squeezed tensor from {original_shape} to {x.shape}")
            return x
        return x

    # Apply to all outputs
    outputs = tensor_tree_map(remove_singleton_dims, outputs)
    processed_feature_dict = tensor_tree_map(remove_singleton_dims, processed_feature_dict)
    batch_for_atom_conversion = tensor_tree_map(remove_singleton_dims, batch_for_atom_conversion)

    logger.info(f"After removing singleton dimensions:")
    logger.info(f"  final_atom_positions: {outputs['final_atom_positions'].shape}")
    logger.info(f"  final_atom_mask: {outputs['final_atom_mask'].shape}")
    logger.info(f"  final_affine_tensor: {outputs['final_affine_tensor'].shape}")

    # Now convert everything to numpy safely
    def safe_to_numpy(x):
        if torch.is_tensor(x):
            return np.array(x.cpu())
        elif isinstance(x, np.ndarray):
            return x
        else:
            return x

    outputs = tensor_tree_map(safe_to_numpy, outputs)
    processed_feature_dict = tensor_tree_map(safe_to_numpy, processed_feature_dict)

    # Create unrelaxed protein exactly like run_pretrained_openfold.py
    unrelaxed_protein = prep_output(
        outputs,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        args.config_preset,
        args.multimer_ri_gap,
        args.subtract_plddt
    )

    # Save outputs
    output_name = f'{tag}_{args.config_preset}'

    unrelaxed_file_suffix = "_unrelaxed.cif" if args.cif_output else "_unrelaxed.pdb"
    unrelaxed_output_path = os.path.join(args.output_dir, f'{output_name}{unrelaxed_file_suffix}')

    with open(unrelaxed_output_path, 'w') as fp:
        if args.cif_output:
            fp.write(protein.to_modelcif(unrelaxed_protein))
        else:
            fp.write(protein.to_pdb(unrelaxed_protein))

    logger.info(f"Unrelaxed structure saved to: {unrelaxed_output_path}")

    # Save timing
    update_timings({tag: {"structure_module_inference": inference_time}},
                   os.path.join(args.output_dir, "timings.json"))

    # Relaxation
    if not args.skip_relaxation:
        relax_protein(config, args.model_device, unrelaxed_protein,
                      args.output_dir, output_name, args.cif_output)

    # Save raw outputs
    if args.save_outputs:
        output_dict_path = os.path.join(args.output_dir, f'{output_name}_output_dict.pkl')
        with open(output_dict_path, "wb") as fp:
            pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"✅ Successfully processed")


def main(args):
    """Main function."""

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = model_config(args.config_preset)

    # Set up data pipeline exactly like run_pretrained_openfold.py
    is_multimer = "multimer" in args.config_preset

    # Only initialize template searcher if we're NOT using precomputed alignments
    if args.use_precomputed_alignments is None:
        if is_multimer:
            template_searcher = hmmsearch.Hmmsearch(
                binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                database_path=args.pdb_seqres_database_path,
            )
        else:
            template_searcher = hhsearch.HHSearch(
                binary_path=args.hhsearch_binary_path,
                databases=[args.pdb70_database_path],
            )
    else:
        template_searcher = None

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=None,  # We don't need template featurizer for structure module only
    )

    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor,
        )

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    # Load model
    logger.info("Loading OpenFold model...")
    model_generator = load_models_from_command_line(
        config, args.model_device, args.openfold_checkpoint_path,
        args.jax_param_path, args.output_dir
    )
    model, _ = next(model_generator)

    # Process the protein
    process_protein(args, config, model, data_processor, feature_processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("fasta_dir", type=str)
    parser.add_argument("template_mmcif_dir", type=str)
    parser.add_argument("--pair_path", type=str, required=True,
                        help="Path to pair representation .pt file")
    parser.add_argument("--msa_path", type=str, required=True,
                        help="Path to MSA representation .pt file")
    parser.add_argument("--output_dir", type=str, default=os.getcwd())
    parser.add_argument("--model_device", type=str, default="cpu")
    parser.add_argument("--config_preset", type=str, default="model_1")
    parser.add_argument("--openfold_checkpoint_path", type=str, default=None)
    parser.add_argument("--jax_param_path", type=str, default=None)
    parser.add_argument("--use_precomputed_alignments", type=str, default=None)
    parser.add_argument("--skip_relaxation", action="store_true")
    parser.add_argument("--cif_output", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--subtract_plddt", action="store_true")
    parser.add_argument("--multimer_ri_gap", type=int, default=200)

    # Tool paths - only needed if not using precomputed alignments
    parser.add_argument("--hhsearch_binary_path", type=str, default="hhsearch")
    parser.add_argument("--hmmsearch_binary_path", type=str, default="hmmsearch")
    parser.add_argument("--hmmbuild_binary_path", type=str, default="hmmbuild")
    parser.add_argument("--pdb70_database_path", type=str, default="")
    parser.add_argument("--pdb_seqres_database_path", type=str, default="")

    args = parser.parse_args()

    # Set up default parameter path if none provided
    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if args.model_device.startswith("cuda") and not torch.cuda.is_available():
        args.model_device = "cpu"

    main(args)