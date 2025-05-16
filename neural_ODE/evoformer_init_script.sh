#!/bin/bash
set -e


# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/evoformer_init

# === Define variables ===
PDB_ID="4cue_A"
INPUT_FASTA_DIR="../checkpointing/monomers/fasta_dir/${PDB_ID}"
TEMPLATE_MMCIF_DIR="../openfold/data/pdb70_mmcif/mmcif_files"
OUTPUT_DIR="../neural_ODE/data"
PRECOMPUTED_ALIGNMENTS="../checkpointing/monomers/RODA/alignments"


# === Run OpenFold ===
/home/visitor/anaconda3/envs/openfold_env/bin/python run_evoformer_init.py \
  "${INPUT_FASTA_DIR}" \
  "${TEMPLATE_MMCIF_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_precomputed_alignments "${PRECOMPUTED_ALIGNMENTS}" \
  --config_preset model_1_ptm \
  --model_device "cuda:0" \
  --save_intermediates \
  --save_outputs
