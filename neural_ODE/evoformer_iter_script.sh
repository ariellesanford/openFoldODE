#!/bin/bash
set -e

#chmod +x run_openfold_inference.sh
#./run_openfold_inference.sh


# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/evoformer_init

# === Define variables ===
INPUT_FASTA_DIR="../checkpointing/monomers/fasta_dir"
TEMPLATE_MMCIF_DIR="../openfold/data/pdb70_mmcif/mmcif_files"
OUTPUT_DIR="../checkpointing/evoformer_inits"
PRECOMPUTED_ALIGNMENTS="../checkpointing/monomers/RODA/alignments"

#INPUT_FASTA_DIR="examples/monomer/fasta_dir"
#TEMPLATE_MMCIF_DIR="data/pdb70_mmcif/mmcif_files"
#OUTPUT_DIR="examples/monomer/predictions"
#PRECOMPUTED_ALIGNMENTS="examples/monomer/alignments"

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

#  --experiment_config_json "/home/visitor/PycharmProjects/openFold/custom_config.json" \

