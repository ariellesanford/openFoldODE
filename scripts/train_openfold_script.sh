#!/bin/bash
set -e

#chmod +x run_openfold_inference.sh
#./run_openfold_inference.sh

# see the first protein in test data for date
# === Change to the desired working directory ===
cd /home/visitor/PycharmProjects/openFold/openfold/

# === Define variables ===
INPUT_FASTA_DIR="examples/monomer/fasta_dir"
TEMPLATE_MMCIF_DIR="data/pdb70_mmcif/mmcif_files"
OUTPUT_DIR="/home/visitor/PycharmProjects/openFold/retrained_openfold_predictions"
PRECOMPUTED_ALIGNMENTS="examples/monomer/alignments"
DATA_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"

/home/visitor/anaconda3/envs/openfold_env/bin/python train_openfold.py \
  "${DATA_DIR}/training_data(75%)/pdb_data/mmcifs" \
  "${DATA_DIR}/training_data(75%)/alignment_data/alignments" \
  "${TEMPLATE_MMCIF_DIR}" \
  "${OUTPUT_DIR}" \
  "2024-04-10" \
  --train_chain_data_cache_path "${DATA_DIR}/training_data(75%)/pdb_data/data_caches/chain_data_cache.json" \
  --template_release_dates_cache_path "${DATA_DIR}/training_data(75%)/pdb_data/data_caches/mmcif_cache.json" \
  --config_preset initial_training \
  --seed 42 \
  --obsolete_pdbs_file_path "${DATA_DIR}/training_data(75%)/pdb_data/obsolete.dat" \
  --num_nodes 1 \
  --gpus 1 \
  --val_data_dir "${DATA_DIR}/validation_data(15%)/pdb_data/mmcifs" \
  --val_alignment_dir "${DATA_DIR}/validation_data(15%)/alignment_data/alignments"

#/home/visitor/anaconda3/envs/openfold_env/bin/python train_openfold.py \
#  "${DATA_DIR}/training_data(75%)/pdb_data/mmcifs" \
#  "${DATA_DIR}/training_data(75%)/alignment_data/alignments" \
#  "${TEMPLATE_MMCIF_DIR}" "${OUTPUT_DIR}" \
#  --max_template_date 2021-10-10 \
#  --train_chain_data_cache_path "${DATA_DIR}/training_data(75%)/pdb_data/data_caches/chain_data_cache.json" \
#  --template_release_dates_cache_path "${DATA_DIR}/training_data(75%)/pdb_data/data_caches/mmcif_cache.json" \
#	--config_preset initial_training \
#  --seed 42 \
#  --obsolete_pdbs_file_path "${DATA_DIR}/training_data(75%)/pdb_data/obsolete.dat" \
#  --num_nodes 1 \
#  --gpus 1 \
#  --num_workers 4 \
#  --val_data_dir "${DATA_DIR}/validation_data(15%)/pdb_data/mmcifs" \
#  --val_alignment_dir "${DATA_DIR}/validation_data(15%)/alignment_data/alignments"