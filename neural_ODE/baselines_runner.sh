#!/bin/bash
set -e


# Memory search with validation splits (recommended)
python loss_baselines.py \
  --data_dir "/media/visitor/Extreme SSD/data/complete_blocks" \
  --splits_dir "data_splits/full" \
  --mode memory_search \
  --max_proteins 20

# Baseline analysis only
#python loss_baselines.py \
#  --data_dir "/media/visitor/Extreme SSD/data/complete_blocks" \
#  --splits_dir "data_splits/full" \
#  --mode baselines_only \
#  --reduced_cluster_size 64 \
#  --max_residues 200