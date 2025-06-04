#!/bin/bash
set -e

#python loss_baselines.py \
#  --data_dir "/media/visitor/Extreme SSD/data/complete_blocks" \
#  --splits_dir "data_splits/full" \
#  --mode baselines_only \
#  --num_val_proteins 10 \
#  --reduced_cluster_size 64

# Mode 2: Hyperparameter search
python loss_baselines.py \
  --data_dir "/media/visitor/Extreme SSD/data/complete_blocks" \
  --splits_dir "data_splits/full" \
  --mode hyperparameter_search \
  --num_train_proteins 20 \
  --num_val_proteins 6 \
  --epochs 18