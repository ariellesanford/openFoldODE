#!/bin/bash

# Run basic testing on all test proteins
python test_model.py \
  --model_path outputs/adjoint_training_20250605_112329_final_model.pt \
  --data_dir "/media/visitor/Extreme SSD/data/complete_blocks" \
  --splits_dir "data_splits/mini"
