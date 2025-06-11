#!/bin/bash


PDB_ID="1tpg_A"
PDB_PATH="/media/visitor/Extreme SSD/data/predictions/${PDB_ID}/predictions/${PDB_ID}_model_1_ptm_relaxed.pdb"
#PDB_PATH="/media/visitor/Extreme SSD/data/predictions/${PDB_ID}/predictions/${PDB_ID}_model_1_ptm_unrelaxed.pdb"
#PDB_PATH="/media/visitor/Extreme SSD/data/predictions/${PDB_ID}/1tpg.pdb"
PDB_PATH="/home/visitor/PycharmProjects/openFold/openfold/examples/monomer/sample_predictions/6KWC_1_model_1_ptm_relaxed.pdb"

PDB_PATH="/media/visitor/Extreme SSD/data/structure_predictions/1fv5_A/openfold_0recycles/predictions/1fv5_A_model_1_ptm_relaxed.pdb"
PDB_PATH="/media/visitor/Extreme SSD/data/structure_predictions/1fv5_A/openfold_deconstructed/1fv5_A_model_1_ptm_relaxed.pdb"
PDB_PATH="/media/visitor/Extreme SSD/data/structure_predictions/1fv5_A/neuralODE/1fv5_A_model_1_ptm_relaxed.pdb"
# Check if file exists
if [ ! -f "$PDB_PATH" ]; then
  echo "PDB file not found: $PDB_PATH"
  exit 1
fi

# Activate pymol_env
source ~/anaconda3/bin/activate pymol_env  # Adjust if you're using venv or another shell

# Run Python with PyMOL
python - <<EOF
import pymol
pymol.finish_launching()
pymol.cmd.load("$PDB_PATH")
pymol.cmd.zoom()
EOF
