#!/usr/bin/env python

import os
import torch
import re

# === Configuration ===
#DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/mini_data/training/blocks"
DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/mini_data/training/blocks"
RECYCLE_SUBDIR = "recycle_0"

def inspect_all_blocks(protein_dir):
    """
    Print shapes of all m_block_*.pt and z_block_*.pt in the given protein directory.
    """
    recycle_dir = os.path.join(protein_dir, RECYCLE_SUBDIR)
    if not os.path.isdir(recycle_dir):
        print(f"{os.path.basename(protein_dir)} - ❌ No recycle_0 directory")
        return

    print(f"\n📂 Inspecting: {os.path.basename(protein_dir)}")

    for fname in sorted(os.listdir(recycle_dir)):
        if re.match(r"^(m|z)_block_\d+\.pt$", fname):
            full_path = os.path.join(recycle_dir, fname)
            try:
                tensor = torch.load(full_path)
                print(f"  ✅ {fname}: {tuple(tensor.shape)}")
            except Exception as e:
                print(f"  ❌ Error loading {fname}: {e}")

def inspect_all_proteins(data_dir):
    """
    Iterate over all protein folders in the data directory and inspect all block files.
    """
    for protein_dir in sorted(os.listdir(data_dir)):
        protein_path = os.path.join(data_dir, protein_dir)
        if os.path.isdir(protein_path):
            inspect_all_blocks(protein_path)
            print("-" * 60)

if __name__ == "__main__":
    inspect_all_proteins(DATA_DIR)
