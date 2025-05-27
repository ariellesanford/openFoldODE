#!/usr/bin/env python

import os
import torch

# === Configuration ===
DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/data"
BLOCK_INDEX = 1

def inspect_block(protein_dir, block_index):
    """
    Inspect m_block_{index}.pt and z_block_{index}.pt in the given protein directory.
    """
    m_path = os.path.join(protein_dir, "recycle_0", f"m_block_{block_index}.pt")
    z_path = os.path.join(protein_dir, "recycle_0", f"z_block_{block_index}.pt")

    if os.path.exists(m_path):
        try:
            m_tensor = torch.load(m_path)
            print(f"{os.path.basename(protein_dir)} - m_block_{block_index}.pt shape: {m_tensor.shape}")
        except Exception as e:
            print(f"Error loading {m_path}: {e}")
    else:
        print(f"{os.path.basename(protein_dir)} - m_block_{block_index}.pt not found.")

    if os.path.exists(z_path):
        try:
            z_tensor = torch.load(z_path)
            print(f"{os.path.basename(protein_dir)} - z_block_{block_index}.pt shape: {z_tensor.shape}")
        except Exception as e:
            print(f"Error loading {z_path}: {e}")
    else:
        print(f"{os.path.basename(protein_dir)} - z_block_{block_index}.pt not found.")

def inspect_all_proteins(data_dir, block_index):
    """
    Iterate over all protein folders in the data directory and inspect the specified block files.
    """
    for protein_dir in os.listdir(data_dir):
        protein_path = os.path.join(data_dir, protein_dir)
        if os.path.isdir(protein_path):
            inspect_block(protein_path, block_index)
            print("-" * 60)

if __name__ == "__main__":
    inspect_all_proteins(DATA_DIR, BLOCK_INDEX)
