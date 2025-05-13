import torch
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import re

# === EDIT THESE PARAMETERS ===
PDB_ID = "1fme_A"
ROOT_FOLDER = f"../checkpointing/monomers/predictions_noRecycles/{PDB_ID}_evoformer_blocks"
CHANNEL = 0
OUTPUT_BASE_DIR = f"plots_noRecycles/{PDB_ID}_plots_m"

ROOT_FOLDER = '/home/visitor/PycharmProjects/openFold/neural_ODE/data/4cue_A_evoformer_blocks'
OUTPUT_BASE_DIR = f"{ROOT_FOLDER}"
# ==============================

def natural_key(file):
    """Sort filenames by numeric index."""
    return int(re.findall(r"m_block_(\d+)\.pt", file)[0])

def plot_stepwise_deltas_for_recycle(recycle_path, channel, output_pdf):
    m_files = sorted([f for f in os.listdir(recycle_path) if f.startswith("m_block_")], key=natural_key)

    if len(m_files) < 2:
        raise ValueError(f"{recycle_path}: Need at least two blocks to compute stepwise deltas.")

    # Load and store all blocks
    m_blocks = [torch.load(os.path.join(recycle_path, f)) for f in m_files]
    m_blocks = [m[0] if m.ndim == 4 else m for m in m_blocks]  # Remove batch dim if present

    # === First pass: calculate global max absolute difference across steps ===
    global_max = 0.0
    for i in range(1, len(m_blocks)):
        cur = m_blocks[i][..., :, channel]
        prev = m_blocks[i - 1][..., :, channel]
        diff = (cur - prev).abs().max().item()
        if diff > global_max:
            global_max = diff

    print(f"🔧 Recycle {os.path.basename(recycle_path)} — Global max ΔChannel {channel}: {global_max:.4f}")

    # === Second pass: plot and save to PDF ===
    with PdfPages(output_pdf) as pdf:
        for i in range(1, len(m_blocks)):
            cur = m_blocks[i][..., :, channel]
            prev = m_blocks[i - 1][..., :, channel]
            diff = (cur - prev).T  # Transpose to (N_seq, N_res)

            fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
            ax.set_facecolor("white")

            im = ax.imshow(
                diff,
                aspect='auto',
                cmap='coolwarm',
                vmin=-global_max,
                vmax=global_max,
                interpolation='nearest'
            )
            plt.colorbar(im, ax=ax, label=f"ΔChannel {channel} (Layer {i} - Layer {i - 1})")

            layer_label = "Initial → Layer 0" if i == 1 else f"Layer {i - 1} → {i}" if i < len(m_blocks) - 1 else f"Layer {i - 1} → Final"
            ax.set_title(f"{os.path.basename(recycle_path)} — {layer_label} — ΔChannel {channel}")
            ax.set_xlabel("Residue Index (N_res)")
            ax.set_ylabel("Sequence Index (N_seq)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved: {output_pdf}")

def plot_deltas_across_recycles(root_folder, channel, output_base_dir):
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    recycle_folders = sorted([f for f in os.listdir(root_folder) if f.startswith("recycle_")])
    for recycle_name in recycle_folders:
        recycle_path = os.path.join(root_folder, recycle_name)
        output_pdf = os.path.join(output_base_dir, f"{PDB_ID}_{recycle_name}_channel{channel}_pair_deltas.pdf")
        plot_stepwise_deltas_for_recycle(recycle_path, channel, output_pdf)

if __name__ == "__main__":
    plot_deltas_across_recycles(ROOT_FOLDER, channel=CHANNEL, output_base_dir=OUTPUT_BASE_DIR)
