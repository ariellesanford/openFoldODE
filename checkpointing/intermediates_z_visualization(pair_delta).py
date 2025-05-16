import torch
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import re

# === EDIT THESE PARAMETERS ===
PDB_ID = "1fme_A"
ROOT_FOLDER = f"../checkpointing/monomers/predictions_debugging/{PDB_ID}_evoformer_blocks"
CHANNEL = 0
OUTPUT_BASE_DIR = f"{PDB_ID}_plots_z"
# ==============================

def natural_key(file):
    """Sort filenames by numeric index."""
    return int(re.findall(r"z_block_(\d+)\.pt", file)[0])

def plot_stepwise_deltas_for_recycle(recycle_path, channel, output_pdf):
    z_files = sorted([f for f in os.listdir(recycle_path) if f.startswith("z_block_")], key=natural_key)

    if len(z_files) < 2:
        raise ValueError(f"{recycle_path}: Need at least two blocks to compute stepwise deltas.")

    # Load and store all blocks
    z_blocks = [torch.load(os.path.join(recycle_path, f)) for f in z_files]
    z_blocks = [z[0] if z.ndim == 4 else z for z in z_blocks]  # Remove batch dim if present

    # === First pass: calculate global max absolute difference across steps ===
    global_max = 0.0
    for i in range(1, len(z_blocks)):
        cur = z_blocks[i][..., channel]
        prev = z_blocks[i - 1][..., channel]
        diff = (cur - prev).abs().max().item()
        if diff > global_max:
            global_max = diff

    print(f"🔧 Recycle {os.path.basename(recycle_path)} — Global max ΔChannel {channel}: {global_max:.4f}")

    # === Second pass: plot and save to PDF ===
    with PdfPages(output_pdf) as pdf:
        for i in range(1, len(z_blocks)):
            cur = z_blocks[i][..., channel]
            prev = z_blocks[i - 1][..., channel]
            diff = (cur - prev).T  # Shape: (N_res, N_res)

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

            layer_label = "Initial → Layer 0" if i == 1 else f"Layer {i - 1} → {i}" if i < len(z_blocks) - 1 else f"Layer {i - 1} → Final"
            ax.set_title(f"{os.path.basename(recycle_path)} — {layer_label} — ΔChannel {channel}")
            ax.set_xlabel("Residue Index (N_res)")
            ax.set_ylabel("Residue Index (N_res)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved: {output_pdf}")

def plot_deltas_across_recycles(root_folder, channel, output_base_dir):
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    recycle_folders = sorted([f for f in os.listdir(root_folder) if f.startswith("recycle_")])
    for recycle_name in recycle_folders:
        recycle_path = os.path.join(root_folder, recycle_name)
        output_pdf = os.path.join(output_base_dir, f"{PDB_ID}_{recycle_name}_channel{channel}_z_pair_deltas.pdf")
        plot_stepwise_deltas_for_recycle(recycle_path, channel, output_pdf)

if __name__ == "__main__":
    plot_deltas_across_recycles(ROOT_FOLDER, channel=CHANNEL, output_base_dir=OUTPUT_BASE_DIR)
