import torch
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import re

# === EDIT THESE PARAMETERS ===
PDB_ID = "4cue_A"
ROOT_FOLDER = f"../checkpointing/monomers/predictions_debugging/{PDB_ID}_evoformer_blocks"
CHANNEL = 0
OUTPUT_BASE_DIR = f"{PDB_ID}_plots_z"
# ==============================

def natural_key(file):
    """Sort filenames by numeric index."""
    return int(re.findall(r"z_block_(\d+)\.pt", file)[0])

def plot_channel_deltas_for_recycle(recycle_path, channel, output_pdf):
    z_files = sorted([f for f in os.listdir(recycle_path) if f.startswith("z_block_")], key=natural_key)

    if not z_files:
        raise ValueError(f"{recycle_path}: No z_block_*.pt files found.")

    # Load and store all blocks
    z_blocks = [torch.load(os.path.join(recycle_path, f)) for f in z_files]
    z_blocks = [z[0] if z.ndim == 4 else z for z in z_blocks]  # Remove batch dim if present

    # Extract the reference (initial) slice
    ref_slice = z_blocks[0][..., channel]  # [N_res, N_res]

    # === First pass: calculate global max absolute difference ===
    global_max = 0.0
    for z in z_blocks:
        cur_slice = z[..., channel]
        diff = (cur_slice - ref_slice).abs().max().item()
        if diff > global_max:
            global_max = diff

    print(f"🔧 {os.path.basename(recycle_path)} — Global max ΔChannel {channel}: {global_max:.4f}")

    # === Second pass: plot diffs with fixed color scale ===
    with PdfPages(output_pdf) as pdf:
        for i, z in enumerate(z_blocks):
            cur_slice = z[..., channel]
            diff = (cur_slice - ref_slice).T  # Shape: (N_res, N_res)

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
            plt.colorbar(im, ax=ax, label=f"ΔChannel {channel} (from initial)")

            title = "Initial Input" if i == 0 else f"Layer {i - 1}" if i < len(z_blocks) - 1 else "Final Output"
            ax.set_title(f"{os.path.basename(recycle_path)} — {title} — ΔChannel {channel}")
            ax.set_xlabel("Residue Index (N_res)")
            ax.set_ylabel("Residue Index (N_res)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved: {output_pdf}")

def plot_deltas_across_recycles(root_folder, channel, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)

    recycle_folders = sorted([f for f in os.listdir(root_folder) if f.startswith("recycle_")])
    for recycle_name in recycle_folders:
        recycle_path = os.path.join(root_folder, recycle_name)
        output_pdf = os.path.join(output_base_dir, f"{PDB_ID}_{recycle_name}_channel{channel}_z_input_deltas.pdf")
        plot_channel_deltas_for_recycle(recycle_path, channel, output_pdf)

if __name__ == "__main__":
    plot_deltas_across_recycles(ROOT_FOLDER, channel=CHANNEL, output_base_dir=OUTPUT_BASE_DIR)
