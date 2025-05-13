import torch
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import re

# === EDIT THESE PARAMETERS ===
PDB_ID = "4d0f_A"
FOLDER = f"../checkpointing/monomers/predictions/{PDB_ID}_evoformer_blocks"
CHANNEL = 2
OUTPUT_PDF = f"{PDB_ID}_plots/{PDB_ID}_channel{CHANNEL}_input_deltas.pdf"
# ==============================

def natural_key(file):
    """Sort filenames by numeric index."""
    return int(re.findall(r"m_block_(\d+)\.pt", file)[0])


def plot_evoformer_channel_deltas(folder, channel=0, output_pdf="evoformer_diffs.pdf"):
    # Create output folder if it doesn't exist
    output_dir = os.path.dirname(output_pdf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    m_files = sorted([f for f in os.listdir(folder) if f.startswith("m_block_")], key=natural_key)

    if not m_files:
        raise ValueError("No m_block_*.pt files found in folder.")

    # Load and store all blocks
    m_blocks = [torch.load(os.path.join(folder, f)) for f in m_files]
    m_blocks = [m[0] if m.ndim == 4 else m for m in m_blocks]  # Remove batch dim if present

    # Extract the reference (initial) slice
    ref_slice = m_blocks[0][..., :, channel]

    # === First pass: calculate global max absolute difference ===
    global_max = 0.0
    for m in m_blocks:
        cur_slice = m[..., :, channel]
        diff = (cur_slice - ref_slice).abs().max().item()
        if diff > global_max:
            global_max = diff

    print(f"🔧 Global max delta for channel {channel}: {global_max:.4f}")

    # === Second pass: plot diffs with fixed color scale ===
    with PdfPages(output_pdf) as pdf:
        for i, m in enumerate(m_blocks):
            cur_slice = m[..., :, channel]
            diff = (cur_slice - ref_slice).T  # Shape: (N_seq, N_res)

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

            title = "Initial Input" if i == 0 else f"Layer {i - 1}" if i < len(m_blocks) - 1 else "Final Output"
            ax.set_title(f"{title} — ΔChannel {channel}")
            ax.set_xlabel("Residue Index (N_res)")
            ax.set_ylabel("Sequence Index (N_seq)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ ΔChannel plots saved to: {output_pdf}")

if __name__ == "__main__":
    plot_evoformer_channel_deltas(FOLDER, channel=CHANNEL, output_pdf=OUTPUT_PDF)
