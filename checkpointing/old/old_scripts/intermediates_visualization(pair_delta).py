import torch
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import re

# === EDIT THESE PARAMETERS ===
PDB_ID = "1sk3_A"
FOLDER = f"../checkpointing/monomers/predictions/{PDB_ID}_evoformer_blocks"
CHANNEL = 3
OUTPUT_PDF = f"{PDB_ID}_plots/{PDB_ID}_channel{CHANNEL}_pair_deltas.pdf"
# ==============================

def natural_key(file):
    """Sort filenames by numeric index."""
    return int(re.findall(r"m_block_(\d+)\.pt", file)[0])

def plot_evoformer_stepwise_deltas(folder, channel=0, output_pdf="evoformer_step_diffs.pdf"):
    # Create output folder if it doesn't exist
    output_dir = os.path.dirname(output_pdf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    m_files = sorted([f for f in os.listdir(folder) if f.startswith("m_block_")], key=natural_key)

    if len(m_files) < 2:
        raise ValueError("Need at least two blocks to compute stepwise deltas.")

    # Load and store all blocks
    m_blocks = [torch.load(os.path.join(folder, f)) for f in m_files]
    m_blocks = [m[0] if m.ndim == 4 else m for m in m_blocks]  # Remove batch dim if present

    # === First pass: calculate global max absolute difference across steps ===
    global_max = 0.0
    for i in range(1, len(m_blocks)):
        cur = m_blocks[i][..., :, channel]
        prev = m_blocks[i - 1][..., :, channel]
        diff = (cur - prev).abs().max().item()
        if diff > global_max:
            global_max = diff

    print(f"🔧 Global max stepwise delta for channel {channel}: {global_max:.4f}")

    # === Second pass: plot diffs between each pair of consecutive layers ===
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
            ax.set_title(f"{layer_label} — ΔChannel {channel}")
            ax.set_xlabel("Residue Index (N_res)")
            ax.set_ylabel("Sequence Index (N_seq)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ Stepwise ΔChannel plots saved to: {output_pdf}")

if __name__ == "__main__":
    plot_evoformer_stepwise_deltas(FOLDER, channel=CHANNEL, output_pdf=OUTPUT_PDF)
