import torch
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import re
import argparse


def natural_key(file):
    """Sort filenames by numeric index."""
    return int(re.findall(r"m_block_(\d+)\.pt", file)[0])


def plot_channel_deltas_for_recycle(recycle_path, channel, output_pdf, device):
    m_files = sorted([f for f in os.listdir(recycle_path) if f.startswith("m_block_")], key=natural_key)

    if not m_files:
        raise ValueError(f"{recycle_path}: No m_block_*.pt files found.")

    # Load and store all blocks
    m_blocks = [torch.load(os.path.join(recycle_path, f), map_location=device) for f in m_files]
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

    print(f"🔧 {os.path.basename(recycle_path)} — Global max ΔChannel {channel}: {global_max:.4f}")

    # === Second pass: plot diffs with fixed color scale ===
    with PdfPages(output_pdf) as pdf:
        for i, m in enumerate(m_blocks):
            cur_slice = m[..., :, channel]
            diff = (cur_slice - ref_slice).T  # Shape: (N_seq, N_res)

            fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
            ax.set_facecolor("white")

            # Move tensor to CPU before plotting
            im = ax.imshow(
                diff.cpu(),
                aspect='auto',
                cmap='coolwarm',
                vmin=-global_max,
                vmax=global_max,
                interpolation='nearest'
            )
            plt.colorbar(im, ax=ax, label=f"ΔChannel {channel} (from initial)")

            title = "Initial Input" if i == 0 else f"Layer {i - 1}" if i < len(m_blocks) - 1 else "Final Output"
            ax.set_title(f"{os.path.basename(recycle_path)} — {title} — ΔChannel {channel}")
            ax.set_xlabel("Residue Index (N_res)")
            ax.set_ylabel("Sequence Index (N_seq)")

            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved: {output_pdf}")


def plot_deltas_across_recycles(root_folder, channel, output_base_dir, device, pdb_id):
    os.makedirs(output_base_dir, exist_ok=True)

    recycle_folders = sorted([f for f in os.listdir(root_folder) if f.startswith("recycle_")])
    for recycle_name in recycle_folders:
        recycle_path = os.path.join(root_folder, recycle_name)
        output_pdf = os.path.join(output_base_dir, f"{pdb_id}_{recycle_name}_channel{channel}_m_input_deltas.pdf")
        plot_channel_deltas_for_recycle(recycle_path, channel, output_pdf, device)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize input deltas in MSA representations.')
    parser.add_argument('--pdb_id', type=str, default="1fme_A", help='PDB ID')
    parser.add_argument('--root_folder', type=str, help='Root folder path')
    parser.add_argument('--channel', type=int, default=0, help='Channel index to visualize')
    parser.add_argument('--output_dir', type=str, help='Output directory for PDF files')
    parser.add_argument('--device', type=str, default="cpu", help='Device for tensor operations')

    args = parser.parse_args()

    # Use provided paths or default to relative paths
    root_folder = args.root_folder
    if not root_folder:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct path relative to script location
        root_folder = os.path.join(script_dir, "..", "quick_inference_data", f"{args.pdb_id}_evoformer_blocks")

    output_dir = args.output_dir if args.output_dir else root_folder

    # Run plotting function
    plot_deltas_across_recycles(
        root_folder=root_folder,
        channel=args.channel,
        output_base_dir=output_dir,
        device=args.device,
        pdb_id=args.pdb_id
    )


if __name__ == "__main__":
    main()