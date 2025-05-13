import os
import torch
from openfold.config import model_config
from openfold.model.evoformer import EvoformerStack

def run_single_evoformer_block(
    m_path: str,
    z_path: str,
    config_preset: str = "model_1_ptm",
    device: str = "cuda:0"
):
    """
    Load M, Z, msa_mask, and pair_mask tensors from disk and run them through a single Evoformer block.

    Args:
        m_path (str): Path to the MSA embedding tensor (.pt file)
        z_path (str): Path to the pairwise embedding tensor (.pt file)
        config_preset (str): Config preset name from OpenFold
        device (str): Device to run inference on

    Returns:
        m_out (torch.Tensor), z_out (torch.Tensor)
    """
    # Extract the directory path
    base_dir = os.path.dirname(m_path)

    # Load tensors
    m = torch.load(m_path, map_location=device)
    z = torch.load(z_path, map_location=device)

    # Add dummy batch dimensions if needed
    if m.ndim == 3:
        m = m.unsqueeze(0)  # [1, N_seq, N_res, C_m]
    if z.ndim == 3:
        z = z.unsqueeze(0)  # [1, N_res, N_res, C_z]

    # Load masks from the same directory
    msa_mask_path = os.path.join(base_dir, "msa_mask.pt")
    pair_mask_path = os.path.join(base_dir, "pair_mask.pt")



    if os.path.exists(msa_mask_path) and os.path.exists(pair_mask_path):
        msa_mask = torch.load(msa_mask_path, map_location=device)
        pair_mask = torch.load(pair_mask_path, map_location=device)

        # Add batch dimension to masks if needed
        if msa_mask.ndim == 2:  # [N_seq, N_res]
            msa_mask = msa_mask.unsqueeze(0)  # [1, N_seq, N_res]
        elif msa_mask.ndim == 3 and msa_mask.shape[0] != 1:
            raise ValueError(f"Unexpected msa_mask shape: {msa_mask.shape}. Expected [1, N_seq, N_res]")

        if pair_mask.ndim == 2:  # [N_res, N_res]
            pair_mask = pair_mask.unsqueeze(0)  # [1, N_res, N_res]
        elif pair_mask.ndim == 3 and pair_mask.shape[0] != 1:
            raise ValueError(f"Unexpected pair_mask shape: {pair_mask.shape}. Expected [1, N_res, N_res]")

        # Check mask dimensions
        if msa_mask.shape != m.shape[:3]:
            raise ValueError(f"msa_mask shape {msa_mask.shape} does not match expected shape {m.shape[:3]}")

        if pair_mask.shape != z.shape[:3]:
            raise ValueError(f"pair_mask shape {pair_mask.shape} does not match expected shape {z.shape[:3]}")
    else:
        raise FileNotFoundError(f"msa_mask.pt or pair_mask.pt not found in {base_dir}")

    # Debugging: Check mask content
    print(f"[DEBUG] msa_mask: min={msa_mask.min().item()}, max={msa_mask.max().item()}, mean={msa_mask.mean().item()}")
    print(
        f"[DEBUG] pair_mask: min={pair_mask.min().item()}, max={pair_mask.max().item()}, mean={pair_mask.mean().item()}")

    # Debugging: Check initial m and z content
    print(f"[DEBUG] Initial m: min={m.min().item()}, max={m.max().item()}, mean={m.mean().item()}")
    print(f"[DEBUG] Initial z: min={z.min().item()}, max={z.max().item()}, mean={z.mean().item()}")


    # Load model config
    config = model_config(config_preset).model

    # Initialize Evoformer stack
    evoformer = EvoformerStack(
        **config["evoformer_stack"]
    ).to(device)

    # Use the first Evoformer block
    block = evoformer.blocks[0]

    # Debugging before Evoformer block execution
    print(f"[DEBUG] Before Evoformer block - m: min={m.min().item()}, max={m.max().item()}, mean={m.mean().item()}")
    print(f"[DEBUG] Before Evoformer block - z: min={z.min().item()}, max={z.max().item()}, mean={z.mean().item()}")

    # Run single block
    with torch.no_grad():
        m_out, z_out = block(
            m, z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=64,
            use_deepspeed_evo_attention=False,
            use_lma=False,
            use_flash=False,
            inplace_safe=True,
            _mask_trans=True
        )

    # Debugging after Evoformer block execution
    print(f"[DEBUG] After Evoformer block - m_out: min={m_out.min().item()}, max={m_out.max().item()}, mean={m_out.mean().item()}")
    print(f"[DEBUG] After Evoformer block - z_out: min={z_out.min().item()}, max={z_out.max().item()}")


    return m_out, z_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single Evoformer block on m, z, msa_mask, and pair_mask tensors")
    parser.add_argument("--m_path", required=True, help="Path to the m_block tensor (.pt)")
    parser.add_argument("--z_path", required=True, help="Path to the z_block tensor (.pt)")
    parser.add_argument("--config_preset", default="model_1_ptm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./evoformer_block_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run single Evoformer block
    m_out, z_out = run_single_evoformer_block(
        args.m_path,
        args.z_path,
        config_preset=args.config_preset,
        device=args.device
    )

    # Extract index from input filenames and increment for output
    m_idx = int(os.path.basename(args.m_path).split("_")[-1].split(".")[0])
    z_idx = int(os.path.basename(args.z_path).split("_")[-1].split(".")[0])

    m_out_path = os.path.join(args.output_dir, f"m_block_{m_idx + 1}.pt")
    z_out_path = os.path.join(args.output_dir, f"z_block_{z_idx + 1}.pt")

    torch.save(m_out, m_out_path)
    torch.save(z_out, z_out_path)

    print(f"✅ Saved m_out to {m_out_path}")
    print(f"✅ Saved z_out to {z_out_path}")
