import os
import torch
import argparse
from openfold.config import model_config
from openfold.model.evoformer import EvoformerBlock

def run_single_evoformer_block(
    m_path: str,
    z_path: str,
    config_preset: str = "model_1_ptm",
    device: str = "cuda:0"
):
    """
    Load M, Z, msa_mask, and pair_mask tensors from disk and run them through a single Evoformer block.
    """
    base_dir = os.path.dirname(m_path)

    # Load tensors
    m = torch.load(m_path, map_location=device)
    z = torch.load(z_path, map_location=device)

    if m.ndim == 3:
        m = m.unsqueeze(0)
    if z.ndim == 3:
        z = z.unsqueeze(0)

    # Load masks
    msa_mask = torch.load(os.path.join(base_dir, "msa_mask.pt"), map_location=device)
    pair_mask = torch.load(os.path.join(base_dir, "pair_mask.pt"), map_location=device)

    if msa_mask.ndim == 2:
        msa_mask = msa_mask.unsqueeze(0)
    if pair_mask.ndim == 2:
        pair_mask = pair_mask.unsqueeze(0)

    if msa_mask.shape != m.shape[:3]:
        raise ValueError(f"msa_mask shape {msa_mask.shape} does not match expected {m.shape[:3]}")
    if pair_mask.shape != z.shape[:3]:
        raise ValueError(f"pair_mask shape {pair_mask.shape} does not match expected {z.shape[:3]}")

    # Print debug info
    print(f"[DEBUG] msa_mask: min={msa_mask.min()}, max={msa_mask.max()}, mean={msa_mask.float().mean()}")
    print(f"[DEBUG] pair_mask: min={pair_mask.min()}, max={pair_mask.max()}, mean={pair_mask.float().mean()}")
    print(f"[DEBUG] Initial m: min={m.min()}, max={m.max()}, mean={m.mean()}")
    print(f"[DEBUG] Initial z: min={z.min()}, max={z.max()}, mean={z.mean()}")

    # Load model config
    config = model_config(config_preset).model["evoformer_stack"]

    block = EvoformerBlock(
        c_m=config["c_m"],
        c_z=config["c_z"],
        c_hidden_msa_att=config["c_hidden_msa_att"],
        c_hidden_opm=config["c_hidden_opm"],
        c_hidden_mul=config["c_hidden_mul"],
        c_hidden_pair_att=config["c_hidden_pair_att"],
        no_heads_msa=config["no_heads_msa"],
        no_heads_pair=config["no_heads_pair"],
        transition_n=config["transition_n"],
        msa_dropout=config["msa_dropout"],
        pair_dropout=config["pair_dropout"],
        no_column_attention=config["no_column_attention"],
        opm_first=config["opm_first"],
        fuse_projection_weights=config["fuse_projection_weights"],
        inf=config["inf"],
        eps=config["eps"],
    ).to(device)

    # # Load model config
    # config = model_config(config_preset).model
    #
    # # Initialize only one EvoformerBlock
    # block = EvoformerBlock(
    #     **config["evoformer_stack"],
    # ).to(device)

    block.eval()  # Add this

    # DEBUG: Check parameter stats
    print("[DEBUG] Checking EvoformerBlock parameters:")
    for name, param in block.named_parameters():
        print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")

    print(f"[DEBUG] Initialized Evoformer block: {block}")

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

    print(f"[DEBUG] After Evoformer block - m_out: min={m_out.min()}, max={m_out.max()}, mean={m_out.mean()}")
    print(f"[DEBUG] After Evoformer block - z_out: min={z_out.min()}, max={z_out.max()}, mean={z_out.mean()}")

    return m_out, z_out

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", required=True)
    parser.add_argument("--z_path", required=True)
    parser.add_argument("--config_preset", default="model_1_ptm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./evoformer_block_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    m_out, z_out = run_single_evoformer_block(
        args.m_path,
        args.z_path,
        config_preset=args.config_preset,
        device=args.device
    )
    end = time.time()

    idx = int(os.path.basename(args.m_path).split("_")[-1].split(".")[0])
    m_out_path = os.path.join(args.output_dir, f"m_block_{idx + 1}.pt")
    z_out_path = os.path.join(args.output_dir, f"z_block_{idx + 1}.pt")

    torch.save(m_out, m_out_path)
    torch.save(z_out, z_out_path)

    print(f"✅ Saved m_out to {m_out_path}")
    print(f"✅ Saved z_out to {z_out_path}")
    print(f"⏱️ Time taken: {end - start:.2f} seconds")
