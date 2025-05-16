import os
import torch
from openfold.config import model_config
from openfold.utils.script_utils import load_models_from_command_line
import time
import argparse


def run_single_evoformer_block(
    m_path: str,
    z_path: str,
    config, device, openfold_checkpoint_path, jax_param_path, output_dir
):
    """
    Load M, Z, msa_mask, and pair_mask tensors from disk and run them through a single Evoformer block.

    Args:
        m_path (str): Path to the MSA embedding tensor (.pt file)
        z_path (str): Path to the pairwise embedding tensor (.pt file)
        checkpoint_path (str): Path to OpenFold checkpoint
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
    idx = int(os.path.basename(args.m_path).split("_")[-1].split(".")[0])
    print(idx)

    # Add dummy batch dimensions if needed
    if m.ndim == 3:
        m = m.unsqueeze(0)
    if z.ndim == 3:
        z = z.unsqueeze(0)

    # Load masks from the same directory
    msa_mask_path = os.path.join(base_dir, "msa_mask.pt")
    pair_mask_path = os.path.join(base_dir, "pair_mask.pt")

    if os.path.exists(msa_mask_path) and os.path.exists(pair_mask_path):
        msa_mask = torch.load(msa_mask_path, map_location=device)
        pair_mask = torch.load(pair_mask_path, map_location=device)

        if msa_mask.ndim == 2:
            msa_mask = msa_mask.unsqueeze(0)
        if pair_mask.ndim == 2:
            pair_mask = pair_mask.unsqueeze(0)

        if msa_mask.shape != m.shape[:3]:
            raise ValueError(f"msa_mask shape {msa_mask.shape} does not match expected shape {m.shape[:3]}")
        if pair_mask.shape != z.shape[:3]:
            raise ValueError(f"pair_mask shape {pair_mask.shape} does not match expected shape {z.shape[:3]}")
    else:
        raise FileNotFoundError(f"msa_mask.pt or pair_mask.pt not found in {base_dir}")

    # Load model with pretrained weights
    config = model_config(config)
    model_generator = load_models_from_command_line(config, device, openfold_checkpoint_path, jax_param_path, output_dir)
    # model_generator = load_models_from_command_line(
    #     config,
    #     args.model_device,
    #     args.openfold_checkpoint_path,
    #     args.jax_param_path,
    #     args.output_dir)
    model, _ = next(model_generator)
    for model, output_dir in model_generator:
        print(f"Model: {model}")
        print(f"Output Directory: {output_dir}")

    # Extract the first Evoformer block
    block = model.evoformer.blocks[idx]

    # Debugging: Check parameter stats
    print("[DEBUG] EvoformerBlock parameter statistics:")
    for name, param in block.named_parameters():
        print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

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

    return m_out, z_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single Evoformer block on m, z, msa_mask, and pair_mask tensors")
    parser.add_argument("--m_path", required=True, help="Path to the m_block tensor (.pt)")
    parser.add_argument("--z_path", required=True, help="Path to the z_block tensor (.pt)")
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument("--config_preset", default="model_1_ptm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./evoformer_block_output")
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    args = parser.parse_args()
    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    print(f"[DEBUG] Loading checkpoint from: {args.jax_param_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Run single Evoformer block
    m_out, z_out = run_single_evoformer_block(
        args.m_path,
        args.z_path,
        args.config_preset,
        args.device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir)

    # Extract index from input filenames and increment for output
    m_idx = int(os.path.basename(args.m_path).split("_")[-1].split(".")[0])
    z_idx = int(os.path.basename(args.z_path).split("_")[-1].split(".")[0])

    m_out_path = os.path.join(args.output_dir, f"m_block_{m_idx + 1}.pt")
    z_out_path = os.path.join(args.output_dir, f"z_block_{z_idx + 1}.pt")

    torch.save(m_out, m_out_path)
    torch.save(z_out, z_out_path)

    end_time = time.time()
    print(f"✅ Saved m_out to {m_out_path}")
    print(f"✅ Saved z_out to {z_out_path}")
    print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")
