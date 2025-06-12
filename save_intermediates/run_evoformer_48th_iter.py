import os
import torch
from openfold.config import model_config
from openfold.utils.script_utils import load_models_from_command_line
import time
import argparse
import gc


def run_48_evoformer_blocks(
        m_path: str,
        z_path: str,
        config, device, openfold_checkpoint_path, jax_param_path, output_dir
):
    """
    Load M and Z tensors from block 0 and run them through all 48 Evoformer blocks.
    Only saves the final 48th block output.
    """
    # Extract the directory path for masks
    base_dir = os.path.dirname(m_path)

    # Load initial tensors (block 0)
    m = torch.load(m_path, map_location=device)
    z = torch.load(z_path, map_location=device)

    # Add dummy batch dimensions if needed
    if m.ndim == 3:
        m = m.unsqueeze(0)
    if z.ndim == 3:
        z = z.unsqueeze(0)

    # Load masks from the same directory
    msa_mask_path = os.path.join(base_dir, "msa_mask.pt")
    pair_mask_path = os.path.join(base_dir, "pair_mask.pt")

    msa_mask = torch.load(msa_mask_path, map_location=device)
    pair_mask = torch.load(pair_mask_path, map_location=device)

    if msa_mask.ndim == 2:
        msa_mask = msa_mask.unsqueeze(0)
    if pair_mask.ndim == 2:
        pair_mask = pair_mask.unsqueeze(0)

    # Load model with pretrained weights
    config = model_config(config)
    model_generator = load_models_from_command_line(config, device, openfold_checkpoint_path, jax_param_path,
                                                    output_dir)

    model, _ = next(model_generator)
    print(f"Model loaded successfully")

    # Run all 48 blocks sequentially
    print(f"Running all 48 Evoformer blocks...")

    with torch.no_grad():
        for block_idx in range(48):
            print(f"  Processing block {block_idx + 1}/48...")

            block = model.evoformer.blocks[block_idx]

            m, z = block(
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

            # Clean up intermediate tensors periodically
            if (block_idx + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"✅ All 48 blocks completed")

    # Clean up model and other references
    del block, model
    del model_generator
    del msa_mask, pair_mask

    # Force garbage collection and GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"🧹 GPU memory cleanup completed")

    return m, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all 48 Evoformer blocks and save only the 48th output")
    parser.add_argument("--m_path", required=True, help="Path to the m_block_0 tensor (.pt)")
    parser.add_argument("--z_path", required=True, help="Path to the z_block_0 tensor (.pt)")
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="Path to OpenFold checkpoint"
    )
    parser.add_argument("--config_preset", default="model_1_ptm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./evoformer_48th_output")
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="Path to JAX model parameters"
    )

    args = parser.parse_args()

    # Find project root and parameters file
    project_root = os.path.dirname(os.path.abspath(__file__))

    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join(project_root, "openfold", "resources", "params",
                                           f"params_{args.config_preset}.npz")

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Run all 48 Evoformer blocks
    m_out, z_out = run_48_evoformer_blocks(
        args.m_path,
        args.z_path,
        args.config_preset,
        args.device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir)

    # Save only the 48th block output
    m_out_path = os.path.join(args.output_dir, "m_block_48.pt")
    z_out_path = os.path.join(args.output_dir, "z_block_48.pt")

    torch.save(m_out, m_out_path)
    torch.save(z_out, z_out_path)

    # Final cleanup
    del m_out, z_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    end_time = time.time()
    print(f"✅ Saved 48th block to {m_out_path} and {z_out_path}")
    print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
    print(f"💾 GPU memory cleaned up")