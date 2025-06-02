# Add this debugging script to diagnose the huge loss issue
# Run this BEFORE training to understand your data

import os
import torch
import torch.nn.functional as F


def diagnose_data_and_loss(data_dir, protein_limit=2):
    """
    Comprehensive diagnosis of why losses are 47k+
    """
    print("🔬 COMPREHENSIVE DATA DIAGNOSIS")
    print("=" * 50)

    # Find protein directories
    protein_dirs = []
    for name in os.listdir(data_dir):
        if name.endswith('_evoformer_blocks'):
            protein_id = name.replace('_evoformer_blocks', '')
            protein_dirs.append(protein_id)

    if not protein_dirs:
        print("❌ No protein directories found!")
        return

    # Limit proteins for faster diagnosis
    proteins_to_check = protein_dirs[:protein_limit]
    print(f"🧬 Checking {len(proteins_to_check)} proteins: {proteins_to_check}")

    for protein_id in proteins_to_check:
        print(f"\n--- PROTEIN: {protein_id} ---")

        try:
            # Check what blocks exist
            protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
            block_files = [f for f in os.listdir(protein_dir) if f.startswith('m_block_') and f.endswith('.pt')]
            block_numbers = sorted([int(f.split('_')[2].split('.')[0]) for f in block_files])

            print(f"📁 Available blocks: {block_numbers[:10]}{'...' if len(block_numbers) > 10 else ''}")

            # Load initial and a few other blocks
            blocks_to_check = [0, 1, 2] if len(block_numbers) >= 3 else block_numbers[:2]

            for block_idx in blocks_to_check:
                if block_idx in block_numbers:
                    m_path = os.path.join(protein_dir, f"m_block_{block_idx}.pt")
                    z_path = os.path.join(protein_dir, f"z_block_{block_idx}.pt")

                    m = torch.load(m_path, map_location='cpu')
                    z = torch.load(z_path, map_location='cpu')

                    # Handle batch dimensions
                    if m.dim() == 4:
                        m = m.squeeze(0)
                    if z.dim() == 4:
                        z = z.squeeze(0)

                    print(f"\n  📊 Block {block_idx}:")
                    print(f"    M shape: {tuple(m.shape)}")
                    print(f"    Z shape: {tuple(z.shape)}")
                    print(f"    M range: [{m.min():.3f}, {m.max():.3f}], std: {m.std():.3f}")
                    print(f"    Z range: [{z.min():.3f}, {z.max():.3f}], std: {z.std():.3f}")

                    # Check for problematic values
                    if m.abs().max() > 20:
                        print(f"    ⚠️  LARGE M VALUES: max abs = {m.abs().max():.1f}")
                    if z.abs().max() > 20:
                        print(f"    ⚠️  LARGE Z VALUES: max abs = {z.abs().max():.1f}")

                    # Check for NaN/Inf
                    if torch.isnan(m).any() or torch.isnan(z).any():
                        print(f"    ❌ NaN values detected!")
                    if torch.isinf(m).any() or torch.isinf(z).any():
                        print(f"    ❌ Inf values detected!")

            # NOW TEST ACTUAL LOSS COMPUTATION
            if len(block_numbers) >= 2:
                print(f"\n  🧪 LOSS COMPUTATION TEST (blocks 0→1):")

                # Load blocks 0 and 1
                m0 = torch.load(os.path.join(protein_dir, "m_block_0.pt"), map_location='cpu').squeeze(0)
                z0 = torch.load(os.path.join(protein_dir, "z_block_0.pt"), map_location='cpu').squeeze(0)
                m1 = torch.load(os.path.join(protein_dir, "m_block_1.pt"), map_location='cpu').squeeze(0)
                z1 = torch.load(os.path.join(protein_dir, "z_block_1.pt"), map_location='cpu').squeeze(0)

                # Reduce cluster size to match training
                m0 = m0[:64]  # Assuming reduced_cluster_size=64
                m1 = m1[:64]

                # Test different loss scenarios
                print(f"    Target M1: [{m1.min():.3f}, {m1.max():.3f}], std: {m1.std():.3f}")
                print(f"    Target Z1: [{z1.min():.3f}, {z1.max():.3f}], std: {z1.std():.3f}")

                # 1. Identity loss (predicting block 0 as block 1)
                identity_msa_loss = F.mse_loss(m0, m1)
                identity_pair_loss = F.mse_loss(z0, z1)
                identity_total = identity_msa_loss + identity_pair_loss
                print(
                    f"    Identity loss (m0→m1): MSA={identity_msa_loss:.1f}, Pair={identity_pair_loss:.1f}, Total={identity_total:.1f}")

                # 2. Zero prediction loss
                zero_msa_loss = F.mse_loss(torch.zeros_like(m1), m1)
                zero_pair_loss = F.mse_loss(torch.zeros_like(z1), z1)
                zero_total = zero_msa_loss + zero_pair_loss
                print(
                    f"    Zero prediction loss: MSA={zero_msa_loss:.1f}, Pair={zero_pair_loss:.1f}, Total={zero_total:.1f}")

                # 3. Random prediction loss
                random_m = torch.randn_like(m1) * m1.std()
                random_z = torch.randn_like(z1) * z1.std()
                random_msa_loss = F.mse_loss(random_m, m1)
                random_pair_loss = F.mse_loss(random_z, z1)
                random_total = random_msa_loss + random_pair_loss
                print(
                    f"    Random prediction loss: MSA={random_msa_loss:.1f}, Pair={random_pair_loss:.1f}, Total={random_total:.1f}")

                # 4. Check if your 47k loss makes sense
                your_loss = 47500  # Your reported loss
                print(f"\n    📈 YOUR LOSS ANALYSIS:")
                print(f"    Your loss: {your_loss:.1f}")
                print(f"    vs Identity: {your_loss / identity_total:.1f}x worse")
                print(f"    vs Zero: {your_loss / zero_total:.1f}x worse")
                print(f"    vs Random: {your_loss / random_total:.1f}x worse")

                if your_loss > zero_total:
                    print(f"    🚨 CRITICAL: Your model is worse than predicting zeros!")
                elif your_loss > random_total:
                    print(f"    ⚠️  WARNING: Your model is worse than random!")
                elif your_loss > identity_total * 2:
                    print(f"    ⚠️  Your model is much worse than identity mapping")
                else:
                    print(f"    ✅ Loss magnitude seems reasonable")

        except Exception as e:
            print(f"❌ Error processing {protein_id}: {e}")

    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    print(f"Run this diagnostic and look for:")
    print(f"1. Very large values (>50) in M or Z blocks")
    print(f"2. Your loss being worse than zero/random baselines")
    print(f"3. NaN or Inf values in the data")


def test_model_prediction(data_dir, protein_id=None):
    """
    Test what your actual model predicts vs targets
    """
    print(f"\n🤖 MODEL PREDICTION TEST")
    print("=" * 40)

    # Find a protein to test
    if protein_id is None:
        for name in os.listdir(data_dir):
            if name.endswith('_evoformer_blocks'):
                protein_id = name.replace('_evoformer_blocks', '')
                break

    if not protein_id:
        print("❌ No protein found for testing")
        return

    print(f"Testing with protein: {protein_id}")

    # You would need to load your actual trained model here
    # For now, let's simulate what might be happening

    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

    # Load initial state
    m0 = torch.load(os.path.join(protein_dir, "m_block_0.pt"), map_location='cpu').squeeze(0)[:64]
    z0 = torch.load(os.path.join(protein_dir, "z_block_0.pt"), map_location='cpu').squeeze(0)
    m1_target = torch.load(os.path.join(protein_dir, "m_block_1.pt"), map_location='cpu').squeeze(0)[:64]
    z1_target = torch.load(os.path.join(protein_dir, "z_block_1.pt"), map_location='cpu').squeeze(0)

    # Simulate what an untrained model might predict (random changes)
    m1_pred = m0 + torch.randn_like(m0) * 0.1  # Small random changes
    z1_pred = z0 + torch.randn_like(z0) * 0.1

    print(f"\nSimulated prediction test:")
    print(f"Target M1: [{m1_target.min():.3f}, {m1_target.max():.3f}]")
    print(f"Pred M1:   [{m1_pred.min():.3f}, {m1_pred.max():.3f}]")
    print(f"Target Z1: [{z1_target.min():.3f}, {z1_target.max():.3f}]")
    print(f"Pred Z1:   [{z1_pred.min():.3f}, {z1_pred.max():.3f}]")

    # Compute loss
    msa_loss = F.mse_loss(m1_pred, m1_target)
    pair_loss = F.mse_loss(z1_pred, z1_target)
    total_loss = msa_loss + pair_loss

    print(f"\nSimulated loss: MSA={msa_loss:.1f}, Pair={pair_loss:.1f}, Total={total_loss:.1f}")


if __name__ == "__main__":
    # Replace with your actual data directory
    DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/data/complete_blocks"

    print("🚀 Starting comprehensive diagnosis...")
    print("This will help us understand why your loss is 47k+")

    diagnose_data_and_loss(DATA_DIR, protein_limit=2)
    test_model_prediction(DATA_DIR)

    print(f"\n💡 Next steps:")
    print(f"1. Run this diagnostic first")
    print(f"2. Look for the specific issues it identifies")
    print(f"3. We'll fix the root cause based on the output")