import torch
import torch.nn.functional as F
import os
import numpy as np


def debug_training_issues(data_dir, protein_id="4cue_A"):
    """
    Debug what's causing the massive loss values
    """
    print("=== Training Loss Debugging ===\n")

    # Load some data to check scales
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

    try:
        m_init = torch.load(os.path.join(protein_dir, "m_block_0.pt"), map_location='cpu')
        z_init = torch.load(os.path.join(protein_dir, "z_block_0.pt"), map_location='cpu')
        m_target = torch.load(os.path.join(protein_dir, "m_block_1.pt"), map_location='cpu')
        z_target = torch.load(os.path.join(protein_dir, "z_block_1.pt"), map_location='cpu')

        # Remove batch dims
        if m_init.dim() == 4:
            m_init = m_init.squeeze(0)
        if z_init.dim() == 4:
            z_init = z_init.squeeze(0)
        if m_target.dim() == 4:
            m_target = m_target.squeeze(0)
        if z_target.dim() == 4:
            z_target = z_target.squeeze(0)

        print(f"Data shapes:")
        print(f"  MSA: {m_init.shape} -> {m_target.shape}")
        print(f"  Pair: {z_init.shape} -> {z_target.shape}")

        # Check data scales
        print(f"\nData value ranges:")
        print(f"  MSA init: [{m_init.min():.3f}, {m_init.max():.3f}], std: {m_init.std():.3f}")
        print(f"  MSA target: [{m_target.min():.3f}, {m_target.max():.3f}], std: {m_target.std():.3f}")
        print(f"  Pair init: [{z_init.min():.3f}, {z_init.max():.3f}], std: {z_init.std():.3f}")
        print(f"  Pair target: [{z_target.min():.3f}, {z_target.max():.3f}], std: {z_target.std():.3f}")

        # Test different loss computations
        print(f"\n=== Loss Computation Tests ===")

        # 1. Identity loss (should be 0)
        identity_loss_m = F.mse_loss(m_init, m_init)
        identity_loss_z = F.mse_loss(z_init, z_init)
        print(f"1. Identity loss (should be ~0): MSA={identity_loss_m:.6f}, Pair={identity_loss_z:.6f}")

        # 2. Zero prediction loss
        zero_m = torch.zeros_like(m_target)
        zero_z = torch.zeros_like(z_target)
        zero_loss_m = F.mse_loss(zero_m, m_target)
        zero_loss_z = F.mse_loss(zero_z, z_target)
        total_zero_loss = zero_loss_m + zero_loss_z
        print(f"2. Zero prediction loss: MSA={zero_loss_m:.3f}, Pair={zero_loss_z:.3f}, Total={total_zero_loss:.3f}")

        # 3. Mean prediction loss
        mean_m = torch.full_like(m_target, m_target.mean())
        mean_z = torch.full_like(z_target, z_target.mean())
        mean_loss_m = F.mse_loss(mean_m, m_target)
        mean_loss_z = F.mse_loss(mean_z, z_target)
        total_mean_loss = mean_loss_m + mean_loss_z
        print(f"3. Mean prediction loss: MSA={mean_loss_m:.3f}, Pair={mean_loss_z:.3f}, Total={total_mean_loss:.3f}")

        # 4. Random prediction loss (this could be huge)
        random_m = torch.randn_like(m_target) * m_target.std() + m_target.mean()
        random_z = torch.randn_like(z_target) * z_target.std() + z_target.mean()
        random_loss_m = F.mse_loss(random_m, m_target)
        random_loss_z = F.mse_loss(random_z, z_target)
        total_random_loss = random_loss_m + random_loss_z
        print(
            f"4. Random prediction loss: MSA={random_loss_m:.3f}, Pair={random_loss_z:.3f}, Total={total_random_loss:.3f}")

        # 5. Test what happens with exploded predictions
        exploded_m = m_target * 10  # 10x larger predictions
        exploded_z = z_target * 10
        exploded_loss_m = F.mse_loss(exploded_m, m_target)
        exploded_loss_z = F.mse_loss(exploded_z, z_target)
        total_exploded_loss = exploded_loss_m + exploded_loss_z
        print(
            f"5. 10x exploded predictions: MSA={exploded_loss_m:.3f}, Pair={exploded_loss_z:.3f}, Total={total_exploded_loss:.3f}")

        # Compare to your actual training loss
        print(f"\n=== Training Loss Analysis ===")
        your_loss = 6208  # Your 4cue_A loss
        print(f"Your training loss: {your_loss:.1f}")
        print(f"vs Zero baseline: {your_loss / total_zero_loss:.1f}x worse")
        print(f"vs Mean baseline: {your_loss / total_mean_loss:.1f}x worse")
        print(f"vs Random baseline: {your_loss / total_random_loss:.1f}x worse")

        if your_loss > total_exploded_loss:
            print(f"🚨 CRITICAL: Your loss is worse than 10x exploded predictions!")
            print(f"   This suggests massive numerical instability")

        # Check for potential causes
        print(f"\n=== Potential Causes ===")

        if your_loss > 100 * total_zero_loss:
            print("❌ EXPLODING GRADIENTS: Loss is 100x worse than baseline")
            print("   Solutions: Lower learning rate, gradient clipping, simpler ODE")

        if total_random_loss > 1000:
            print("⚠️  HIGH VARIANCE DATA: Random predictions give huge loss")
            print("   Solutions: Data normalization, loss scaling")

        if z_target.std() > 50:
            print("⚠️  LARGE PAIR VALUES: Pair representation has very large values")
            print("   Solutions: Normalize pair data, separate loss scales")

        # Suggest fixes
        print(f"\n=== Immediate Fixes to Try ===")
        print("1. LOWER LEARNING RATE: Try 1e-5 or 1e-6 instead of 1e-3")
        print("2. GRADIENT CLIPPING: Add torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
        print("3. SIMPLER INTEGRATION: Use fewer time points (5-10 instead of 25+)")
        print("4. DATA NORMALIZATION: Normalize MSA and pair data to [-1, 1] range")
        print("5. LOSS SCALING: Use separate loss weights for MSA vs pair")

    except Exception as e:
        print(f"Error loading data: {e}")


def suggest_immediate_training_fixes():
    """
    Provide immediate actionable fixes for the training
    """
    print("\n" + "=" * 60)
    print("IMMEDIATE ACTION PLAN")
    print("=" * 60)

    print("\n🔥 CRITICAL FIXES (Do These First):")
    print("1. Reduce learning rate from 1e-3 to 1e-5")
    print("2. Reduce time points from 25 to 5")
    print("3. Add gradient clipping")

    print("\n📝 Code changes needed in train_evoformer_ode.py:")
    print("""
# Change learning rate:
learning_rate = 1e-5  # Instead of 1e-3

# Change time points:
num_points = 5  # Instead of 25

# Add gradient clipping in train_step:
if USE_AMP:
    scaler.scale(batch_loss).backward()
    scaler.unscale_(optimizer)  # Unscale before clipping
    torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
else:
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
    optimizer.step()
""")

    print("\n🧪 Testing strategy:")
    print("1. Test on ONE protein first")
    print("2. Run for just 1-2 epochs")
    print("3. Watch for loss to drop below 1000")
    print("4. If still exploding, reduce learning rate further")


if __name__ == "__main__":
    data_dir = "/home/visitor/PycharmProjects/openFold/neural_ODE/data/quick_inference_data"
    debug_training_issues(data_dir)
    suggest_immediate_training_fixes()