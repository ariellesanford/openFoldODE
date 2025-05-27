import torch
import os
import numpy as np


def analyze_data_statistics(data_dir, protein_id):
    """
    Analyze the scale and variance of your actual data to understand what loss values mean
    """
    protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

    # Load a few blocks to understand data scale
    m_blocks = []
    z_blocks = []

    for i in range(min(5, 49)):  # Load first 5 blocks
        m_path = os.path.join(protein_dir, f"m_block_{i}.pt")
        z_path = os.path.join(protein_dir, f"z_block_{i}.pt")

        if os.path.exists(m_path) and os.path.exists(z_path):
            m = torch.load(m_path, map_location='cpu')  # Force CPU loading
            z = torch.load(z_path, map_location='cpu')  # Force CPU loading

            # Remove batch dim if present
            if m.dim() == 4:
                m = m.squeeze(0)
            if z.dim() == 4:
                z = z.squeeze(0)

            m_blocks.append(m)
            z_blocks.append(z)

    if not m_blocks:
        print("No data found!")
        return

    # Calculate statistics
    print(f"=== Data Statistics for {protein_id} ===")

    # MSA (m) statistics
    m_values = torch.cat([m.flatten() for m in m_blocks])
    print(f"MSA Data:")
    print(f"  Mean: {m_values.mean().item():.6f}")
    print(f"  Std:  {m_values.std().item():.6f}")
    print(f"  Min:  {m_values.min().item():.6f}")
    print(f"  Max:  {m_values.max().item():.6f}")
    print(f"  Range: {(m_values.max() - m_values.min()).item():.6f}")

    # Pair (z) statistics
    z_values = torch.cat([z.flatten() for z in z_blocks])
    print(f"Pair Data:")
    print(f"  Mean: {z_values.mean().item():.6f}")
    print(f"  Std:  {z_values.std().item():.6f}")
    print(f"  Min:  {z_values.min().item():.6f}")
    print(f"  Max:  {z_values.max().item():.6f}")
    print(f"  Range: {(z_values.max() - z_values.min()).item():.6f}")

    # Calculate baseline losses
    print(f"\n=== Baseline Loss Analysis ===")

    # Zero prediction baseline
    zero_mse_m = torch.mean(m_values ** 2).item()
    zero_mse_z = torch.mean(z_values ** 2).item()
    zero_baseline = zero_mse_m + zero_mse_z
    print(f"Zero Prediction Baseline: {zero_baseline:.6f}")
    print(f"  (MSE if you always predict 0)")

    # Mean prediction baseline
    m_mean = m_values.mean()
    z_mean = z_values.mean()
    mean_mse_m = torch.mean((m_values - m_mean) ** 2).item()
    mean_mse_z = torch.mean((z_values - z_mean) ** 2).item()
    mean_baseline = mean_mse_m + mean_mse_z
    print(f"Mean Prediction Baseline: {mean_baseline:.6f}")
    print(f"  (MSE if you always predict the mean)")

    # Random prediction baseline (using std as noise level)
    random_noise_m = torch.randn_like(m_values) * m_values.std()
    random_noise_z = torch.randn_like(z_values) * z_values.std()
    random_mse_m = torch.mean((m_values - random_noise_m) ** 2).item()
    random_mse_z = torch.mean((z_values - random_noise_z) ** 2).item()
    random_baseline = random_mse_m + random_mse_z
    print(f"Random Noise Baseline: {random_baseline:.6f}")
    print(f"  (MSE against random predictions)")

    # Block-to-block differences (natural variation)
    if len(m_blocks) > 1:
        block_diffs_m = []
        block_diffs_z = []
        for i in range(len(m_blocks) - 1):
            diff_m = torch.mean((m_blocks[i] - m_blocks[i + 1]) ** 2).item()
            diff_z = torch.mean((z_blocks[i] - z_blocks[i + 1]) ** 2).item()
            block_diffs_m.append(diff_m)
            block_diffs_z.append(diff_z)

        natural_var_m = np.mean(block_diffs_m)
        natural_var_z = np.mean(block_diffs_z)
        natural_variation = natural_var_m + natural_var_z
        print(f"Natural Block-to-Block Variation: {natural_variation:.6f}")
        print(f"  (MSE between consecutive Evoformer blocks)")

    print(f"\n=== Loss Interpretation Guidelines ===")
    if 'natural_variation' in locals():
        print(f"Excellent:  < {natural_variation:.2f} (better than natural variation)")
        print(f"Good:       < {mean_baseline / 10:.2f} (much better than mean prediction)")
        print(f"Decent:     < {mean_baseline / 2:.2f} (significantly better than mean)")
        print(f"Poor:       > {mean_baseline:.2f} (worse than predicting mean)")
        print(f"Terrible:   > {zero_baseline:.2f} (worse than predicting zero)")
    else:
        print(f"Good:       < {mean_baseline / 10:.2f} (much better than mean prediction)")
        print(f"Decent:     < {mean_baseline / 2:.2f} (significantly better than mean)")
        print(f"Poor:       > {mean_baseline:.2f} (worse than predicting mean)")
        print(f"Terrible:   > {zero_baseline:.2f} (worse than predicting zero)")


def compare_current_losses_to_baselines(loss_values, baselines):
    """
    Compare your current loss values to the computed baselines
    """
    print(f"\n=== Your Current Losses Analysis ===")

    for loss_val in loss_values:
        print(f"\nLoss: {loss_val:.2f}")
        if loss_val < baselines['natural']:
            print("  ✅ EXCELLENT - Better than natural Evoformer variation!")
        elif loss_val < baselines['mean'] / 10:
            print("  🎯 VERY GOOD - Much better than mean prediction")
        elif loss_val < baselines['mean'] / 2:
            print("  ✔️ GOOD - Significantly better than mean prediction")
        elif loss_val < baselines['mean']:
            print("  ⚠️ DECENT - Better than mean but could improve")
        elif loss_val < baselines['zero']:
            print("  ❌ POOR - Worse than predicting the mean")
        else:
            print("  💥 TERRIBLE - Worse than predicting zero")


if __name__ == "__main__":
    # Analyze your data - UPDATE THESE PATHS
    data_dir = "/home/visitor/PycharmProjects/openFold/neural_ODE/data/quick_inference_data"
    protein_id = "1fme_A"  # Or whichever protein you want to analyze

    analyze_data_statistics(data_dir, protein_id)

    # Your recent loss values from the reports
    recent_losses = [18.14, 39.10, 40.80, 54.31, 73.82]  # Examples from your reports

    # You'll need to run the analysis first to get these baselines
    # baselines = {'natural': X, 'mean': Y, 'zero': Z}  # Fill in after running analysis