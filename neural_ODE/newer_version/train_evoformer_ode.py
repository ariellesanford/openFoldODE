import os
import gc
import torch
import argparse
import sys
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc  # Import from existing module
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


# Function to get project root directory
def get_project_root():
    """Get the path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir


# === Parse command line arguments ===
parser = argparse.ArgumentParser(description='Train Evoformer ODE with configurable memory optimizations')

parser.add_argument('--memory_split_size', type=int, default=128, help='Memory split size (MB)')
parser.add_argument('--use_amp', action='store_true', default=False, help='Use Automatic Mixed Precision')
parser.add_argument('--no-use_amp', dest='use_amp', action='store_false', help='Disable AMP')
parser.add_argument('--use_checkpoint', action='store_true', default=False, help='Use gradient checkpointing')
parser.add_argument('--no-use_checkpoint', dest='use_checkpoint', action='store_false', help='Disable checkpointing')
parser.add_argument('--gradient_accumulation', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--chunk_size', type=int, default=10, help='Chunk size for integration (0 disables)')
parser.add_argument('--reduced_precision_integration', action='store_true', default=False, help='Reduced precision ODE')
parser.add_argument('--no-reduced_precision_integration', dest='reduced_precision_integration', action='store_false')
parser.add_argument('--clean_memory', action='store_true', default=False, help='Clean memory aggressively')
parser.add_argument('--no-clean_memory', dest='clean_memory', action='store_false')
parser.add_argument('--reduced_cluster_size', type=int, default=64, help='Max cluster size')
parser.add_argument('--reduced_hidden_dim', type=int, default=128, help='Hidden dimension')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_time_points', type=int, default=25, help='Number of time points')
parser.add_argument('--integrator', choices=['dopri5', 'rk4', 'euler'], default='dopri5', help='ODE integrator')
parser.add_argument('--batch_size', type=int, default=5, help='Time step batch size')
parser.add_argument('--use_fast_ode', action='store_true', default=False, help='Use fast ODE implementation')
parser.add_argument('--no-use_fast_ode', dest='use_fast_ode', action='store_false')
parser.add_argument('--monitor_memory', action='store_true', default=False, help='Monitor memory')
parser.add_argument('--no-monitor_memory', dest='monitor_memory', action='store_false')
parser.add_argument('--cpu-only', action='store_true', default=False, help='Force CPU mode')
parser.add_argument('--no-cpu-only', dest='cpu_only', action='store_false')
parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
parser.add_argument('--test-configs', action='store_true', default=False)
parser.add_argument('--test-single-step', action='store_true', default=False)
parser.add_argument('--test-protein', type=str, default=None)
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')

args = parser.parse_args()

# Set memory config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.memory_split_size}"

PROJECT_ROOT = get_project_root()

# Device setup
if args.cpu_only:
    device = "cpu"
    args.use_amp = False
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        args.use_amp = False


def clear_memory():
    if args.clean_memory:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


# Trigger CUDA init
if device == "cuda":
    _ = torch.tensor([0.0], device=device)

clear_memory()

# Config
c_m = 256
c_z = 128
hidden_dim = args.reduced_hidden_dim
learning_rate = args.learning_rate

# Data dirs
DATA_DIR = args.data_dir if args.data_dir else os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = args.output_dir if args.output_dir else PROJECT_ROOT

USE_AMP = args.use_amp and device == "cuda"
scaler = GradScaler(enabled=USE_AMP)


# === NORMALIZATION FUNCTIONS ===
def compute_normalization_stats(data_dir, num_proteins=5):
    """
    Compute global normalization statistics from a sample of proteins
    """
    print("🔢 Computing normalization statistics...")

    all_m_values = []
    all_z_values = []

    # Sample proteins to compute stats
    protein_dirs = []
    for name in os.listdir(data_dir):
        if name.endswith('_evoformer_blocks'):
            protein_id = name.replace('_evoformer_blocks', '')
            protein_dirs.append(protein_id)

    sample_proteins = protein_dirs[:num_proteins]

    for protein_id in sample_proteins:
        protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")

        # Sample a few blocks from each protein
        for block_idx in [0, 1, 2, 10, 20]:
            m_path = os.path.join(protein_dir, f"m_block_{block_idx}.pt")
            z_path = os.path.join(protein_dir, f"z_block_{block_idx}.pt")

            if os.path.exists(m_path) and os.path.exists(z_path):
                m = torch.load(m_path, map_location='cpu').squeeze(0)
                z = torch.load(z_path, map_location='cpu').squeeze(0)

                all_m_values.append(m.flatten())
                all_z_values.append(z.flatten())

    # Compute global statistics
    all_m = torch.cat(all_m_values)
    all_z = torch.cat(all_z_values)

    m_mean = all_m.mean()
    m_std = all_m.std()
    z_mean = all_z.mean()
    z_std = all_z.std()

    stats = {
        'm_mean': m_mean.item(),
        'm_std': m_std.item(),
        'z_mean': z_mean.item(),
        'z_std': z_std.item()
    }

    print(f"📊 Normalization stats computed:")
    print(f"   M: mean={stats['m_mean']:.3f}, std={stats['m_std']:.3f}")
    print(f"   Z: mean={stats['z_mean']:.3f}, std={stats['z_std']:.3f}")

    return stats


def normalize_tensor(tensor, mean, std):
    """Normalize tensor to zero mean, unit variance"""
    return (tensor - mean) / (std + 1e-8)


def denormalize_tensor(tensor, mean, std):
    """Denormalize tensor back to original scale"""
    return tensor * (std + 1e-8) + mean


def get_dataset(input_dir):
    datasets = []
    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)
        if (os.path.isdir(full_path) and name.endswith('_evoformer_blocks') and
                os.path.isdir(os.path.join(full_path, 'recycle_0'))):
            protein_id = name.replace('_evoformer_blocks', '')
            datasets.append(protein_id)
    return datasets


def load_initial_input_normalized(protein_id, norm_stats):
    """
    Load and normalize initial M and Z tensors
    """
    protein_dir = os.path.join(DATA_DIR, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, "m_block_0.pt")
    z_path = os.path.join(protein_dir, "z_block_0.pt")

    m_init = torch.load(m_path, map_location=device)
    z_init = torch.load(z_path, map_location=device)

    if m_init.dim() == 4 and m_init.size(0) == 1:
        m_init = m_init.squeeze(0)
    if z_init.dim() == 4 and z_init.size(0) == 1:
        z_init = z_init.squeeze(0)

    # Reduce cluster size
    m_init = m_init[:args.reduced_cluster_size]

    # NORMALIZE THE DATA
    m_init = normalize_tensor(m_init, norm_stats['m_mean'], norm_stats['m_std'])
    z_init = normalize_tensor(z_init, norm_stats['z_mean'], norm_stats['z_std'])

    return m_init, z_init


def load_block_normalized(protein_id, block_index, norm_stats):
    """
    Load and normalize M and Z tensors for a specific block
    """
    protein_dir = os.path.join(DATA_DIR, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, f"m_block_{block_index}.pt")
    z_path = os.path.join(protein_dir, f"z_block_{block_index}.pt")

    if not os.path.exists(m_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Block {block_index} not found for {protein_id}")

    m = torch.load(m_path, map_location=device)
    z = torch.load(z_path, map_location=device)

    m = m.squeeze(0)
    z = z.squeeze(0)
    m = m[:args.reduced_cluster_size]

    # NORMALIZE THE DATA
    m = normalize_tensor(m, norm_stats['m_mean'], norm_stats['m_std'])
    z = normalize_tensor(z, norm_stats['z_mean'], norm_stats['z_std'])

    return m, z


# Keep original functions for testing
def load_initial_input(protein_id):
    """Original function for testing normalization"""
    protein_dir = os.path.join(DATA_DIR, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, "m_block_0.pt")
    z_path = os.path.join(protein_dir, "z_block_0.pt")

    m_init = torch.load(m_path, map_location=device)
    z_init = torch.load(z_path, map_location=device)

    if m_init.dim() == 4 and m_init.size(0) == 1:
        m_init = m_init.squeeze(0)
    if z_init.dim() == 4 and z_init.size(0) == 1:
        z_init = z_init.squeeze(0)

    m_init = m_init[:args.reduced_cluster_size]
    return m_init, z_init

class CheckpointedEvoformerODEFunc(torch.nn.Module):
    def __init__(self, ode_func, use_checkpoint=True):
        super().__init__()
        self.ode_func = ode_func
        self.use_checkpoint = use_checkpoint

    def forward(self, t, state):
        if self.use_checkpoint:
            return checkpoint(self.ode_func.forward, t, state, use_reentrant=False)
        else:
            return self.ode_func(t, state)


# Initialize model
if args.use_fast_ode:
    from evoformer_ode import EvoformerODEFuncFast as OdeFunction
else:
    from evoformer_ode import EvoformerODEFunc as OdeFunction

ode_func = OdeFunction(c_m, c_z, hidden_dim).to(device)
checkpointed_ode_func = CheckpointedEvoformerODEFunc(ode_func, use_checkpoint=args.use_checkpoint)
optimizer = optim.Adam(ode_func.parameters(), lr=learning_rate)

num_points = args.num_time_points
t_grid_full = torch.linspace(0, 1, num_points).to(device)


# PROBLEM: Loss varies dramatically based on:
# 1. Number of time steps (25 vs 48)
# 2. Number of proteins (1 vs 4 vs 100)
# 3. Protein size (28 residues vs 115 residues vs 500)
# 4. Cluster size (32 vs 64 vs 128 sequences)

def analyze_loss_scaling_factors(data_dir):
    """
    Analyze how loss scales with different protein properties
    """
    print("🔍 ANALYZING LOSS SCALING FACTORS...")

    # Get dataset info
    dataset = get_dataset(data_dir)

    protein_info = []
    for protein_id in dataset:
        try:
            protein_dir = os.path.join(data_dir, f"{protein_id}_evoformer_blocks", "recycle_0")
            m = torch.load(os.path.join(protein_dir, "m_block_0.pt"), map_location='cpu').squeeze(0)
            z = torch.load(os.path.join(protein_dir, "z_block_0.pt"), map_location='cpu').squeeze(0)

            num_sequences = m.shape[0]  # e.g., 516
            num_residues = m.shape[1]  # e.g., 28 or 115
            m_elements = m.numel()  # Total M tensor size
            z_elements = z.numel()  # Total Z tensor size

            protein_info.append({
                'protein_id': protein_id,
                'num_sequences': num_sequences,
                'num_residues': num_residues,
                'total_m_elements': m_elements,
                'total_z_elements': z_elements,
                'total_elements': m_elements + z_elements
            })

        except Exception as e:
            print(f"Error processing {protein_id}: {e}")

    print(f"\n📊 PROTEIN SIZE ANALYSIS:")
    print(f"{'Protein':<12} {'Sequences':<10} {'Residues':<9} {'M Elements':<12} {'Z Elements':<12} {'Total':<12}")
    print("-" * 80)

    total_elements_all = 0
    for info in protein_info:
        print(f"{info['protein_id']:<12} {info['num_sequences']:<10} {info['num_residues']:<9} "
              f"{info['total_m_elements']:<12} {info['total_z_elements']:<12} {info['total_elements']:<12}")
        total_elements_all += info['total_elements']

    print(f"\n📈 SCALING IMPLICATIONS:")
    print(f"   Smallest protein: {min(p['total_elements'] for p in protein_info):,} elements")
    print(f"   Largest protein: {max(p['total_elements'] for p in protein_info):,} elements")
    print(
        f"   Size ratio: {max(p['total_elements'] for p in protein_info) / min(p['total_elements'] for p in protein_info):.1f}x")
    print(f"   Total elements across all: {total_elements_all:,}")

    return protein_info


def calculate_proper_loss_normalization(pred_m, target_m, pred_z, target_z):
    """
    Calculate loss with proper normalization across all dimensions
    """
    # Method 1: Per-element loss (most fair)
    msa_loss_per_element = F.mse_loss(pred_m, target_m) / pred_m.numel()
    pair_loss_per_element = F.mse_loss(pred_z, target_z) / pred_z.numel()

    # Method 2: Per-sequence-residue loss (protein-size normalized)
    num_sequences = pred_m.shape[0]
    num_residues = pred_m.shape[1]

    msa_loss_per_seq_res = F.mse_loss(pred_m, target_m) / (num_sequences * num_residues)
    pair_loss_per_res_pair = F.mse_loss(pred_z, target_z) / (num_residues * num_residues)

    # Method 3: Relative loss (scale-invariant)
    msa_relative_loss = F.mse_loss(pred_m, target_m) / (target_m.var() + 1e-8)
    pair_relative_loss = F.mse_loss(pred_z, target_z) / (target_z.var() + 1e-8)

    return {
        'per_element': msa_loss_per_element + pair_loss_per_element,
        'per_seq_res': msa_loss_per_seq_res + pair_loss_per_res_pair,
        'relative': msa_relative_loss + pair_relative_loss,
        'raw': F.mse_loss(pred_m, target_m) + F.mse_loss(pred_z, target_z)
    }


def train_step_with_proper_normalization(protein_id, step_idx, norm_stats, protein_info):
    """
    Training step with size-aware loss normalization
    """
    clear_memory()

    # Get protein size info
    protein_data = next((p for p in protein_info if p['protein_id'] == protein_id), None)
    if not protein_data:
        print(f"Warning: No size info for {protein_id}")
        protein_data = {'total_elements': 1}  # Fallback

    m_init, z_init = load_initial_input_normalized(protein_id, norm_stats)
    ode_state = (m_init, z_init)

    batch_size = args.batch_size
    total_loss = 0
    total_time_steps = 0

    active_ode_func = checkpointed_ode_func if args.use_checkpoint else ode_func

    all_time_batches = [(batch_start, min(batch_start + batch_size, len(t_grid_full)))
                        for batch_start in range(1, len(t_grid_full), batch_size)]

    time_batches = all_time_batches[:1] if args.test_single_step else all_time_batches
    current_state = ode_state

    for batch_idx, (batch_start, batch_end) in enumerate(time_batches):
        start_idx = batch_start - 1 if batch_idx == 0 else batch_start - 1
        t_grid_batch = t_grid_full[start_idx:batch_end]

        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            rtol = 1e-4 if args.reduced_precision_integration else 1e-5
            atol = 1e-5 if args.reduced_precision_integration else 1e-6

            pred_trajectory = odeint(
                active_ode_func,
                current_state,
                t_grid_batch,
                method=args.integrator,
                rtol=rtol,
                atol=atol
            )

            if isinstance(pred_trajectory, tuple):
                current_state = (pred_trajectory[0][-1].detach(), pred_trajectory[1][-1].detach())
            else:
                current_state = (pred_trajectory[0][-1].detach(), pred_trajectory[1][-1].detach())

            batch_loss = 0
            batch_time_steps = 0
            start_i = 1 if batch_idx == 0 else 0

            for i in range(start_i, len(t_grid_batch)):
                time_idx = start_idx + i
                try:
                    gt_m, gt_z = load_block_normalized(protein_id, time_idx, norm_stats)
                except FileNotFoundError:
                    continue

                if isinstance(pred_trajectory, tuple):
                    if i < pred_trajectory[0].shape[0] and i < pred_trajectory[1].shape[0]:
                        pred_m, pred_z = pred_trajectory[0][i], pred_trajectory[1][i]
                    else:
                        continue
                else:
                    if i < pred_trajectory[0].shape[0] and i < pred_trajectory[1].shape[0]:
                        pred_m, pred_z = pred_trajectory[0][i], pred_trajectory[1][i]
                    else:
                        continue

                # Calculate properly normalized loss
                loss_metrics = calculate_proper_loss_normalization(pred_m, gt_m, pred_z, gt_z)

                # Use per-element loss for fair comparison across protein sizes
                step_loss = loss_metrics['per_element']
                batch_loss += step_loss
                batch_time_steps += 1

                # Diagnostic for first protein only
                if step_idx == 0 and time_idx == 1:
                    print(f"🔍 {protein_id} ({protein_data['total_elements']:,} elements):")
                    print(f"   Per-element: {loss_metrics['per_element']:.6f}")
                    print(f"   Raw loss: {loss_metrics['raw']:.3f}")
                    print(f"   Relative: {loss_metrics['relative']:.3f}")

            if batch_time_steps > 0:
                batch_loss = batch_loss / batch_time_steps
                total_loss += batch_loss.item()
                total_time_steps += 1

        # Backprop (same as before)
        if USE_AMP:
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
            if args.gradient_accumulation <= 1 or (step_idx + 1) % args.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
            if args.gradient_accumulation <= 1 or (step_idx + 1) % args.gradient_accumulation == 0:
                optimizer.step()

        del pred_trajectory, batch_loss
        clear_memory()

    del m_init, z_init, ode_state, current_state
    clear_memory()

    return total_loss / max(total_time_steps, 1)


def train_with_size_normalization(input_dir):
    """
    Training with proper size normalization
    """
    # Step 1: Analyze protein sizes
    protein_info = analyze_loss_scaling_factors(input_dir)

    # Step 2: Compute normalization stats
    norm_stats = compute_normalization_stats(input_dir, num_proteins=5)

    # Step 3: Calculate baseline with size normalization
    dataset = get_dataset(input_dir)
    test_protein = dataset[0]

    m_norm, z_norm = load_initial_input_normalized(test_protein, norm_stats)
    m1_norm, z1_norm = load_block_normalized(test_protein, 1, norm_stats)

    baseline_metrics = calculate_proper_loss_normalization(m_norm, m1_norm, z_norm, z1_norm)

    print(f"\n📊 BASELINE LOSS METRICS ({test_protein}):")
    print(f"   Per-element: {baseline_metrics['per_element']:.6f}")
    print(f"   Raw loss: {baseline_metrics['raw']:.3f}")
    print(f"   Relative: {baseline_metrics['relative']:.3f}")

    print(f"\n🚀 Training with size-normalized losses...")

    if args.test_single_step:
        epochs_to_run = 1
        dataset_to_process = dataset[:1]
    else:
        epochs_to_run = args.epochs
        dataset_to_process = dataset

    for epoch in range(epochs_to_run):
        clear_memory()
        epoch_loss = 0
        successful_proteins = 0

        for step_idx, protein_id in enumerate(dataset_to_process):
            try:
                loss = train_step_with_proper_normalization(protein_id, step_idx, norm_stats, protein_info)
                epoch_loss += loss
                successful_proteins += 1
            except RuntimeError as e:
                if device == "cuda" and "CUDA out of memory" in str(e):
                    print(f"❌ OOM: {protein_id}")
                    clear_memory()
                    continue
                else:
                    raise e

        if successful_proteins > 0:
            avg_loss = epoch_loss / successful_proteins
            baseline_per_element = baseline_metrics['per_element']

            print(f"📈 Epoch {epoch + 1}: Per-element Loss = {avg_loss:.6f}")

            # Progress relative to baseline
            if avg_loss < baseline_per_element * 0.5:
                print(
                    f"   ✅ Excellent! {((baseline_per_element - avg_loss) / baseline_per_element * 100):.1f}% better than baseline")
            elif avg_loss < baseline_per_element * 1.2:
                print(f"   ✅ Good - near baseline performance")
            else:
                ratio = avg_loss / baseline_per_element
                print(f"   ⚠️  {ratio:.1f}x worse than baseline")

        if not args.test_single_step and successful_proteins > 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"evoformer_ode_size_normalized_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ode_func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_per_element': avg_loss,
                'normalization_stats': norm_stats,
                'protein_info': protein_info,
                'baseline_metrics': baseline_metrics
            }, checkpoint_path)


if __name__ == "__main__":
    try:
        if args.test_configs:
            print("Use memory_config_tester.py for configuration testing.")
            sys.exit(0)
        else:
            train_with_size_normalization(DATA_DIR)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

print("\n🎯 SIZE-NORMALIZED LOSS CALCULATION:")
print("✅ Loss per element (fair across protein sizes)")
print("✅ Accounts for varying sequence/residue counts")
print("✅ Shows actual per-element error (~1e-5 to 1e-3 range)")
print("✅ Comparable across different proteins and runs")